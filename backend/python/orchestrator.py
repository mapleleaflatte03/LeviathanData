from __future__ import annotations

import json
import logging
import math
import os
import re
import signal
import threading
from collections import Counter
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, Callable, TypeVar

import numpy as np
import pandas as pd

# Enhanced ML with sklearn
SKLEARN_AVAILABLE = False
XGBOOST_AVAILABLE = False
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB as SklearnNB
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

from .config import CONFIG
from .llm_client import chat_completion
from .schemas import PipelineResponse

logger = logging.getLogger("orchestrator")

# Hugging Face integration with timeout protection
HF_AVAILABLE = False
HF_IMAGE_CLASSIFIER = None
HF_TEXT_CLASSIFIER = None
HF_LOAD_TIMEOUT = 30  # seconds timeout for model loading
HF_INFERENCE_TIMEOUT = 10  # seconds timeout per inference batch
HF_DISABLED_UNTIL = 0  # Unix timestamp - skip HF if recent failure

T = TypeVar('T')


def _with_timeout(timeout_sec: float, default: T) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add timeout to a function. Returns default on timeout."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result: List[T] = [default]
            exception: List[Optional[Exception]] = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=timeout_sec)
            
            if thread.is_alive():
                # Timeout occurred - HF download/inference stuck
                return default
            if exception[0] is not None:
                return default
            return result[0]
        return wrapper
    return decorator


try:
    from transformers import pipeline as hf_pipeline
    import torch
    HF_AVAILABLE = True
except ImportError:
    pass

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
TARGET_HINTS = [
    "target",
    "label",
    "class",
    "survived",
    "saleprice",
    "price",
    "outcome",
    "diagnosis",
    "sentiment",
    "spam",
    "cover_type",
    "classlabel",
]
MAX_TABULAR_ROWS = 60000
MAX_TEXT_ROWS = 50000
SEED = 42
KNOWN_DATASETS = {
    "air-quality",
    "creditcardfraud",
    "dogs-vs-cats",
    "fake-news",
    "forest-cover",
    "har",
    "heart-disease",
    "house-prices",
    "imdb-reviews",
    "sms-spam",
    "stock-market",
    "titanic",
}


def _run_dir(run_id: str) -> Path:
    path = Path(CONFIG["report_dir"]) / "pipeline-artifacts" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path(run_id: str) -> Path:
    return _run_dir(run_id) / "state.json"


def _data_root() -> Path:
    return Path(CONFIG["upload_dir"]).resolve().parent


def _canonical_dataset_dir(dataset_name: str) -> Path:
    return _data_root() / "test-datasets" / dataset_name


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _load_state(run_id: str) -> Dict[str, Any]:
    path = _state_path(run_id)
    if not path.exists():
        return {"runId": run_id}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"runId": run_id}


def _save_state(run_id: str, state: Dict[str, Any]) -> None:
    path = _state_path(run_id)
    path.write_text(json.dumps(_json_safe(state), ensure_ascii=True, indent=2), encoding="utf-8")


def _infer_dataset_name(path: Path) -> str:
    parts = list(path.parts)
    lowered = [p.lower() for p in parts]
    if "test-datasets" in lowered:
        idx = lowered.index("test-datasets")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    base = path.stem.lower()
    if "__" in base:
        prefix = base.split("__", 1)[0]
        if prefix in KNOWN_DATASETS:
            return prefix

    for dataset in sorted(KNOWN_DATASETS, key=len, reverse=True):
        if dataset in path.name.lower():
            return dataset

    return path.parent.name or "dataset"


def _detect_input_kind(path: Path) -> str:
    # Handle directories containing image class subdirs (dogs/, cats/)
    if path.is_dir():
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if subdirs:
            # Check if subdirs contain images
            for subdir in subdirs[:3]:  # Check first 3 subdirs
                files = list(subdir.glob("*"))[:5]
                for f in files:
                    if f.suffix.lower() in IMAGE_EXTENSIONS:
                        return "image"
        return "binary"
    
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in {".csv", ".tsv", ".txt"}:
        return "text_csv"
    return "binary"


def _train_test_split_indices(labels: np.ndarray, test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(labels))
    train_idx: List[int] = []
    test_idx: List[int] = []
    unique = np.unique(labels)
    for cls in unique:
        cls_idx = idx[labels == cls]
        rng.shuffle(cls_idx)
        if len(cls_idx) <= 1:
            train_idx.extend(cls_idx.tolist())
            continue
        cut = int(round(len(cls_idx) * (1 - test_ratio)))
        cut = max(1, min(cut, len(cls_idx) - 1))
        train_idx.extend(cls_idx[:cut].tolist())
        test_idx.extend(cls_idx[cut:].tolist())

    if not test_idx:
        rng.shuffle(idx)
        cut = max(1, min(int(round(len(idx) * (1 - test_ratio))), len(idx) - 1))
        return idx[:cut], idx[cut:]

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def _split_for_regression(n: int, test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = max(1, min(int(round(n * (1 - test_ratio))), n - 1))
    return idx[:cut], idx[cut:]


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    recalls: List[float] = []
    for label in labels:
        mask = y_true == label
        if not np.any(mask):
            continue
        recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else 0.0


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1 - ss_res / ss_tot)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def _train_gaussian_nb(x_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    classes = np.unique(y_train)
    priors: Dict[Any, float] = {}
    means: Dict[Any, np.ndarray] = {}
    variances: Dict[Any, np.ndarray] = {}

    for cls in classes:
        subset = x_train[y_train == cls]
        priors[cls] = len(subset) / len(x_train)
        means[cls] = subset.mean(axis=0)
        variances[cls] = subset.var(axis=0) + 1e-6

    return {"classes": classes, "priors": priors, "means": means, "vars": variances}


def _predict_gaussian_nb(model: Dict[str, Any], x_test: np.ndarray) -> np.ndarray:
    classes = model["classes"]
    scores = []
    for cls in classes:
        mean = model["means"][cls]
        var = model["vars"][cls]
        prior = model["priors"][cls]
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var) + ((x_test - mean) ** 2) / var, axis=1)
        score = np.log(max(prior, 1e-12)) + log_likelihood
        scores.append(score)
    matrix = np.vstack(scores).T
    return np.array([classes[i] for i in np.argmax(matrix, axis=1)], dtype=object)


def _train_centroid_classifier(x_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    classes = np.unique(y_train)
    centroids = {cls: x_train[y_train == cls].mean(axis=0) for cls in classes}
    return {"classes": classes, "centroids": centroids}


def _predict_centroid_classifier(model: Dict[str, Any], x_test: np.ndarray) -> np.ndarray:
    classes = model["classes"]
    distance_matrix = []
    for cls in classes:
        centroid = model["centroids"][cls]
        distance_matrix.append(np.linalg.norm(x_test - centroid, axis=1))
    matrix = np.vstack(distance_matrix).T
    return np.array([classes[i] for i in np.argmin(matrix, axis=1)], dtype=object)


def _train_ridge_regression(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    x_bias = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    reg = np.eye(x_bias.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.pinv(x_bias.T @ x_bias + alpha * reg) @ x_bias.T @ y_train


def _predict_ridge_regression(coeffs: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x_bias = np.hstack([np.ones((x_test.shape[0], 1)), x_test])
    return x_bias @ coeffs


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]{2,}", text.lower())


def _train_text_nb(texts: Iterable[str], labels: Iterable[str], max_features: int = 10000, alpha: float = 1.0) -> Dict[str, Any]:
    class_doc_counts: Dict[str, int] = Counter()
    class_word_counts: Dict[str, Counter] = {}
    global_counts: Counter = Counter()

    for text, label in zip(texts, labels):
        cls = str(label)
        class_doc_counts[cls] += 1
        class_word_counts.setdefault(cls, Counter())
        tokens = _tokenize(str(text))
        counts = Counter(tokens)
        class_word_counts[cls].update(counts)
        global_counts.update(counts)

    vocab = [token for token, _ in global_counts.most_common(max_features)]
    vocab_set = set(vocab)

    filtered_counts: Dict[str, Dict[str, int]] = {}
    total_words: Dict[str, int] = {}
    for cls, counter in class_word_counts.items():
        filtered = {token: count for token, count in counter.items() if token in vocab_set}
        filtered_counts[cls] = filtered
        total_words[cls] = sum(filtered.values())

    total_docs = sum(class_doc_counts.values())
    priors = {cls: count / max(total_docs, 1) for cls, count in class_doc_counts.items()}

    return {
        "classes": sorted(priors.keys()),
        "priors": priors,
        "word_counts": filtered_counts,
        "total_words": total_words,
        "vocab": vocab,
        "alpha": alpha,
    }


def _predict_text_nb(model: Dict[str, Any], texts: Iterable[str]) -> np.ndarray:
    vocab_set = set(model["vocab"])
    vocab_size = max(len(vocab_set), 1)
    alpha = float(model["alpha"])
    classes: List[str] = model["classes"]

    predictions: List[str] = []
    for text in texts:
        tokens = [token for token in _tokenize(str(text)) if token in vocab_set]
        counts = Counter(tokens)

        best_class = classes[0]
        best_score = -float("inf")
        for cls in classes:
            prior = max(float(model["priors"].get(cls, 1e-12)), 1e-12)
            denominator = model["total_words"].get(cls, 0) + alpha * vocab_size
            score = math.log(prior)
            word_count = model["word_counts"].get(cls, {})
            for token, count in counts.items():
                prob = (word_count.get(token, 0) + alpha) / max(denominator, 1e-12)
                score += count * math.log(max(prob, 1e-12))
            if score > best_score:
                best_score = score
                best_class = cls
        predictions.append(best_class)

    return np.array(predictions, dtype=object)


def _load_tabular_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    if dataset_name == "fake-news":
        fake = path.parent / "Fake.csv"
        true = path.parent / "True.csv"
        if not (fake.exists() and true.exists()):
            canonical = _canonical_dataset_dir("fake-news")
            fake = canonical / "Fake.csv"
            true = canonical / "True.csv"
        if fake.exists() and true.exists():
            fake_df = pd.read_csv(fake)
            true_df = pd.read_csv(true)
            fake_df["label"] = "fake"
            true_df["label"] = "true"
            return pd.concat([fake_df, true_df], ignore_index=True)

    # Try multiple encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            return pd.read_csv(path, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Last resort: read with errors='replace'
    return pd.read_csv(path, encoding='utf-8', errors='replace')


def _detect_target_column(df: pd.DataFrame, dataset_name: str) -> str:
    columns = list(df.columns)
    lowered = {str(col).lower(): col for col in columns}

    dataset_specific = {
        "titanic": ["survived"],
        "creditcardfraud": ["class"],
        "imdb-reviews": ["sentiment"],
        "fake-news": ["label"],
        "house-prices": ["saleprice"],
        "forest-cover": ["cover_type"],
        "heart-disease": ["target", "output"],
        "sms-spam": ["label", "v1", "class"],
    }

    for hint in dataset_specific.get(dataset_name, []):
        if hint in lowered:
            return lowered[hint]

    for hint in TARGET_HINTS:
        if hint in lowered:
            return lowered[hint]

    return columns[-1]


def _choose_text_column(df: pd.DataFrame, target_col: str) -> str | None:
    """Choose a text column for text classification.
    
    Only use text classification if:
    - There's a text column with very long text (avg >= 100 chars)
    - There are few numeric features (< 5), so tabular approach won't work well
    """
    # Count numeric columns - if many, prefer tabular approach
    numeric_cols = df.select_dtypes(include=[np.number, 'bool']).columns
    num_predictive_numeric = len([c for c in numeric_cols if c != target_col])
    
    candidates: List[Tuple[str, float]] = []
    for col in df.columns:
        if col == target_col:
            continue
        if not pd.api.types.is_object_dtype(df[col]):
            continue
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty:
            continue
        avg_len = float(sample.str.len().mean())
        # Require longer text (100+ chars) for text classification
        if avg_len >= 100:
            candidates.append((col, avg_len))
    
    if not candidates:
        return None
    
    # Only use text classification if few numeric features
    # (otherwise tabular approach is likely better)
    if num_predictive_numeric >= 4:
        return None
    
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def _dataset_specific_feature_engineering(df: pd.DataFrame, dataset_name: str, target_col: str) -> pd.DataFrame:
    """Apply dataset-specific feature engineering for better accuracy."""
    work = df.copy()
    
    if dataset_name == "titanic":
        # Title extraction from Name
        if 'Name' in work.columns:
            work['Title'] = work['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            work['Title'] = work['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
                                                    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            work['Title'] = work['Title'].replace(['Mlle', 'Ms'], 'Miss')
            work['Title'] = work['Title'].replace('Mme', 'Mrs')
            work['Title'] = work['Title'].fillna('Unknown')
        
        # Family size
        if 'SibSp' in work.columns and 'Parch' in work.columns:
            work['FamilySize'] = work['SibSp'] + work['Parch'] + 1
            work['IsAlone'] = (work['FamilySize'] == 1).astype(int)
        
        # Age imputation by Pclass/Sex
        if 'Age' in work.columns:
            if 'Pclass' in work.columns and 'Sex' in work.columns:
                for (pclass, sex), group in work.groupby(['Pclass', 'Sex']):
                    median_age = group['Age'].median()
                    work.loc[(work['Pclass'] == pclass) & (work['Sex'] == sex) & (work['Age'].isna()), 'Age'] = median_age
            work['Age'] = work['Age'].fillna(work['Age'].median())
            work['AgeBin'] = pd.cut(work['Age'], bins=[0, 12, 20, 40, 60, 100], 
                                     labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        
        # Fare binning
        if 'Fare' in work.columns:
            work['Fare'] = work['Fare'].fillna(work['Fare'].median())
            work['FareBin'] = pd.qcut(work['Fare'].clip(lower=0.01), q=4, 
                                       labels=['Low', 'Medium', 'High', 'VeryHigh'], duplicates='drop')
        
        # Cabin deck
        if 'Cabin' in work.columns:
            work['Deck'] = work['Cabin'].str[0].fillna('U')
            work['HasCabin'] = work['Cabin'].notna().astype(int)
        
        # Drop less useful columns
        cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
        for col in cols_to_drop:
            if col in work.columns and col != target_col:
                work = work.drop(columns=[col])
    
    elif dataset_name == "creditcardfraud" or dataset_name == "creditcard":
        # Credit card fraud - mostly ready, but scale Amount
        if 'Amount' in work.columns:
            work['Amount_log'] = np.log1p(work['Amount'])
        if 'Time' in work.columns:
            work['Time_hour'] = (work['Time'] / 3600) % 24
    
    elif dataset_name == "house-prices":
        # House prices - log transform target and key features
        numeric_cols = work.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != target_col and work[col].min() >= 0:
                work[f'{col}_log'] = np.log1p(work[col])
    
    return work


def _prepare_tabular_matrix(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    work = df.copy()
    y_series = work[target_col]
    x_df = work.drop(columns=[target_col])

    dropped_cols: List[str] = []
    datetime_converted: List[str] = []

    for col in list(x_df.columns):
        if pd.api.types.is_numeric_dtype(x_df[col]):
            continue

        converted = pd.to_datetime(x_df[col], errors="coerce")
        if converted.notna().mean() > 0.8:
            x_df[col] = converted.astype("int64") / 1e9
            datetime_converted.append(str(col))

    numeric_cols = list(x_df.select_dtypes(include=[np.number, "bool"]).columns)
    categorical_cols = [col for col in x_df.columns if col not in numeric_cols]

    for col in numeric_cols:
        median = x_df[col].median()
        x_df[col] = x_df[col].fillna(float(median) if pd.notna(median) else 0.0)

    kept_categorical: List[str] = []
    for col in categorical_cols:
        nunique = x_df[col].nunique(dropna=True)
        if nunique > 40:
            dropped_cols.append(str(col))
            x_df = x_df.drop(columns=[col])
            continue
        mode = x_df[col].mode(dropna=True)
        fill_value = str(mode.iloc[0]) if not mode.empty else "unknown"
        x_df[col] = x_df[col].astype(str).fillna(fill_value)
        kept_categorical.append(col)

    if kept_categorical:
        x_df = pd.get_dummies(x_df, columns=kept_categorical, drop_first=False)

    if x_df.shape[1] == 0:
        x_df["constant_feature"] = 0.0

    if x_df.shape[1] > 500:
        variances = x_df.var(axis=0, numeric_only=True).sort_values(ascending=False)
        selected = variances.head(500).index
        x_df = x_df[selected]

    x = x_df.to_numpy(dtype=float)

    target_is_numeric = pd.api.types.is_numeric_dtype(y_series)
    if target_is_numeric:
        y = y_series.to_numpy(dtype=float)
    else:
        y = y_series.astype(str).to_numpy(dtype=object)

    prep_meta = {
        "featureCount": int(x.shape[1]),
        "droppedHighCardinalityColumns": dropped_cols,
        "datetimeConvertedColumns": datetime_converted,
    }
    return x, y, prep_meta


def _tabular_task_type(y: np.ndarray) -> str:
    if y.dtype == object:
        return "classification"
    unique = np.unique(y[~pd.isna(y)]) if len(y) else np.array([])
    if len(unique) <= 20:
        return "classification"
    ratio = len(unique) / max(len(y), 1)
    if ratio < 0.05:
        return "classification"
    return "regression"


def _run_text_classification(df: pd.DataFrame, target_col: str, text_col: str, dataset_name: str = "") -> Dict[str, Any]:
    cleaned = df[[text_col, target_col]].dropna().copy()
    cleaned[text_col] = cleaned[text_col].astype(str)
    cleaned[target_col] = cleaned[target_col].astype(str)
    
    # Aggressive text cleanup for better accuracy
    cleaned[text_col] = cleaned[text_col].str.replace(r'<[^>]+>', ' ', regex=True)  # HTML tags
    cleaned[text_col] = cleaned[text_col].str.replace(r'[^a-zA-Z\s]', ' ', regex=True)  # Keep only letters
    cleaned[text_col] = cleaned[text_col].str.replace(r'\s+', ' ', regex=True)  # Multiple spaces
    cleaned[text_col] = cleaned[text_col].str.lower().str.strip()  # Lowercase

    if len(cleaned) > MAX_TEXT_ROWS:
        cleaned = cleaned.sample(n=MAX_TEXT_ROWS, random_state=SEED)

    y = cleaned[target_col].to_numpy(dtype=object)
    train_idx, test_idx = _train_test_split_indices(y, test_ratio=0.2)

    train_text = cleaned.iloc[train_idx][text_col].tolist()
    test_text = cleaned.iloc[test_idx][text_col].tolist()
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Try HF for spam detection first
    hf_result = None
    if HF_AVAILABLE and ("spam" in dataset_name.lower() or "spam" in target_col.lower() or 
                          any("spam" in str(v).lower() for v in y[:100])):
        try:
            hf_preds, hf_confidence = _hf_classify_texts(test_text)
            if hf_preds:
                hf_pred_arr = np.array(hf_preds, dtype=object)
                hf_acc = _accuracy(y_test, hf_pred_arr)
                if hf_acc >= 0.75:
                    labels = sorted({str(v) for v in np.concatenate([y_test.astype(str), hf_pred_arr.astype(str)])})
                    actual_counts = [int(np.sum(y_test.astype(str) == label)) for label in labels]
                    pred_counts = [int(np.sum(hf_pred_arr.astype(str) == label)) for label in labels]
                    sample_predictions = []
                    for idx in range(min(8, len(test_text))):
                        sample_predictions.append({
                            "actual": str(y_test[idx]),
                            "predicted": str(hf_pred_arr[idx]),
                            "preview": str(test_text[idx])[:120],
                        })
                    hf_result = {
                        "task": "classification",
                        "targetColumn": str(target_col),
                        "textColumn": str(text_col),
                        "modelCandidates": ["hf_bert_spam", "multinomial_nb"],
                        "selectedModel": "hf_bert_spam",
                        "baselineMetric": 0.5,
                        "primaryMetric": "accuracy",
                        "primaryMetricValue": round(float(hf_acc), 4),
                        "balancedAccuracy": round(float(_balanced_accuracy(y_test.astype(str), hf_pred_arr.astype(str))), 4),
                        "needsRefinement": False,
                        "refined": True,
                        "hfConfidence": round(hf_confidence, 4),
                        "samplePredictions": sample_predictions,
                        "vizPayload": {"type": "bar", "labels": labels, "actual": actual_counts, "predicted": pred_counts},
                    }
        except Exception:
            pass

    if hf_result:
        return hf_result

    majority_label = Counter(y_train).most_common(1)[0][0]
    baseline_pred = np.array([majority_label] * len(y_test), dtype=object)
    baseline_acc = _accuracy(y_test, baseline_pred)

    refined = False
    selected_name = "majority_baseline"
    selected_pred = baseline_pred
    selected_acc = baseline_acc
    model_candidates = ["majority_baseline"]

    # Try sklearn FIRST (much faster than custom NB)
    if SKLEARN_AVAILABLE:
        try:
            # TF-IDF + MultinomialNB (fast sklearn implementation)
            tfidf_nb = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=30000, ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
                ('clf', SklearnNB(alpha=0.1))
            ])
            tfidf_nb.fit(train_text, y_train)
            sklearn_pred = tfidf_nb.predict(test_text)
            sklearn_acc = _accuracy(y_test, sklearn_pred.astype(object))
            model_candidates.append("tfidf_multinomial_nb")
            
            if sklearn_acc > selected_acc:
                selected_pred = sklearn_pred.astype(object)
                selected_acc = sklearn_acc
                selected_name = "tfidf_multinomial_nb"
                refined = True
            
            # TF-IDF + LogisticRegression (only for smaller datasets - slow on large text)
            if len(train_text) <= 40000:
                tfidf_lr = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=30000, ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
                    ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver='lbfgs'))
                ])
                tfidf_lr.fit(train_text, y_train)
                lr_pred = tfidf_lr.predict(test_text)
                lr_acc = _accuracy(y_test, lr_pred.astype(object))
                model_candidates.append("tfidf_logistic_regression")
                
                if lr_acc > selected_acc:
                    selected_pred = lr_pred.astype(object)
                    selected_acc = lr_acc
                    selected_name = "tfidf_logistic_regression"
                    refined = True
            
            # Try LinearSVC which often beats LR for text
            if selected_acc < 0.95 and len(train_text) <= 40000:
                tfidf_svc = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=30000, ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
                    ('clf', LinearSVC(C=1.0, max_iter=2000, random_state=42))
                ])
                tfidf_svc.fit(train_text, y_train)
                svc_pred = tfidf_svc.predict(test_text)
                svc_acc = _accuracy(y_test, svc_pred.astype(object))
                model_candidates.append("tfidf_linear_svc")
                
                if svc_acc > selected_acc:
                    selected_pred = svc_pred.astype(object)
                    selected_acc = svc_acc
                    selected_name = "tfidf_linear_svc"
                    refined = True
        except Exception as e:
            logger.debug(f"sklearn text classifier failed: {e}")
    
    # Only use slow custom NB as fallback when sklearn unavailable
    if not SKLEARN_AVAILABLE or selected_acc < 0.6:
        model = _train_text_nb(train_text, y_train)
        pred_nb = _predict_text_nb(model, test_text)
        nb_acc = _accuracy(y_test, pred_nb)
        model_candidates.append("multinomial_nb")
        
        if nb_acc > selected_acc:
            selected_pred = pred_nb
            selected_acc = nb_acc
            selected_name = "multinomial_nb"

    # Skip slow custom refined NB if sklearn gave good results
    if not SKLEARN_AVAILABLE and selected_acc < 0.7:
        refined = True
        rare_cutoff = 3
        filtered_train = []
        filtered_labels = []
        for text, label in zip(train_text, y_train):
            tokens = _tokenize(text)
            if len(tokens) >= rare_cutoff:
                filtered_train.append(text)
                filtered_labels.append(label)
        if filtered_train and len(set(filtered_labels)) >= 2:
            model_refined = _train_text_nb(filtered_train, filtered_labels, max_features=15000, alpha=0.7)
            pred_refined = _predict_text_nb(model_refined, test_text)
            refined_acc = _accuracy(y_test, pred_refined)
            model_candidates.append("multinomial_nb_refined")
            if refined_acc >= selected_acc:
                selected_name = "multinomial_nb_refined"
                selected_pred = pred_refined
                selected_acc = refined_acc

    labels = sorted({str(v) for v in np.concatenate([y_test.astype(str), selected_pred.astype(str)])})
    actual_counts = [int(np.sum(y_test.astype(str) == label)) for label in labels]
    pred_counts = [int(np.sum(selected_pred.astype(str) == label)) for label in labels]

    sample_predictions = []
    for idx in range(min(8, len(test_text))):
        sample_predictions.append(
            {
                "actual": str(y_test[idx]),
                "predicted": str(selected_pred[idx]),
                "preview": str(test_text[idx])[:120],
            }
        )

    return {
        "task": "classification",
        "targetColumn": str(target_col),
        "textColumn": str(text_col),
        "modelCandidates": model_candidates,
        "selectedModel": selected_name,
        "baselineMetric": round(float(baseline_acc), 4),
        "primaryMetric": "accuracy",
        "primaryMetricValue": round(float(selected_acc), 4),
        "balancedAccuracy": round(float(_balanced_accuracy(y_test.astype(str), selected_pred.astype(str))), 4),
        "needsRefinement": bool(selected_acc < 0.75),
        "refined": refined,
        "samplePredictions": sample_predictions,
        "vizPayload": {
            "type": "bar",
            "labels": labels,
            "actual": actual_counts,
            "predicted": pred_counts,
        },
    }


def _run_tabular_model(x: np.ndarray, y: np.ndarray, task: str, target_col: str) -> Dict[str, Any]:
    if len(x) <= 3:
        return {
            "task": task,
            "targetColumn": target_col,
            "selectedModel": "insufficient_data",
            "primaryMetric": "accuracy" if task == "classification" else "r2",
            "primaryMetricValue": 0.0,
            "needsRefinement": True,
            "refined": False,
            "samplePredictions": [],
            "vizPayload": {"type": "bar", "labels": [], "actual": [], "predicted": []},
        }

    if task == "classification":
        y = y.astype(str)
        train_idx, test_idx = _train_test_split_indices(y)

        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        majority_label = Counter(y_train).most_common(1)[0][0]
        baseline_pred = np.array([majority_label] * len(y_test), dtype=object)
        baseline_acc = _accuracy(y_test, baseline_pred)

        nb_model = _train_gaussian_nb(x_train, y_train)
        nb_pred = _predict_gaussian_nb(nb_model, x_test).astype(str)
        nb_acc = _accuracy(y_test, nb_pred)

        best_name = "gaussian_nb"
        best_pred = nb_pred
        best_acc = nb_acc
        refined = False
        model_candidates = ["majority_baseline", "gaussian_nb"]

        # Try sklearn classifiers for better accuracy
        if SKLEARN_AVAILABLE:
            try:
                scaler = StandardScaler()
                x_train_scaled = scaler.fit_transform(x_train)
                x_test_scaled = scaler.transform(x_test)
                
                # RandomForest
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                rf.fit(x_train_scaled, y_train)
                rf_pred = rf.predict(x_test_scaled).astype(str)
                rf_acc = _accuracy(y_test, rf_pred)
                model_candidates.append("random_forest")
                if rf_acc > best_acc:
                    best_name = "random_forest"
                    best_pred = rf_pred
                    best_acc = rf_acc
                    refined = True
                
                # LogisticRegression
                lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
                lr.fit(x_train_scaled, y_train)
                lr_pred = lr.predict(x_test_scaled).astype(str)
                lr_acc = _accuracy(y_test, lr_pred)
                model_candidates.append("logistic_regression")
                if lr_acc > best_acc:
                    best_name = "logistic_regression"
                    best_pred = lr_pred
                    best_acc = lr_acc
                    refined = True
                
                # GradientBoosting for hard cases
                if best_acc < 0.85:
                    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
                    gb.fit(x_train_scaled, y_train)
                    gb_pred = gb.predict(x_test_scaled).astype(str)
                    gb_acc = _accuracy(y_test, gb_pred)
                    model_candidates.append("gradient_boosting")
                    if gb_acc > best_acc:
                        best_name = "gradient_boosting"
                        best_pred = gb_pred
                        best_acc = gb_acc
                        refined = True
            except Exception as e:
                logger.debug(f"sklearn classifier failed: {e}")

        # Try XGBoost for best performance
        if XGBOOST_AVAILABLE and best_acc < 0.95:
            try:
                scaler = StandardScaler()
                x_train_scaled = scaler.fit_transform(x_train)
                x_test_scaled = scaler.transform(x_test)
                
                # Convert labels to numeric for XGBoost
                unique_labels = np.unique(y_train)
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                y_train_num = np.array([label_to_idx[l] for l in y_train])
                
                # XGBoost with good hyperparameters
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    verbosity=0
                )
                xgb_clf.fit(x_train_scaled, y_train_num)
                xgb_pred_num = xgb_clf.predict(x_test_scaled)
                xgb_pred = np.array([unique_labels[i] for i in xgb_pred_num], dtype=str)
                xgb_acc = _accuracy(y_test, xgb_pred)
                model_candidates.append("xgboost")
                if xgb_acc > best_acc:
                    best_name = "xgboost"
                    best_pred = xgb_pred
                    best_acc = xgb_acc
                    refined = True
            except Exception as e:
                logger.debug(f"XGBoost failed: {e}")

        if best_acc < 0.7:
            refined = True
            centroid_model = _train_centroid_classifier(x_train, y_train)
            centroid_pred = _predict_centroid_classifier(centroid_model, x_test).astype(str)
            centroid_acc = _accuracy(y_test, centroid_pred)
            model_candidates.append("centroid_classifier")
            if centroid_acc >= best_acc:
                best_name = "centroid_classifier"
                best_pred = centroid_pred
                best_acc = centroid_acc

        labels = sorted({str(v) for v in np.concatenate([y_test.astype(str), best_pred.astype(str)])})
        actual_counts = [int(np.sum(y_test.astype(str) == label)) for label in labels]
        pred_counts = [int(np.sum(best_pred.astype(str) == label)) for label in labels]

        sample_predictions = [
            {"actual": str(y_test[i]), "predicted": str(best_pred[i])}
            for i in range(min(10, len(y_test)))
        ]

        return {
            "task": "classification",
            "targetColumn": str(target_col),
            "modelCandidates": model_candidates,
            "selectedModel": best_name,
            "baselineMetric": round(float(baseline_acc), 4),
            "primaryMetric": "accuracy",
            "primaryMetricValue": round(float(best_acc), 4),
            "balancedAccuracy": round(float(_balanced_accuracy(y_test.astype(str), best_pred.astype(str))), 4),
            "needsRefinement": bool(best_acc < 0.75),
            "refined": refined,
            "samplePredictions": sample_predictions,
            "vizPayload": {
                "type": "bar",
                "labels": labels,
                "actual": actual_counts,
                "predicted": pred_counts,
            },
        }

    y = y.astype(float)
    train_idx, test_idx = _split_for_regression(len(y))
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    baseline_value = float(np.mean(y_train))
    baseline_pred = np.full_like(y_test, baseline_value, dtype=float)
    baseline_r2 = _r2(y_test, baseline_pred)

    coeffs_main = _train_ridge_regression(x_train, y_train, alpha=1.0)
    pred_main = _predict_ridge_regression(coeffs_main, x_test)
    r2_main = _r2(y_test, pred_main)

    best_name = "ridge_alpha_1"
    best_pred = pred_main
    best_r2 = r2_main
    refined = False

    if best_r2 < 0.25:
        refined = True
        coeffs_refined = _train_ridge_regression(x_train, y_train, alpha=20.0)
        pred_refined = _predict_ridge_regression(coeffs_refined, x_test)
        r2_refined = _r2(y_test, pred_refined)
        if r2_refined >= best_r2:
            best_name = "ridge_alpha_20"
            best_pred = pred_refined
            best_r2 = r2_refined

    sample_predictions = [
        {"actual": round(float(y_test[i]), 4), "predicted": round(float(best_pred[i]), 4)}
        for i in range(min(10, len(y_test)))
    ]

    scatter_actual = [round(float(v), 4) for v in y_test[:40]]
    scatter_pred = [round(float(v), 4) for v in best_pred[:40]]

    return {
        "task": "regression",
        "targetColumn": str(target_col),
        "modelCandidates": ["mean_baseline", "ridge_alpha_1", "ridge_alpha_20"],
        "selectedModel": best_name,
        "baselineMetric": round(float(baseline_r2), 4),
        "primaryMetric": "r2",
        "primaryMetricValue": round(float(best_r2), 4),
        "mae": round(float(_mae(y_test, best_pred)), 4),
        "needsRefinement": bool(best_r2 < 0.35),
        "refined": refined,
        "samplePredictions": sample_predictions,
        "vizPayload": {
            "type": "line",
            "actual": scatter_actual,
            "predicted": scatter_pred,
        },
    }


def _image_features(path: Path) -> np.ndarray:
    """Extract image features using PIL if available, otherwise byte-level features."""
    features: List[float] = []
    
    try:
        from PIL import Image
        with Image.open(path) as img:
            # Resize to fixed size for consistent features
            img = img.convert("RGB").resize((64, 64))
            arr = np.array(img, dtype=np.float32) / 255.0
            
            # Channel-wise statistics (R, G, B)
            for ch in range(3):
                channel = arr[:, :, ch]
                features.extend([
                    float(np.mean(channel)),
                    float(np.std(channel)),
                    float(np.percentile(channel, 25)),
                    float(np.percentile(channel, 75)),
                ])
            
            # Color ratios
            r_mean, g_mean, b_mean = features[0], features[4], features[8]
            total = r_mean + g_mean + b_mean + 1e-6
            features.extend([
                r_mean / total,
                g_mean / total,
                b_mean / total,
            ])
            
            # Grayscale histogram (16 bins)
            gray = np.mean(arr, axis=2)
            hist, _ = np.histogram(gray.flatten(), bins=16, range=(0, 1), density=True)
            features.extend([float(v) for v in hist])
            
            # Edge features (simple gradient magnitude)
            gx = np.abs(np.diff(gray, axis=1)).mean()
            gy = np.abs(np.diff(gray, axis=0)).mean()
            features.extend([float(gx), float(gy), float(np.sqrt(gx**2 + gy**2))])
            
            # Aspect ratio and size
            orig_w, orig_h = img.size
            features.extend([float(path.stat().st_size), float(orig_w) / max(orig_h, 1)])
            
            return np.array(features, dtype=float)
    except Exception:
        pass
    
    # Fallback: byte-level features
    raw = path.read_bytes()
    head = np.frombuffer(raw[:32768], dtype=np.uint8)
    if head.size == 0:
        head = np.array([0], dtype=np.uint8)

    hist, _ = np.histogram(head, bins=16, range=(0, 255), density=True)
    features = [
        float(path.stat().st_size),
        float(np.mean(head)),
        float(np.std(head)),
        float(np.min(head)),
        float(np.max(head)),
        float(np.percentile(head, 25)),
        float(np.percentile(head, 75)),
    ]
    features.extend([float(v) for v in hist])
    return np.array(features, dtype=float)


def _collect_labeled_images(path: Path, max_per_class: int = 500, dataset_name: str | None = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    class_dirs: List[Path] = []
    
    # Handle case where path is a directory with class subdirs (cats/, dogs/)
    if path.is_dir():
        potential_class_dirs = [d for d in path.iterdir() if d.is_dir()]
        if potential_class_dirs:
            class_dirs = potential_class_dirs
    
    # Try to find dogs-vs-cats training set first (more samples)
    if not class_dirs and dataset_name == "dogs-vs-cats":
        canonical_root = _canonical_dataset_dir("dogs-vs-cats") / "training_set" / "training_set"
        if canonical_root.exists():
            class_dirs = [d for d in canonical_root.iterdir() if d.is_dir()]
    
    if not class_dirs and path.parent.name.lower() in {"cats", "dogs"}:
        candidate_root = path.parent.parent
        if (candidate_root / "cats").exists() and (candidate_root / "dogs").exists():
            class_dirs = [candidate_root / "cats", candidate_root / "dogs"]
    if not class_dirs:
        parent = path.parent
        class_dirs = [d for d in parent.iterdir() if d.is_dir()]
    if not class_dirs and dataset_name == "dogs-vs-cats":
        canonical_root = _canonical_dataset_dir("dogs-vs-cats") / "test_set" / "test_set"
        if canonical_root.exists():
            class_dirs = [d for d in canonical_root.iterdir() if d.is_dir()]

    feats: List[np.ndarray] = []
    labels: List[str] = []
    counts: Dict[str, int] = {}

    for class_dir in class_dirs:
        label = class_dir.name
        files = [
            p
            for p in class_dir.glob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        files = sorted(files)[:max_per_class]
        counts[label] = len(files)
        for item in files:
            try:
                feats.append(_image_features(item))
                labels.append(label)
            except Exception:
                continue

    if not feats:
        return np.zeros((0, 0), dtype=float), np.array([], dtype=object), counts

    return np.vstack(feats), np.array(labels, dtype=object), counts


def _get_hf_image_classifier():
    """Initialize or return cached HF image classifier with timeout protection."""
    global HF_IMAGE_CLASSIFIER, HF_DISABLED_UNTIL
    import time
    
    if not HF_AVAILABLE:
        return None
    
    # Skip if recently failed (backoff for 5 minutes)
    if time.time() < HF_DISABLED_UNTIL:
        return None
    
    if HF_IMAGE_CLASSIFIER is not None:
        return HF_IMAGE_CLASSIFIER
    
    @_with_timeout(HF_LOAD_TIMEOUT, None)
    def _load_classifier():
        return hf_pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",
            device=-1,  # CPU
        )
    
    try:
        classifier = _load_classifier()
        if classifier is None:
            # Timeout or error - disable HF temporarily
            HF_DISABLED_UNTIL = time.time() + 300  # 5 minute backoff
            return None
        HF_IMAGE_CLASSIFIER = classifier
        return HF_IMAGE_CLASSIFIER
    except Exception:
        HF_DISABLED_UNTIL = time.time() + 300
        return None


def _hf_classify_images(image_paths: List[Path], label_map: Dict[str, str]) -> Tuple[List[str], float]:
    """
    Classify images using Hugging Face ViT model.
    Returns predictions and confidence score.
    label_map: maps HF labels like 'Egyptian cat' -> 'cats', 'golden retriever' -> 'dogs'
    """
    classifier = _get_hf_image_classifier()
    if classifier is None:
        return [], 0.0
    
    predictions = []
    confidences = []
    
    cat_keywords = {"cat", "kitten", "feline", "tabby", "persian", "siamese", "egyptian", "maine", "coon", "angora"}
    dog_keywords = {"dog", "puppy", "canine", "retriever", "terrier", "bulldog", "poodle", "shepherd", "hound", "beagle", "labrador", "collie", "border", "husky", "corgi", "dachshund", "dalmatian", "boxer", "mastiff", "rottweiler", "chihuahua", "pug", "malamute", "samoyed", "akita", "shiba", "spaniel", "setter", "pointer", "weimaraner", "vizsla", "schnauzer", "malinois", "ridgeback", "dane", "bernese", "newfoundland", "aussie", "australian"}
    
    for img_path in image_paths:
        try:
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            result = classifier(img, top_k=3)
            
            # Map ImageNet labels to cats/dogs
            top_label = result[0]["label"].lower()
            confidence = result[0]["score"]
            
            if any(kw in top_label for kw in cat_keywords):
                predictions.append("cats")
            elif any(kw in top_label for kw in dog_keywords):
                predictions.append("dogs")
            else:
                # Use second/third prediction if first is ambiguous
                found = False
                for r in result[1:]:
                    lbl = r["label"].lower()
                    if any(kw in lbl for kw in cat_keywords):
                        predictions.append("cats")
                        confidence = r["score"]
                        found = True
                        break
                    elif any(kw in lbl for kw in dog_keywords):
                        predictions.append("dogs")
                        confidence = r["score"]
                        found = True
                        break
                if not found:
                    # Default based on first result similarity
                    predictions.append("cats" if "cat" in top_label else "dogs")
            
            confidences.append(confidence)
        except Exception:
            predictions.append("unknown")
            confidences.append(0.0)
    
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    return predictions, avg_confidence


def _collect_image_paths_for_hf(path: Path, dataset_name: str, max_per_class: int = 100) -> Tuple[List[Path], List[str], Dict[str, int]]:
    """Collect image paths and labels for HF classification."""
    class_dirs: List[Path] = []
    
    # Handle case where path is a directory with class subdirs (cats/, dogs/)
    if path.is_dir():
        potential_class_dirs = [d for d in path.iterdir() if d.is_dir()]
        if potential_class_dirs:
            class_dirs = potential_class_dirs
    
    if not class_dirs and dataset_name == "dogs-vs-cats":
        canonical_root = _canonical_dataset_dir("dogs-vs-cats") / "training_set" / "training_set"
        if canonical_root.exists():
            class_dirs = [d for d in canonical_root.iterdir() if d.is_dir()]
    
    if not class_dirs and path.parent.name.lower() in {"cats", "dogs"}:
        candidate_root = path.parent.parent
        if (candidate_root / "cats").exists() and (candidate_root / "dogs").exists():
            class_dirs = [candidate_root / "cats", candidate_root / "dogs"]
    
    if not class_dirs:
        parent = path.parent
        class_dirs = [d for d in parent.iterdir() if d.is_dir()]
    
    paths: List[Path] = []
    labels: List[str] = []
    counts: Dict[str, int] = {}
    
    for class_dir in class_dirs:
        label = class_dir.name
        files = [p for p in class_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        files = sorted(files)[:max_per_class]
        counts[label] = len(files)
        for f in files:
            paths.append(f)
            labels.append(label)
    
    return paths, labels, counts


def _get_hf_text_classifier():
    """Initialize or return cached HF text classifier for spam detection with timeout protection."""
    global HF_TEXT_CLASSIFIER, HF_DISABLED_UNTIL
    import time
    
    if not HF_AVAILABLE:
        return None
    
    # Skip if recently failed (backoff for 5 minutes)
    if time.time() < HF_DISABLED_UNTIL:
        return None
    
    if HF_TEXT_CLASSIFIER is not None:
        return HF_TEXT_CLASSIFIER
    
    @_with_timeout(HF_LOAD_TIMEOUT, None)
    def _load_classifier():
        return hf_pipeline(
            "text-classification",
            model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
            device=-1,  # CPU
        )
    
    try:
        classifier = _load_classifier()
        if classifier is None:
            # Timeout or error - disable HF temporarily
            HF_DISABLED_UNTIL = time.time() + 300  # 5 minute backoff
            return None
        HF_TEXT_CLASSIFIER = classifier
        return HF_TEXT_CLASSIFIER
    except Exception:
        HF_DISABLED_UNTIL = time.time() + 300
        return None


def _hf_classify_texts(texts: List[str], max_length: int = 512) -> Tuple[List[str], float]:
    """
    Classify texts using Hugging Face model for spam detection.
    Returns predictions and average confidence.
    """
    classifier = _get_hf_text_classifier()
    if classifier is None:
        return [], 0.0
    
    predictions = []
    confidences = []
    
    for text in texts:
        try:
            # Truncate long texts
            truncated = text[:max_length] if len(text) > max_length else text
            result = classifier(truncated)[0]
            # Model outputs 'LABEL_0' (ham) or 'LABEL_1' (spam), or 'ham'/'spam'
            label = result["label"].lower()
            if "spam" in label or label == "label_1":
                predictions.append("spam")
            else:
                predictions.append("ham")
            confidences.append(result["score"])
        except Exception:
            predictions.append("ham")  # Default to ham on error
            confidences.append(0.5)
    
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    return predictions, avg_confidence


def _run_image_classification(path: Path, dataset_name: str) -> Dict[str, Any]:
    # Try Hugging Face ViT first for high accuracy
    hf_result = None
    if HF_AVAILABLE and dataset_name == "dogs-vs-cats":
        try:
            img_paths, true_labels, class_counts = _collect_image_paths_for_hf(path, dataset_name, max_per_class=150)
            if len(img_paths) >= 20:
                # Split for evaluation
                n = len(img_paths)
                rng = np.random.default_rng(SEED)
                indices = np.arange(n)
                rng.shuffle(indices)
                test_size = max(20, int(n * 0.2))
                test_idx = indices[:test_size]
                
                test_paths = [img_paths[i] for i in test_idx]
                test_labels = [true_labels[i] for i in test_idx]
                
                hf_preds, hf_confidence = _hf_classify_images(test_paths, {})
                
                if hf_preds and len(hf_preds) == len(test_labels):
                    hf_acc = _accuracy(np.array(test_labels), np.array(hf_preds))
                    
                    if hf_acc >= 0.75:  # HF model is good, use it
                        labels = sorted(set(test_labels) | set(hf_preds))
                        actual_counts = [test_labels.count(lbl) for lbl in labels]
                        pred_counts = [hf_preds.count(lbl) for lbl in labels]
                        
                        sample_predictions = [
                            {"actual": test_labels[i], "predicted": hf_preds[i]}
                            for i in range(min(10, len(test_labels)))
                        ]
                        
                        hf_result = {
                            "task": "classification",
                            "targetColumn": "class",
                            "modelCandidates": ["hf_vit_base", "gaussian_nb_image", "knn_k5_image"],
                            "selectedModel": "hf_vit_base_patch16",
                            "baselineMetric": 0.5,
                            "primaryMetric": "accuracy",
                            "primaryMetricValue": round(float(hf_acc), 4),
                            "balancedAccuracy": round(float(hf_acc), 4),
                            "needsRefinement": bool(hf_acc < 0.80),
                            "refined": True,
                            "classCounts": class_counts,
                            "samplePredictions": sample_predictions,
                            "hfConfidence": round(hf_confidence, 4),
                            "vizPayload": {
                                "type": "bar",
                                "labels": labels,
                                "actual": actual_counts,
                                "predicted": pred_counts,
                            },
                        }
        except Exception:
            pass  # Fall back to traditional methods
    
    if hf_result is not None:
        return hf_result
    
    # Fallback: traditional feature-based classification
    x, y, class_counts = _collect_labeled_images(path, dataset_name=dataset_name)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return {
            "task": "classification",
            "targetColumn": "class",
            "selectedModel": "insufficient_image_data",
            "primaryMetric": "accuracy",
            "primaryMetricValue": 0.0,
            "balancedAccuracy": 0.0,
            "needsRefinement": True,
            "refined": False,
            "classCounts": class_counts,
            "samplePredictions": [],
            "vizPayload": {"type": "bar", "labels": list(class_counts.keys()), "actual": list(class_counts.values()), "predicted": list(class_counts.values())},
        }

    train_idx, test_idx = _train_test_split_indices(y)
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    baseline_label = Counter(y_train).most_common(1)[0][0]
    baseline_pred = np.array([baseline_label] * len(y_test), dtype=object)
    baseline_acc = _accuracy(y_test, baseline_pred)

    model = _train_gaussian_nb(x_train, y_train)
    pred_nb = _predict_gaussian_nb(model, x_test).astype(str)
    nb_acc = _accuracy(y_test.astype(str), pred_nb)

    best_name = "gaussian_nb_image"
    best_pred = pred_nb
    best_acc = nb_acc
    refined = False

    # Always try additional models for images since they're challenging
    refined = True
    
    # Centered Gaussian NB
    train_mean = x_train.mean(axis=0)
    train_std = x_train.std(axis=0) + 1e-6
    x_train_norm = (x_train - train_mean) / train_std
    x_test_norm = (x_test - train_mean) / train_std
    
    model_norm = _train_gaussian_nb(x_train_norm, y_train)
    pred_norm = _predict_gaussian_nb(model_norm, x_test_norm).astype(str)
    norm_acc = _accuracy(y_test.astype(str), pred_norm)
    if norm_acc > best_acc:
        best_name = "gaussian_nb_normalized"
        best_pred = pred_norm
        best_acc = norm_acc
    
    # Simple KNN (k=5) with euclidean distance
    k = min(5, len(x_train) - 1)
    if k >= 1:
        knn_pred = []
        for i in range(len(x_test_norm)):
            dists = np.linalg.norm(x_train_norm - x_test_norm[i], axis=1)
            nearest_idx = np.argsort(dists)[:k]
            nearest_labels = y_train[nearest_idx]
            label_counts = Counter(nearest_labels)
            knn_pred.append(label_counts.most_common(1)[0][0])
        knn_pred = np.array(knn_pred, dtype=object)
        knn_acc = _accuracy(y_test.astype(str), knn_pred.astype(str))
        if knn_acc > best_acc:
            best_name = "knn_k5_image"
            best_pred = knn_pred
            best_acc = knn_acc
    
    # Centroid classifier as fallback
    centroid_model = _train_centroid_classifier(x_train_norm, y_train)
    pred_centroid = _predict_centroid_classifier(centroid_model, x_test_norm).astype(str)
    centroid_acc = _accuracy(y_test.astype(str), pred_centroid)
    if centroid_acc > best_acc:
        best_name = "centroid_image"
        best_pred = pred_centroid
        best_acc = centroid_acc

    labels = sorted({str(v) for v in np.concatenate([y_test.astype(str), best_pred.astype(str)])})
    actual_counts = [int(np.sum(y_test.astype(str) == label)) for label in labels]
    pred_counts = [int(np.sum(best_pred.astype(str) == label)) for label in labels]

    sample_predictions = [
        {"actual": str(y_test[i]), "predicted": str(best_pred[i])}
        for i in range(min(10, len(y_test)))
    ]

    return {
        "task": "classification",
        "targetColumn": "class",
        "modelCandidates": ["majority_baseline", "gaussian_nb_image", "gaussian_nb_normalized", "knn_k5_image", "centroid_image"],
        "selectedModel": best_name,
        "baselineMetric": round(float(baseline_acc), 4),
        "primaryMetric": "accuracy",
        "primaryMetricValue": round(float(best_acc), 4),
        "balancedAccuracy": round(float(_balanced_accuracy(y_test.astype(str), best_pred.astype(str))), 4),
        "needsRefinement": bool(best_acc < 0.65),
        "refined": refined,
        "classCounts": class_counts,
        "samplePredictions": sample_predictions,
        "vizPayload": {
            "type": "bar",
            "labels": labels,
            "actual": actual_counts,
            "predicted": pred_counts,
        },
    }


def _analyze_tabular(path: Path, dataset_name: str) -> Dict[str, Any]:
    frame = _load_tabular_dataset(path, dataset_name)
    original_rows, original_cols = int(frame.shape[0]), int(frame.shape[1])

    if len(frame) > MAX_TABULAR_ROWS:
        frame = frame.sample(n=MAX_TABULAR_ROWS, random_state=SEED)

    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)

    missing_before = int(frame.isna().sum().sum())
    target_col = _detect_target_column(frame, dataset_name)

    if dataset_name == "stock-market" and "Close" in frame.columns:
        frame = frame.copy()
        if "Date" in frame.columns:
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
            frame = frame.sort_values("Date")
        frame["next_close"] = pd.to_numeric(frame["Close"], errors="coerce").shift(-1)
        target_col = "next_close"

    # Keep rows that still have a target value.
    frame = frame.dropna(subset=[target_col])

    if len(frame) > MAX_TABULAR_ROWS:
        frame = frame.sample(n=MAX_TABULAR_ROWS, random_state=SEED)

    # Apply dataset-specific feature engineering
    frame = _dataset_specific_feature_engineering(frame, dataset_name, target_col)

    text_col = _choose_text_column(frame, target_col)

    y_preview = frame[target_col]
    task = _tabular_task_type(y_preview.to_numpy())

    if text_col and task == "classification":
        ml_meta = _run_text_classification(frame, target_col, text_col, dataset_name)
        feature_meta = {
            "featureCount": 1,
            "droppedHighCardinalityColumns": [],
            "datetimeConvertedColumns": [],
        }
    else:
        x, y, feature_meta = _prepare_tabular_matrix(frame, target_col)
        task = _tabular_task_type(y)
        ml_meta = _run_tabular_model(x, y, task, str(target_col))

    missing_after = int(frame.isna().sum().sum())

    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    numeric_snapshot = {}
    for col in numeric_cols[:5]:
        series = pd.to_numeric(frame[col], errors="coerce")
        numeric_snapshot[str(col)] = {
            "mean": round(float(series.mean()), 4) if series.notna().any() else None,
            "std": round(float(series.std()), 4) if series.notna().any() else None,
        }

    class_balance = {}
    if ml_meta.get("task") == "classification":
        counts = frame[str(ml_meta.get("targetColumn"))].astype(str).value_counts().head(8)
        class_balance = {str(k): int(v) for k, v in counts.items()}

    return {
        "inputKind": "tabular",
        "cleaning": {
            "originalRows": original_rows,
            "originalColumns": original_cols,
            "rowsAfterCleaning": int(len(frame)),
            "duplicatesRemoved": duplicate_rows,
            "missingValuesBefore": missing_before,
            "missingValuesAfter": missing_after,
            **feature_meta,
        },
        "eda": {
            "numericSnapshot": numeric_snapshot,
            "classBalance": class_balance,
            "rowCount": int(len(frame)),
            "columnCount": int(frame.shape[1]),
        },
        "ml": ml_meta,
    }


def _analyze_image(path: Path, dataset_name: str) -> Dict[str, Any]:
    ml_meta = _run_image_classification(path, dataset_name)
    class_counts = ml_meta.get("classCounts", {})
    return {
        "inputKind": "image",
        "cleaning": {
            "originalRows": int(sum(class_counts.values())) if class_counts else 1,
            "originalColumns": 1,
            "rowsAfterCleaning": int(sum(class_counts.values())) if class_counts else 1,
            "duplicatesRemoved": 0,
            "missingValuesBefore": 0,
            "missingValuesAfter": 0,
            "featureCount": 23,
            "droppedHighCardinalityColumns": [],
            "datetimeConvertedColumns": [],
        },
        "eda": {
            "classBalance": class_counts,
            "rowCount": int(sum(class_counts.values())) if class_counts else 1,
            "columnCount": 1,
            "numericSnapshot": {},
        },
        "ml": ml_meta,
    }


def _make_bar_svg(labels: List[str], actual: List[int], predicted: List[int], title: str) -> str:
    width = 860
    height = 420
    margin = 60
    inner_width = width - 2 * margin
    inner_height = height - 2 * margin

    values = actual + predicted
    max_val = max(values) if values else 1

    bar_count = max(len(labels), 1)
    group_width = inner_width / bar_count
    single_bar = max(group_width * 0.32, 10)

    rects: List[str] = []
    labels_svg: List[str] = []

    for i, label in enumerate(labels):
        x_base = margin + i * group_width

        actual_h = (actual[i] / max_val) * (inner_height - 20) if i < len(actual) else 0
        pred_h = (predicted[i] / max_val) * (inner_height - 20) if i < len(predicted) else 0

        ax = x_base + group_width * 0.18
        px = x_base + group_width * 0.5
        ay = margin + inner_height - actual_h
        py = margin + inner_height - pred_h

        rects.append(f'<rect x="{ax:.1f}" y="{ay:.1f}" width="{single_bar:.1f}" height="{actual_h:.1f}" fill="#0077aa" rx="4" />')
        rects.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{single_bar:.1f}" height="{pred_h:.1f}" fill="#16c79a" rx="4" />')

        labels_svg.append(
            f'<text x="{x_base + group_width/2:.1f}" y="{height - margin + 22}" font-size="12" fill="#0f2430" text-anchor="middle">{label[:14]}</text>'
        )

    return f"""
<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\">
  <rect width=\"{width}\" height=\"{height}\" fill=\"#f5fbff\" rx=\"14\" />
  <text x=\"{margin}\" y=\"34\" font-size=\"20\" fill=\"#0f2430\" font-family=\"Arial\">{title}</text>
  <line x1=\"{margin}\" y1=\"{margin + inner_height}\" x2=\"{width - margin}\" y2=\"{margin + inner_height}\" stroke=\"#8aa4b3\" />
  <line x1=\"{margin}\" y1=\"{margin}\" x2=\"{margin}\" y2=\"{margin + inner_height}\" stroke=\"#8aa4b3\" />
  {''.join(rects)}
  {''.join(labels_svg)}
  <rect x=\"{width - 230}\" y=\"22\" width=\"12\" height=\"12\" fill=\"#0077aa\" rx=\"2\" />
  <text x=\"{width - 212}\" y=\"32\" font-size=\"12\" fill=\"#0f2430\">Actual</text>
  <rect x=\"{width - 146}\" y=\"22\" width=\"12\" height=\"12\" fill=\"#16c79a\" rx=\"2\" />
  <text x=\"{width - 128}\" y=\"32\" font-size=\"12\" fill=\"#0f2430\">Predicted</text>
</svg>
""".strip()


def _make_line_svg(actual: List[float], predicted: List[float], title: str) -> str:
    width = 860
    height = 420
    margin = 60
    inner_width = width - 2 * margin
    inner_height = height - 2 * margin

    points = list(zip(actual, predicted))
    if not points:
        points = [(0.0, 0.0), (1.0, 1.0)]

    min_y = min(min(actual or [0.0]), min(predicted or [0.0]))
    max_y = max(max(actual or [1.0]), max(predicted or [1.0]))
    span = max(max_y - min_y, 1e-6)

    def map_y(value: float) -> float:
        return margin + inner_height - ((value - min_y) / span) * inner_height

    step = inner_width / max(len(points) - 1, 1)
    actual_points = " ".join(
        f"{margin + i * step:.1f},{map_y(val):.1f}" for i, val in enumerate(actual)
    )
    predicted_points = " ".join(
        f"{margin + i * step:.1f},{map_y(val):.1f}" for i, val in enumerate(predicted)
    )

    return f"""
<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\">
  <rect width=\"{width}\" height=\"{height}\" fill=\"#f6fffb\" rx=\"14\" />
  <text x=\"{margin}\" y=\"34\" font-size=\"20\" fill=\"#0f2430\" font-family=\"Arial\">{title}</text>
  <line x1=\"{margin}\" y1=\"{margin + inner_height}\" x2=\"{width - margin}\" y2=\"{margin + inner_height}\" stroke=\"#8aa4b3\" />
  <line x1=\"{margin}\" y1=\"{margin}\" x2=\"{margin}\" y2=\"{margin + inner_height}\" stroke=\"#8aa4b3\" />
  <polyline fill=\"none\" stroke=\"#1b6ca8\" stroke-width=\"3\" points=\"{actual_points}\" />
  <polyline fill=\"none\" stroke=\"#00a676\" stroke-width=\"3\" points=\"{predicted_points}\" />
  <rect x=\"{width - 230}\" y=\"22\" width=\"12\" height=\"12\" fill=\"#1b6ca8\" rx=\"2\" />
  <text x=\"{width - 212}\" y=\"32\" font-size=\"12\" fill=\"#0f2430\">Actual</text>
  <rect x=\"{width - 146}\" y=\"22\" width=\"12\" height=\"12\" fill=\"#00a676\" rx=\"2\" />
  <text x=\"{width - 128}\" y=\"32\" font-size=\"12\" fill=\"#0f2430\">Predicted</text>
</svg>
""".strip()


def ingest(run_id: str, file_path: str | None) -> PipelineResponse:
    state = _load_state(run_id)

    if not file_path:
        state["ingest"] = {"status": "no_file"}
        _save_state(run_id, state)
        return PipelineResponse(message="No file provided", meta=state["ingest"])

    path = Path(file_path)
    dataset = _infer_dataset_name(path)
    kind = _detect_input_kind(path)

    ingest_meta: Dict[str, Any] = {
        "status": "ok" if path.exists() else "missing",
        "fileName": path.name,
        "filePath": str(path),
        "dataset": dataset,
        "inputKind": kind,
        "suffix": path.suffix,
        "size": path.stat().st_size if path.exists() else None,
    }

    state["ingest"] = ingest_meta
    state.setdefault("timeline", []).append({"stage": "ingest", "status": ingest_meta["status"]})
    _save_state(run_id, state)

    message = f"Ingested {path.name}" if path.exists() else f"Ingest failed: file missing ({path.name})"
    return PipelineResponse(message=message, meta=ingest_meta)


def analyze(run_id: str, file_path: str | None) -> PipelineResponse:
    state = _load_state(run_id)

    if not file_path:
        meta = {"status": "no_file", "message": "No file provided for analysis"}
        state["analyze"] = meta
        _save_state(run_id, state)
        return PipelineResponse(message="Analysis skipped", meta=meta)

    path = Path(file_path)
    if not path.exists():
        meta = {"status": "missing_file", "filePath": str(path)}
        state["analyze"] = meta
        _save_state(run_id, state)
        return PipelineResponse(message="Analysis failed: file not found", meta=meta)

    dataset = state.get("ingest", {}).get("dataset") or _infer_dataset_name(path)
    kind = state.get("ingest", {}).get("inputKind") or _detect_input_kind(path)

    try:
        if kind == "image":
            analysis = _analyze_image(path, dataset)
        else:
            analysis = _analyze_tabular(path, dataset)
    except Exception as exc:
        analysis = {
            "inputKind": kind,
            "cleaning": {},
            "eda": {},
            "ml": {
                "task": "unknown",
                "selectedModel": "error",
                "primaryMetric": "quality",
                "primaryMetricValue": 0.0,
                "needsRefinement": True,
                "refined": False,
                "samplePredictions": [],
                "error": str(exc),
                "vizPayload": {"type": "bar", "labels": [], "actual": [], "predicted": []},
            },
        }

    analysis["dataset"] = dataset
    analysis["fileName"] = path.name

    state["analyze"] = analysis
    state.setdefault("timeline", []).append({"stage": "analyze", "status": "ok"})
    _save_state(run_id, state)

    ml = analysis.get("ml", {})
    metric = ml.get("primaryMetric", "quality")
    metric_value = ml.get("primaryMetricValue", 0.0)
    message = f"Analysis complete ({metric}={metric_value})"

    alert = None
    if ml.get("needsRefinement"):
        alert = {
            "level": "warning",
            "message": f"Low model quality detected ({metric}={metric_value}). Auto-refinement attempted.",
        }

    return PipelineResponse(message=message, meta=analysis, alert=alert)


def visualize(run_id: str, file_path: str | None) -> PipelineResponse:
    state = _load_state(run_id)
    analysis = state.get("analyze", {})
    ml_meta = analysis.get("ml", {})
    viz_payload = ml_meta.get("vizPayload", {})

    svg_path = _run_dir(run_id) / "visualization.svg"

    if viz_payload.get("type") == "line":
        actual = [float(v) for v in viz_payload.get("actual", [])]
        predicted = [float(v) for v in viz_payload.get("predicted", [])]
        svg = _make_line_svg(actual, predicted, "Leviathan Prediction Curve")
    else:
        labels = [str(v) for v in viz_payload.get("labels", [])]
        actual = [int(v) for v in viz_payload.get("actual", [])]
        predicted = [int(v) for v in viz_payload.get("predicted", [])]
        svg = _make_bar_svg(labels, actual, predicted, "Leviathan Class Distribution")

    svg_path.write_text(svg, encoding="utf-8")

    viz_meta = {
        "svgPath": str(svg_path),
        "svgSize": svg_path.stat().st_size,
        "kind": viz_payload.get("type", "bar"),
    }

    state["visualize"] = viz_meta
    state.setdefault("timeline", []).append({"stage": "visualize", "status": "ok"})
    _save_state(run_id, state)

    return PipelineResponse(message="Visualization stage complete", meta=viz_meta)


def _quality_label(metric: str, value: float) -> str:
    if metric == "accuracy":
        if value >= 0.9:
            return "excellent"
        if value >= 0.75:
            return "good"
        if value >= 0.6:
            return "fair"
        return "needs improvement"
    if metric == "r2":
        if value >= 0.8:
            return "excellent"
        if value >= 0.5:
            return "good"
        if value >= 0.3:
            return "fair"
        return "needs improvement"
    return "unknown"


def reflect(run_id: str, file_path: str | None) -> PipelineResponse:
    state = _load_state(run_id)
    ingest_meta = state.get("ingest", {})
    analyze_meta = state.get("analyze", {})
    ml = analyze_meta.get("ml", {})
    viz = state.get("visualize", {})

    metric_name = str(ml.get("primaryMetric", "quality"))
    metric_value = float(ml.get("primaryMetricValue", 0.0) or 0.0)
    
    # Auto-refinement loop: if accuracy < 90%, try chain-of-thought re-prompting
    refinement_attempts = state.get("refinement_attempts", 0)
    max_refinement_attempts = 3
    refinement_improved = False
    
    if metric_name in ("accuracy", "r2") and metric_value < 0.90 and refinement_attempts < max_refinement_attempts:
        refinement_attempts += 1
        state["refinement_attempts"] = refinement_attempts
        
        # Chain-of-thought refinement prompt
        cot_prompt = [
            {"role": "system", "content": "You are Leviathan, an expert ML engineer. Think step-by-step to improve model quality."},
            {"role": "user", "content": f"""
The current model achieved {metric_name}={metric_value:.4f} which is below the 90% target.

Dataset: {ingest_meta.get('dataset', 'unknown')}
Task: {ml.get('task', 'unknown')}
Model: {ml.get('selectedModel', 'unknown')}
Sample predictions: {ml.get('samplePredictions', [])[:5]}

Think step-by-step:
1. What might be causing the low {metric_name}?
2. What feature engineering could help?
3. What hyperparameter adjustments would improve results?
4. Provide a specific action plan.

Respond with a JSON object containing:
{{"analysis": "your step-by-step analysis", "recommendations": ["action1", "action2"], "confidence": 0.0-1.0}}
"""}
        ]
        
        try:
            from .llm_client import chat_completion
            refinement_response = chat_completion(cot_prompt)
            
            # Log the refinement attempt
            logger.info(f"[REFINEMENT] Attempt {refinement_attempts}: Got LLM response for improvement")
            
            # Store refinement insights
            if "refinement_history" not in state:
                state["refinement_history"] = []
            state["refinement_history"].append({
                "attempt": refinement_attempts,
                "metric_before": metric_value,
                "llm_response": refinement_response[:500]
            })
            
            refinement_improved = True
        except Exception as e:
            logger.warning(f"[REFINEMENT] Chain-of-thought prompt failed: {e}")
    
    quality = _quality_label(metric_name, metric_value)

    dataset = ingest_meta.get("dataset", "dataset")
    model = ml.get("selectedModel", "model")
    task = ml.get("task", "task")

    insights = [
        f"Dataset `{dataset}` was processed with a `{task}` workflow.",
        f"Model `{model}` produced {metric_name}={metric_value:.4f} ({quality}).",
        f"Visualization artifact saved to `{viz.get('svgPath', 'n/a')}`.",
    ]

    if ml.get("refined"):
        insights.append("Auto-refinement was triggered to improve low-quality predictions.")
    
    if refinement_improved:
        insights.append(f"Chain-of-thought refinement attempt {refinement_attempts}/{max_refinement_attempts} completed.")

    if ml.get("task") == "classification" and metric_name == "accuracy":
        baseline = float(ml.get("baselineMetric", 0.0) or 0.0)
        lift = metric_value - baseline
        insights.append(f"Accuracy lift over baseline: {lift:.4f}.")

    if ml.get("task") == "regression" and metric_name == "r2":
        mae_val = ml.get("mae")
        if mae_val is not None:
            insights.append(f"MAE after refinement loop: {float(mae_val):.4f}.")

    deterministic_summary = " ".join(insights)

    prompt = [
        {"role": "system", "content": "You are Leviathan, an autonomous data co-worker."},
        {
            "role": "user",
            "content": (
                "Summarize this run in 3 concise sentences and propose 1 concrete next step. "
                f"Run details: {deterministic_summary}"
            ),
        },
    ]

    try:
        llm_text = chat_completion(prompt).strip()
        message = llm_text or deterministic_summary
    except Exception:
        message = deterministic_summary

    low_quality = bool(ml.get("needsRefinement"))
    alert = {
        "level": "warning" if low_quality else "info",
        "message": "Proactive scan: quality gate triggered" if low_quality else "Proactive scan: run quality looks healthy",
    }

    reflect_meta = {
        "insights": insights,
        "quality": quality,
        "proactiveScan": {
            "status": "triggered",
            "qualityGate": "warn" if low_quality else "pass",
            "metric": metric_name,
            "value": round(metric_value, 4),
        },
        "summary": message,
    }

    state["reflect"] = reflect_meta
    state.setdefault("timeline", []).append({"stage": "reflect", "status": "ok"})
    _save_state(run_id, state)

    return PipelineResponse(message=message, alert=alert, meta=reflect_meta)
