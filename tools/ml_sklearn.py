"""Tool: ml_sklearn
Scikit-learn machine learning toolkit.

Supported operations:
- train: Train classification/regression models
- predict: Make predictions with trained model
- evaluate: Evaluate model performance
- preprocess: Data preprocessing (scale, encode, impute)
- cluster: Clustering algorithms
- decompose: Dimensionality reduction (PCA, t-SNE)
"""
from typing import Any, Dict, List, Optional, Union
import json
import pickle
import base64
from pathlib import Path


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


sklearn = _optional_import("sklearn")
np = _optional_import("numpy")
joblib = _optional_import("joblib")

# Lazy imports for sklearn modules
_sklearn_modules = {}


def _get_sklearn_module(name: str):
    """Lazily import sklearn submodules."""
    if sklearn is None:
        raise ImportError("scikit-learn is not installed")
    
    if name not in _sklearn_modules:
        if name == "linear_model":
            from sklearn import linear_model
            _sklearn_modules[name] = linear_model
        elif name == "ensemble":
            from sklearn import ensemble
            _sklearn_modules[name] = ensemble
        elif name == "svm":
            from sklearn import svm
            _sklearn_modules[name] = svm
        elif name == "tree":
            from sklearn import tree
            _sklearn_modules[name] = tree
        elif name == "neighbors":
            from sklearn import neighbors
            _sklearn_modules[name] = neighbors
        elif name == "naive_bayes":
            from sklearn import naive_bayes
            _sklearn_modules[name] = naive_bayes
        elif name == "cluster":
            from sklearn import cluster
            _sklearn_modules[name] = cluster
        elif name == "decomposition":
            from sklearn import decomposition
            _sklearn_modules[name] = decomposition
        elif name == "preprocessing":
            from sklearn import preprocessing
            _sklearn_modules[name] = preprocessing
        elif name == "metrics":
            from sklearn import metrics
            _sklearn_modules[name] = metrics
        elif name == "model_selection":
            from sklearn import model_selection
            _sklearn_modules[name] = model_selection
        elif name == "impute":
            from sklearn import impute
            _sklearn_modules[name] = impute
    
    return _sklearn_modules.get(name)


# Model registry
MODEL_CLASSES = {
    # Classification
    "logistic_regression": ("linear_model", "LogisticRegression"),
    "random_forest_classifier": ("ensemble", "RandomForestClassifier"),
    "gradient_boosting_classifier": ("ensemble", "GradientBoostingClassifier"),
    "svc": ("svm", "SVC"),
    "decision_tree_classifier": ("tree", "DecisionTreeClassifier"),
    "knn_classifier": ("neighbors", "KNeighborsClassifier"),
    "naive_bayes": ("naive_bayes", "GaussianNB"),
    # Regression
    "linear_regression": ("linear_model", "LinearRegression"),
    "ridge": ("linear_model", "Ridge"),
    "lasso": ("linear_model", "Lasso"),
    "random_forest_regressor": ("ensemble", "RandomForestRegressor"),
    "gradient_boosting_regressor": ("ensemble", "GradientBoostingRegressor"),
    "svr": ("svm", "SVR"),
    "decision_tree_regressor": ("tree", "DecisionTreeRegressor"),
    "knn_regressor": ("neighbors", "KNeighborsRegressor"),
    # Clustering
    "kmeans": ("cluster", "KMeans"),
    "dbscan": ("cluster", "DBSCAN"),
    "agglomerative": ("cluster", "AgglomerativeClustering"),
    # Decomposition
    "pca": ("decomposition", "PCA"),
    "tsne": ("sklearn.manifold", "TSNE"),
}


def _get_model_class(model_name: str):
    """Get model class by name."""
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}")
    
    module_name, class_name = MODEL_CLASSES[model_name]
    module = _get_sklearn_module(module_name)
    return getattr(module, class_name)


def _train(
    X: List[List[float]],
    y: List[Any],
    model_name: str,
    model_params: Optional[Dict] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train a model and return metrics."""
    model_selection = _get_sklearn_module("model_selection")
    metrics = _get_sklearn_module("metrics")
    
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=random_state
    )
    
    # Create and train model
    model_class = _get_model_class(model_name)
    model_params = model_params or {}
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    # Determine if classification or regression
    is_classifier = hasattr(model, "predict_proba") or model_name in [
        "logistic_regression", "random_forest_classifier", "gradient_boosting_classifier",
        "svc", "decision_tree_classifier", "knn_classifier", "naive_bayes"
    ]
    
    if is_classifier:
        result_metrics = {
            "accuracy": float(metrics.accuracy_score(y_test, y_pred)),
            "precision": float(metrics.precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(metrics.recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(metrics.f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }
    else:
        result_metrics = {
            "mse": float(metrics.mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),
            "mae": float(metrics.mean_absolute_error(y_test, y_pred)),
            "r2": float(metrics.r2_score(y_test, y_pred)),
        }
    
    # Serialize model
    model_bytes = pickle.dumps(model)
    model_b64 = base64.b64encode(model_bytes).decode("utf-8")
    
    return {
        "model_name": model_name,
        "metrics": result_metrics,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "model_b64": model_b64,
    }


def _predict(model_b64: str, X: List[List[float]]) -> Dict[str, Any]:
    """Make predictions with a serialized model."""
    model_bytes = base64.b64decode(model_b64)
    model = pickle.loads(model_bytes)
    
    X_arr = np.array(X)
    predictions = model.predict(X_arr).tolist()
    
    result = {"predictions": predictions}
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_arr).tolist()
        result["probabilities"] = probabilities
    
    return result


def _preprocess(
    X: List[List[float]],
    operations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply preprocessing operations."""
    preprocessing = _get_sklearn_module("preprocessing")
    impute = _get_sklearn_module("impute")
    
    X_arr = np.array(X)
    transformers = []
    
    for op in operations:
        op_type = op.get("type")
        
        if op_type == "standard_scale":
            scaler = preprocessing.StandardScaler()
            X_arr = scaler.fit_transform(X_arr)
            transformers.append(("standard_scaler", scaler))
        
        elif op_type == "minmax_scale":
            feature_range = tuple(op.get("feature_range", (0, 1)))
            scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
            X_arr = scaler.fit_transform(X_arr)
            transformers.append(("minmax_scaler", scaler))
        
        elif op_type == "robust_scale":
            scaler = preprocessing.RobustScaler()
            X_arr = scaler.fit_transform(X_arr)
            transformers.append(("robust_scaler", scaler))
        
        elif op_type == "normalize":
            norm = op.get("norm", "l2")
            X_arr = preprocessing.normalize(X_arr, norm=norm)
        
        elif op_type == "impute":
            strategy = op.get("strategy", "mean")
            imputer = impute.SimpleImputer(strategy=strategy)
            X_arr = imputer.fit_transform(X_arr)
            transformers.append(("imputer", imputer))
        
        elif op_type == "polynomial":
            degree = op.get("degree", 2)
            poly = preprocessing.PolynomialFeatures(degree=degree)
            X_arr = poly.fit_transform(X_arr)
            transformers.append(("polynomial", poly))
    
    return {
        "X_transformed": X_arr.tolist(),
        "shape": list(X_arr.shape),
    }


def _cluster(
    X: List[List[float]],
    algorithm: str = "kmeans",
    params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Perform clustering."""
    metrics = _get_sklearn_module("metrics")
    
    X_arr = np.array(X)
    params = params or {}
    
    model_class = _get_model_class(algorithm)
    model = model_class(**params)
    
    labels = model.fit_predict(X_arr)
    
    result = {
        "labels": labels.tolist(),
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
    }
    
    if hasattr(model, "cluster_centers_"):
        result["centers"] = model.cluster_centers_.tolist()
    
    if hasattr(model, "inertia_"):
        result["inertia"] = float(model.inertia_)
    
    # Silhouette score if more than 1 cluster
    if len(set(labels)) > 1:
        result["silhouette_score"] = float(metrics.silhouette_score(X_arr, labels))
    
    return result


def _decompose(
    X: List[List[float]],
    algorithm: str = "pca",
    n_components: int = 2,
    params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Perform dimensionality reduction."""
    decomposition = _get_sklearn_module("decomposition")
    
    X_arr = np.array(X)
    params = params or {}
    params["n_components"] = n_components
    
    if algorithm == "pca":
        model = decomposition.PCA(**params)
    elif algorithm == "tsne":
        from sklearn.manifold import TSNE
        model = TSNE(**params)
    else:
        raise ValueError(f"Unknown decomposition algorithm: {algorithm}")
    
    X_transformed = model.fit_transform(X_arr)
    
    result = {
        "X_transformed": X_transformed.tolist(),
        "shape": list(X_transformed.shape),
    }
    
    if hasattr(model, "explained_variance_ratio_"):
        result["explained_variance_ratio"] = model.explained_variance_ratio_.tolist()
        result["total_explained_variance"] = float(sum(model.explained_variance_ratio_))
    
    if hasattr(model, "components_"):
        result["components"] = model.components_.tolist()
    
    return result


def _cross_validate(
    X: List[List[float]],
    y: List[Any],
    model_name: str,
    model_params: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = "accuracy",
) -> Dict[str, Any]:
    """Perform cross-validation."""
    model_selection = _get_sklearn_module("model_selection")
    
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    model_class = _get_model_class(model_name)
    model_params = model_params or {}
    model = model_class(**model_params)
    
    scores = model_selection.cross_val_score(model, X_arr, y_arr, cv=cv, scoring=scoring)
    
    return {
        "scores": scores.tolist(),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "cv_folds": cv,
        "scoring": scoring,
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run sklearn operations.
    
    Args:
        args: Dictionary with:
            - operation: "train", "predict", "preprocess", "cluster", "decompose", "cross_validate"
            - X: Feature data
            - y: Target data (for supervised learning)
            - model_name: Name of model to use
            - model_params: Model hyperparameters
    
    Returns:
        Result dictionary with operation output
    """
    args = args or {}
    operation = args.get("operation", "train")
    
    if sklearn is None:
        return {"tool": "ml_sklearn", "status": "error", "error": "scikit-learn not installed"}
    
    try:
        if operation == "train":
            result = _train(
                X=args.get("X", []),
                y=args.get("y", []),
                model_name=args.get("model_name", "random_forest_classifier"),
                model_params=args.get("model_params"),
                test_size=args.get("test_size", 0.2),
                random_state=args.get("random_state", 42),
            )
        
        elif operation == "predict":
            result = _predict(
                model_b64=args.get("model_b64", ""),
                X=args.get("X", []),
            )
        
        elif operation == "preprocess":
            result = _preprocess(
                X=args.get("X", []),
                operations=args.get("operations", []),
            )
        
        elif operation == "cluster":
            result = _cluster(
                X=args.get("X", []),
                algorithm=args.get("algorithm", "kmeans"),
                params=args.get("params"),
            )
        
        elif operation == "decompose":
            result = _decompose(
                X=args.get("X", []),
                algorithm=args.get("algorithm", "pca"),
                n_components=args.get("n_components", 2),
                params=args.get("params"),
            )
        
        elif operation == "cross_validate":
            result = _cross_validate(
                X=args.get("X", []),
                y=args.get("y", []),
                model_name=args.get("model_name", "random_forest_classifier"),
                model_params=args.get("model_params"),
                cv=args.get("cv", 5),
                scoring=args.get("scoring", "accuracy"),
            )
        
        elif operation == "list_models":
            result = {"models": list(MODEL_CLASSES.keys())}
        
        else:
            return {"tool": "ml_sklearn", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "ml_sklearn", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "ml_sklearn", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "train_classifier": {
            "operation": "train",
            "X": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]],
            "y": [0, 0, 0, 0, 1, 1, 1, 1],
            "model_name": "random_forest_classifier",
            "model_params": {"n_estimators": 100, "max_depth": 5},
        },
        "cluster": {
            "operation": "cluster",
            "X": [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]],
            "algorithm": "kmeans",
            "params": {"n_clusters": 2},
        },
        "preprocess": {
            "operation": "preprocess",
            "X": [[1, 100], [2, 200], [3, 300]],
            "operations": [
                {"type": "standard_scale"},
            ],
        },
        "pca": {
            "operation": "decompose",
            "X": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            "algorithm": "pca",
            "n_components": 2,
        },
    }
