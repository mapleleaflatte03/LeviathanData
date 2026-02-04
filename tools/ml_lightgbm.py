"""Tool: ml_lightgbm
LightGBM gradient boosting for fast ML.

Supported operations:
- train: Train LightGBM model
- predict: Make predictions
- cross_validate: Cross-validation
- feature_importance: Get feature importance
- save_model: Save model
- load_model: Load model
"""
from typing import Any, Dict, List, Optional, Union
import json
import os


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


lgb = _optional_import("lightgbm")
np = _optional_import("numpy")

# Model cache
_model_cache: Dict[str, Any] = {}


def _train(
    X: List[List[float]],
    y: List[float],
    task: str = "classification",
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 100,
    early_stopping_rounds: Optional[int] = None,
    eval_set: Optional[tuple] = None,
    feature_names: Optional[List[str]] = None,
    model_name: str = "default",
) -> Dict[str, Any]:
    """Train a LightGBM model."""
    if lgb is None:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
    
    X_arr = np.array(X) if np else X
    y_arr = np.array(y) if np else y
    
    # Default parameters
    default_params = {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": num_boost_round,
        "verbosity": -1,
    }
    
    if task == "classification":
        n_classes = len(set(y))
        if n_classes == 2:
            default_params["objective"] = "binary"
            default_params["metric"] = "binary_logloss"
        else:
            default_params["objective"] = "multiclass"
            default_params["metric"] = "multi_logloss"
            default_params["num_class"] = n_classes
    else:
        default_params["objective"] = "regression"
        default_params["metric"] = "rmse"
    
    if params:
        default_params.update(params)
    
    train_data = lgb.Dataset(X_arr, label=y_arr, feature_name=feature_names)
    
    valid_sets = [train_data]
    valid_names = ["train"]
    
    if eval_set:
        X_val, y_val = eval_set
        valid_data = lgb.Dataset(
            np.array(X_val) if np else X_val,
            label=np.array(y_val) if np else y_val,
            feature_name=feature_names,
        )
        valid_sets.append(valid_data)
        valid_names.append("eval")
    
    callbacks = []
    if early_stopping_rounds:
        callbacks.append(lgb.early_stopping(early_stopping_rounds))
    callbacks.append(lgb.log_evaluation(period=0))  # Suppress logging
    
    evals_result = {}
    
    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    
    _model_cache[model_name] = model
    
    return {
        "model_name": model_name,
        "task": task,
        "num_features": model.num_feature(),
        "best_iteration": model.best_iteration,
        "feature_names": model.feature_name(),
    }


def _predict(
    X: List[List[float]],
    model_name: str = "default",
    num_iteration: Optional[int] = None,
) -> Dict[str, Any]:
    """Make predictions."""
    if lgb is None:
        raise ImportError("lightgbm is not installed")
    
    if model_name not in _model_cache:
        return {"error": f"Model '{model_name}' not found"}
    
    model = _model_cache[model_name]
    X_arr = np.array(X) if np else X
    
    predictions = model.predict(X_arr, num_iteration=num_iteration)
    
    return {
        "predictions": predictions.tolist() if hasattr(predictions, "tolist") else list(predictions),
        "count": len(predictions),
    }


def _cross_validate(
    X: List[List[float]],
    y: List[float],
    task: str = "classification",
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 100,
    nfold: int = 5,
    stratified: bool = True,
) -> Dict[str, Any]:
    """Perform cross-validation."""
    if lgb is None:
        raise ImportError("lightgbm is not installed")
    
    X_arr = np.array(X) if np else X
    y_arr = np.array(y) if np else y
    
    default_params = {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "verbosity": -1,
    }
    
    if task == "classification":
        n_classes = len(set(y))
        if n_classes == 2:
            default_params["objective"] = "binary"
            default_params["metric"] = "binary_logloss"
        else:
            default_params["objective"] = "multiclass"
            default_params["metric"] = "multi_logloss"
            default_params["num_class"] = n_classes
    else:
        default_params["objective"] = "regression"
        default_params["metric"] = "rmse"
    
    if params:
        default_params.update(params)
    
    train_data = lgb.Dataset(X_arr, label=y_arr)
    
    cv_results = lgb.cv(
        default_params,
        train_data,
        num_boost_round=num_boost_round,
        nfold=nfold,
        stratified=stratified,
        return_cvbooster=False,
    )
    
    results = {k: list(v) for k, v in cv_results.items()}
    
    return {
        "cv_results": results,
        "nfold": nfold,
        "num_boost_round": num_boost_round,
    }


def _feature_importance(
    model_name: str = "default",
    importance_type: str = "gain",
) -> Dict[str, Any]:
    """Get feature importance."""
    if lgb is None:
        raise ImportError("lightgbm is not installed")
    
    if model_name not in _model_cache:
        return {"error": f"Model '{model_name}' not found"}
    
    model = _model_cache[model_name]
    importance = model.feature_importance(importance_type=importance_type)
    feature_names = model.feature_name()
    
    importance_dict = dict(zip(feature_names, importance.tolist()))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "importance": dict(sorted_importance),
        "importance_type": importance_type,
    }


def _save_model(
    model_name: str = "default",
    file_path: str = "model.txt",
) -> Dict[str, Any]:
    """Save model to file."""
    if lgb is None:
        raise ImportError("lightgbm is not installed")
    
    if model_name not in _model_cache:
        return {"error": f"Model '{model_name}' not found"}
    
    model = _model_cache[model_name]
    model.save_model(file_path)
    
    return {"saved_to": file_path}


def _load_model(
    file_path: str,
    model_name: str = "default",
) -> Dict[str, Any]:
    """Load model from file."""
    if lgb is None:
        raise ImportError("lightgbm is not installed")
    
    model = lgb.Booster(model_file=file_path)
    _model_cache[model_name] = model
    
    return {
        "loaded": model_name,
        "num_features": model.num_feature(),
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run LightGBM operations."""
    args = args or {}
    operation = args.get("operation", "train")
    
    if lgb is None:
        return {
            "tool": "ml_lightgbm",
            "status": "error",
            "error": "lightgbm not installed. Run: pip install lightgbm",
        }
    
    try:
        if operation == "train":
            result = _train(
                X=args.get("X", []),
                y=args.get("y", []),
                task=args.get("task", "classification"),
                params=args.get("params"),
                num_boost_round=args.get("num_boost_round", 100),
                early_stopping_rounds=args.get("early_stopping_rounds"),
                eval_set=args.get("eval_set"),
                feature_names=args.get("feature_names"),
                model_name=args.get("model_name", "default"),
            )
        
        elif operation == "predict":
            result = _predict(
                X=args.get("X", []),
                model_name=args.get("model_name", "default"),
                num_iteration=args.get("num_iteration"),
            )
        
        elif operation == "cross_validate":
            result = _cross_validate(
                X=args.get("X", []),
                y=args.get("y", []),
                task=args.get("task", "classification"),
                params=args.get("params"),
                num_boost_round=args.get("num_boost_round", 100),
                nfold=args.get("nfold", 5),
                stratified=args.get("stratified", True),
            )
        
        elif operation == "feature_importance":
            result = _feature_importance(
                model_name=args.get("model_name", "default"),
                importance_type=args.get("importance_type", "gain"),
            )
        
        elif operation == "save_model":
            result = _save_model(
                model_name=args.get("model_name", "default"),
                file_path=args.get("file_path", "model.txt"),
            )
        
        elif operation == "load_model":
            result = _load_model(
                file_path=args.get("file_path", ""),
                model_name=args.get("model_name", "default"),
            )
        
        else:
            return {"tool": "ml_lightgbm", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "ml_lightgbm", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "ml_lightgbm", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "train": {
            "operation": "train",
            "X": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "y": [0, 0, 1, 1],
            "task": "classification",
            "num_boost_round": 50,
        },
        "predict": {
            "operation": "predict",
            "X": [[2, 3], [6, 7]],
        },
    }
