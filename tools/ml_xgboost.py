"""Tool: ml_xgboost
XGBoost gradient boosting for high-performance ML.

Supported operations:
- train: Train XGBoost model
- predict: Make predictions
- cross_validate: Cross-validation
- feature_importance: Get feature importance
- save_model: Save model to file
- load_model: Load model from file
- tune: Hyperparameter tuning
"""
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import base64
import io
import os


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


xgb = _optional_import("xgboost")
np = _optional_import("numpy")
sklearn = _optional_import("sklearn")

# Model cache
_model_cache: Dict[str, Any] = {}


def _train(
    X: List[List[float]],
    y: List[float],
    task: str = "classification",
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 100,
    early_stopping_rounds: Optional[int] = None,
    eval_set: Optional[Tuple[List[List[float]], List[float]]] = None,
    model_name: str = "default",
) -> Dict[str, Any]:
    """Train an XGBoost model."""
    if xgb is None:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")
    
    X_arr = np.array(X) if np else X
    y_arr = np.array(y) if np else y
    
    # Default parameters
    default_params = {
        "max_depth": 6,
        "eta": 0.3,
        "eval_metric": "logloss" if task == "classification" else "rmse",
    }
    
    if task == "classification":
        # Check if binary or multiclass
        n_classes = len(set(y))
        if n_classes == 2:
            default_params["objective"] = "binary:logistic"
        else:
            default_params["objective"] = "multi:softprob"
            default_params["num_class"] = n_classes
    else:
        default_params["objective"] = "reg:squarederror"
    
    if params:
        default_params.update(params)
    
    dtrain = xgb.DMatrix(X_arr, label=y_arr)
    
    evals = [(dtrain, "train")]
    if eval_set:
        X_val, y_val = eval_set
        dval = xgb.DMatrix(np.array(X_val) if np else X_val, label=np.array(y_val) if np else y_val)
        evals.append((dval, "eval"))
    
    evals_result = {}
    
    model = xgb.train(
        default_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False,
    )
    
    _model_cache[model_name] = model
    
    return {
        "model_name": model_name,
        "task": task,
        "num_features": model.num_features(),
        "best_iteration": model.best_iteration if hasattr(model, "best_iteration") else num_boost_round,
        "best_score": model.best_score if hasattr(model, "best_score") else None,
        "evals_result": {k: {m: list(v) for m, v in d.items()} for k, d in evals_result.items()},
    }


def _predict(
    X: List[List[float]],
    model_name: str = "default",
    output_margin: bool = False,
) -> Dict[str, Any]:
    """Make predictions with trained model."""
    if xgb is None:
        raise ImportError("xgboost is not installed")
    
    if model_name not in _model_cache:
        return {"error": f"Model '{model_name}' not found. Train a model first."}
    
    model = _model_cache[model_name]
    X_arr = np.array(X) if np else X
    dtest = xgb.DMatrix(X_arr)
    
    predictions = model.predict(dtest, output_margin=output_margin)
    
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
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Perform cross-validation."""
    if xgb is None:
        raise ImportError("xgboost is not installed")
    
    X_arr = np.array(X) if np else X
    y_arr = np.array(y) if np else y
    
    default_params = {
        "max_depth": 6,
        "eta": 0.3,
    }
    
    if task == "classification":
        n_classes = len(set(y))
        if n_classes == 2:
            default_params["objective"] = "binary:logistic"
            default_metrics = ["logloss", "auc"]
        else:
            default_params["objective"] = "multi:softprob"
            default_params["num_class"] = n_classes
            default_metrics = ["mlogloss"]
    else:
        default_params["objective"] = "reg:squarederror"
        default_metrics = ["rmse", "mae"]
    
    if params:
        default_params.update(params)
    
    dtrain = xgb.DMatrix(X_arr, label=y_arr)
    
    cv_results = xgb.cv(
        default_params,
        dtrain,
        num_boost_round=num_boost_round,
        nfold=nfold,
        stratified=stratified,
        metrics=metrics or default_metrics,
        as_pandas=False,
        verbose_eval=False,
    )
    
    # Convert to regular dict
    results = {}
    for key, values in cv_results.items():
        results[key] = list(values) if hasattr(values, "tolist") else values
    
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
    if xgb is None:
        raise ImportError("xgboost is not installed")
    
    if model_name not in _model_cache:
        return {"error": f"Model '{model_name}' not found"}
    
    model = _model_cache[model_name]
    importance = model.get_score(importance_type=importance_type)
    
    # Sort by importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "importance": dict(sorted_importance),
        "importance_type": importance_type,
        "num_features": len(importance),
    }


def _save_model(
    model_name: str = "default",
    file_path: Optional[str] = None,
    format: str = "json",
) -> Dict[str, Any]:
    """Save model to file or return as base64."""
    if xgb is None:
        raise ImportError("xgboost is not installed")
    
    if model_name not in _model_cache:
        return {"error": f"Model '{model_name}' not found"}
    
    model = _model_cache[model_name]
    
    if file_path:
        if format == "json":
            model.save_model(file_path)
        else:
            model.save_model(file_path)
        return {"saved_to": file_path, "format": format}
    else:
        # Return as base64
        buffer = io.BytesIO()
        model.save_model(buffer)
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")
        return {"model_base64": encoded, "format": "binary"}


def _load_model(
    model_name: str = "default",
    file_path: Optional[str] = None,
    model_base64: Optional[str] = None,
) -> Dict[str, Any]:
    """Load model from file or base64."""
    if xgb is None:
        raise ImportError("xgboost is not installed")
    
    model = xgb.Booster()
    
    if file_path:
        model.load_model(file_path)
    elif model_base64:
        buffer = io.BytesIO(base64.b64decode(model_base64))
        model.load_model(buffer)
    else:
        return {"error": "Provide file_path or model_base64"}
    
    _model_cache[model_name] = model
    
    return {
        "loaded": model_name,
        "num_features": model.num_features(),
    }


def _tune(
    X: List[List[float]],
    y: List[float],
    task: str = "classification",
    param_grid: Optional[Dict[str, List[Any]]] = None,
    nfold: int = 3,
    num_boost_round: int = 100,
) -> Dict[str, Any]:
    """Simple grid search hyperparameter tuning."""
    if xgb is None:
        raise ImportError("xgboost is not installed")
    
    default_grid = {
        "max_depth": [3, 6, 9],
        "eta": [0.1, 0.3],
        "min_child_weight": [1, 3],
    }
    
    param_grid = param_grid or default_grid
    
    # Generate all combinations
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    X_arr = np.array(X) if np else X
    y_arr = np.array(y) if np else y
    
    best_score = float("inf")
    best_params = {}
    results = []
    
    base_params = {}
    if task == "classification":
        n_classes = len(set(y))
        if n_classes == 2:
            base_params["objective"] = "binary:logistic"
            metric = "logloss"
        else:
            base_params["objective"] = "multi:softprob"
            base_params["num_class"] = n_classes
            metric = "mlogloss"
    else:
        base_params["objective"] = "reg:squarederror"
        metric = "rmse"
    
    dtrain = xgb.DMatrix(X_arr, label=y_arr)
    
    for combo in combinations:
        params = base_params.copy()
        for i, key in enumerate(keys):
            params[key] = combo[i]
        
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            nfold=nfold,
            metrics=[metric],
            as_pandas=False,
            verbose_eval=False,
        )
        
        final_score = cv_results[f"test-{metric}-mean"][-1]
        
        result = {
            "params": {keys[i]: combo[i] for i in range(len(keys))},
            "score": final_score,
        }
        results.append(result)
        
        if final_score < best_score:
            best_score = final_score
            best_params = result["params"]
    
    return {
        "best_params": best_params,
        "best_score": best_score,
        "metric": metric,
        "all_results": results,
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run XGBoost operations.
    
    Args:
        args: Dictionary with:
            - operation: XGBoost operation to perform
            - X, y: Training data
            - Various operation-specific parameters
    
    Returns:
        Result dictionary with operation output
    """
    args = args or {}
    operation = args.get("operation", "train")
    
    if xgb is None:
        return {
            "tool": "ml_xgboost",
            "status": "error",
            "error": "xgboost not installed. Run: pip install xgboost",
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
                model_name=args.get("model_name", "default"),
            )
        
        elif operation == "predict":
            result = _predict(
                X=args.get("X", []),
                model_name=args.get("model_name", "default"),
                output_margin=args.get("output_margin", False),
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
                metrics=args.get("metrics"),
            )
        
        elif operation == "feature_importance":
            result = _feature_importance(
                model_name=args.get("model_name", "default"),
                importance_type=args.get("importance_type", "gain"),
            )
        
        elif operation == "save_model":
            result = _save_model(
                model_name=args.get("model_name", "default"),
                file_path=args.get("file_path"),
                format=args.get("format", "json"),
            )
        
        elif operation == "load_model":
            result = _load_model(
                model_name=args.get("model_name", "default"),
                file_path=args.get("file_path"),
                model_base64=args.get("model_base64"),
            )
        
        elif operation == "tune":
            result = _tune(
                X=args.get("X", []),
                y=args.get("y", []),
                task=args.get("task", "classification"),
                param_grid=args.get("param_grid"),
                nfold=args.get("nfold", 3),
                num_boost_round=args.get("num_boost_round", 100),
            )
        
        else:
            return {
                "tool": "ml_xgboost",
                "status": "error",
                "error": f"Unknown operation: {operation}",
            }
        
        return {"tool": "ml_xgboost", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "ml_xgboost", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "train_classifier": {
            "operation": "train",
            "X": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "y": [0, 0, 1, 1],
            "task": "classification",
            "params": {"max_depth": 3, "eta": 0.1},
            "num_boost_round": 50,
        },
        "predict": {
            "operation": "predict",
            "X": [[2, 3], [6, 7]],
            "model_name": "default",
        },
        "cross_validate": {
            "operation": "cross_validate",
            "X": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
            "y": [0, 0, 1, 1, 1],
            "task": "classification",
            "nfold": 3,
        },
        "tune": {
            "operation": "tune",
            "X": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "y": [0, 0, 1, 1],
            "param_grid": {
                "max_depth": [3, 6],
                "eta": [0.1, 0.3],
            },
        },
    }
