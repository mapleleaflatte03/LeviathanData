"""Tool: ml_flaml
FLAML automated machine learning.

Supported operations:
- automl: Run AutoML to find best model
- predict: Make predictions with best model
- feature_importance: Get feature importance
- tune: Hyperparameter tuning for specific estimator
"""
from typing import Any, Dict, List, Optional, Union
import json
import os
import time


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


flaml = _optional_import("flaml")
np = _optional_import("numpy")
pd = _optional_import("pandas")

# AutoML cache
_automl_cache: Dict[str, Any] = {}


def _automl(
    X: List[List[float]],
    y: List[float],
    task: str = "classification",
    time_budget: int = 60,
    metric: Optional[str] = None,
    estimator_list: Optional[List[str]] = None,
    n_splits: int = 5,
    model_name: str = "default",
) -> Dict[str, Any]:
    """Run AutoML to find the best model."""
    if flaml is None:
        raise ImportError("flaml is not installed. Run: pip install flaml")
    
    from flaml import AutoML
    
    X_arr = np.array(X) if np else X
    y_arr = np.array(y) if np else y
    
    automl = AutoML()
    
    settings = {
        "time_budget": time_budget,
        "task": task,
        "n_splits": n_splits,
        "verbose": 0,
    }
    
    if metric:
        settings["metric"] = metric
    
    if estimator_list:
        settings["estimator_list"] = estimator_list
    
    start_time = time.time()
    automl.fit(X_arr, y_arr, **settings)
    elapsed = time.time() - start_time
    
    _automl_cache[model_name] = automl
    
    return {
        "model_name": model_name,
        "best_estimator": automl.best_estimator,
        "best_config": automl.best_config,
        "best_loss": automl.best_loss,
        "best_iteration": automl.best_iteration,
        "training_time": elapsed,
        "feature_names": automl.feature_names_in_.tolist() if hasattr(automl, "feature_names_in_") and automl.feature_names_in_ is not None else None,
    }


def _predict(
    X: List[List[float]],
    model_name: str = "default",
) -> Dict[str, Any]:
    """Make predictions with best model."""
    if flaml is None:
        raise ImportError("flaml is not installed")
    
    if model_name not in _automl_cache:
        return {"error": f"AutoML model '{model_name}' not found. Run automl first."}
    
    automl = _automl_cache[model_name]
    X_arr = np.array(X) if np else X
    
    predictions = automl.predict(X_arr)
    
    return {
        "predictions": predictions.tolist() if hasattr(predictions, "tolist") else list(predictions),
        "count": len(predictions),
    }


def _predict_proba(
    X: List[List[float]],
    model_name: str = "default",
) -> Dict[str, Any]:
    """Get prediction probabilities."""
    if flaml is None:
        raise ImportError("flaml is not installed")
    
    if model_name not in _automl_cache:
        return {"error": f"AutoML model '{model_name}' not found"}
    
    automl = _automl_cache[model_name]
    X_arr = np.array(X) if np else X
    
    try:
        proba = automl.predict_proba(X_arr)
        return {
            "probabilities": proba.tolist() if hasattr(proba, "tolist") else list(proba),
            "count": len(proba),
        }
    except Exception as e:
        return {"error": f"predict_proba not available: {str(e)}"}


def _feature_importance(
    model_name: str = "default",
) -> Dict[str, Any]:
    """Get feature importance from best model."""
    if flaml is None:
        raise ImportError("flaml is not installed")
    
    if model_name not in _automl_cache:
        return {"error": f"AutoML model '{model_name}' not found"}
    
    automl = _automl_cache[model_name]
    
    try:
        importance = automl.feature_importances_
        feature_names = automl.feature_names_in_
        
        if feature_names is not None and importance is not None:
            importance_dict = dict(zip(feature_names.tolist(), importance.tolist()))
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            return {"importance": dict(sorted_importance)}
        elif importance is not None:
            return {"importance": importance.tolist()}
        else:
            return {"error": "Feature importance not available"}
    except Exception as e:
        return {"error": f"Could not get feature importance: {str(e)}"}


def _tune(
    X: List[List[float]],
    y: List[float],
    estimator: str = "xgboost",
    task: str = "classification",
    time_budget: int = 30,
    metric: Optional[str] = None,
    model_name: str = "default",
) -> Dict[str, Any]:
    """Tune a specific estimator."""
    if flaml is None:
        raise ImportError("flaml is not installed")
    
    from flaml import AutoML
    
    X_arr = np.array(X) if np else X
    y_arr = np.array(y) if np else y
    
    automl = AutoML()
    
    settings = {
        "time_budget": time_budget,
        "task": task,
        "estimator_list": [estimator],
        "verbose": 0,
    }
    
    if metric:
        settings["metric"] = metric
    
    automl.fit(X_arr, y_arr, **settings)
    _automl_cache[model_name] = automl
    
    return {
        "model_name": model_name,
        "estimator": estimator,
        "best_config": automl.best_config,
        "best_loss": automl.best_loss,
    }


def _get_search_space(
    estimator: str = "xgboost",
) -> Dict[str, Any]:
    """Get the search space for an estimator."""
    if flaml is None:
        raise ImportError("flaml is not installed")
    
    try:
        from flaml.automl.model import get_estimator_class
        est_class = get_estimator_class(estimator)
        search_space = est_class.search_space()
        
        # Convert to serializable format
        space_info = {}
        for key, value in search_space.items():
            space_info[key] = str(value)
        
        return {"estimator": estimator, "search_space": space_info}
    except Exception as e:
        return {"error": f"Could not get search space: {str(e)}"}


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run FLAML operations."""
    args = args or {}
    operation = args.get("operation", "automl")
    
    if flaml is None:
        return {
            "tool": "ml_flaml",
            "status": "error",
            "error": "flaml not installed. Run: pip install flaml",
        }
    
    try:
        if operation == "automl":
            result = _automl(
                X=args.get("X", []),
                y=args.get("y", []),
                task=args.get("task", "classification"),
                time_budget=args.get("time_budget", 60),
                metric=args.get("metric"),
                estimator_list=args.get("estimator_list"),
                n_splits=args.get("n_splits", 5),
                model_name=args.get("model_name", "default"),
            )
        
        elif operation == "predict":
            result = _predict(
                X=args.get("X", []),
                model_name=args.get("model_name", "default"),
            )
        
        elif operation == "predict_proba":
            result = _predict_proba(
                X=args.get("X", []),
                model_name=args.get("model_name", "default"),
            )
        
        elif operation == "feature_importance":
            result = _feature_importance(
                model_name=args.get("model_name", "default"),
            )
        
        elif operation == "tune":
            result = _tune(
                X=args.get("X", []),
                y=args.get("y", []),
                estimator=args.get("estimator", "xgboost"),
                task=args.get("task", "classification"),
                time_budget=args.get("time_budget", 30),
                metric=args.get("metric"),
                model_name=args.get("model_name", "default"),
            )
        
        elif operation == "search_space":
            result = _get_search_space(
                estimator=args.get("estimator", "xgboost"),
            )
        
        else:
            return {"tool": "ml_flaml", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "ml_flaml", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "ml_flaml", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "automl": {
            "operation": "automl",
            "X": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
            "y": [0, 0, 1, 1, 1],
            "task": "classification",
            "time_budget": 30,
        },
        "tune_xgboost": {
            "operation": "tune",
            "X": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "y": [0, 0, 1, 1],
            "estimator": "xgboost",
            "time_budget": 20,
        },
    }
