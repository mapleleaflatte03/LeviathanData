"""Tool: ml_tensorflow
TensorFlow deep learning for neural networks.

Supported operations:
- create_model: Create a Sequential model
- train: Train a model
- predict: Make predictions
- evaluate: Evaluate model
- save_model: Save model
- load_model: Load model
- summary: Get model summary
"""
from typing import Any, Dict, List, Optional, Union
import json
import os


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


tf = _optional_import("tensorflow")
np = _optional_import("numpy")

# Model cache
_model_cache: Dict[str, Any] = {}


def _create_model(
    layers: List[Dict[str, Any]],
    model_name: str = "default",
    optimizer: str = "adam",
    loss: str = "sparse_categorical_crossentropy",
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a Sequential model."""
    if tf is None:
        raise ImportError("tensorflow is not installed. Run: pip install tensorflow")
    
    from tensorflow import keras
    from tensorflow.keras import layers as keras_layers
    
    model = keras.Sequential()
    
    layer_map = {
        "dense": keras_layers.Dense,
        "dropout": keras_layers.Dropout,
        "flatten": keras_layers.Flatten,
        "conv2d": keras_layers.Conv2D,
        "maxpooling2d": keras_layers.MaxPooling2D,
        "batchnormalization": keras_layers.BatchNormalization,
        "lstm": keras_layers.LSTM,
        "gru": keras_layers.GRU,
        "embedding": keras_layers.Embedding,
        "input": keras_layers.InputLayer,
    }
    
    for layer_config in layers:
        layer_type = layer_config.pop("type", "dense").lower()
        layer_class = layer_map.get(layer_type)
        
        if layer_class is None:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
        model.add(layer_class(**layer_config))
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics or ["accuracy"],
    )
    
    _model_cache[model_name] = model
    
    return {
        "model_name": model_name,
        "num_layers": len(model.layers),
        "optimizer": optimizer,
        "loss": loss,
    }


def _train(
    X: List[List[float]],
    y: List[float],
    model_name: str = "default",
    epochs: int = 10,
    batch_size: int = 32,
    validation_split: float = 0.2,
    verbose: int = 0,
) -> Dict[str, Any]:
    """Train the model."""
    if tf is None:
        raise ImportError("tensorflow is not installed")
    
    if model_name not in _model_cache:
        return {"error": f"Model '{model_name}' not found. Create a model first."}
    
    model = _model_cache[model_name]
    X_arr = np.array(X) if np else X
    y_arr = np.array(y) if np else y
    
    history = model.fit(
        X_arr,
        y_arr,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )
    
    return {
        "model_name": model_name,
        "epochs_trained": epochs,
        "history": {
            k: [float(v) for v in vals] for k, vals in history.history.items()
        },
        "final_loss": float(history.history["loss"][-1]),
        "final_accuracy": float(history.history.get("accuracy", [0])[-1]),
    }


def _predict(
    X: List[List[float]],
    model_name: str = "default",
) -> Dict[str, Any]:
    """Make predictions."""
    if tf is None:
        raise ImportError("tensorflow is not installed")
    
    if model_name not in _model_cache:
        return {"error": f"Model '{model_name}' not found"}
    
    model = _model_cache[model_name]
    X_arr = np.array(X) if np else X
    
    predictions = model.predict(X_arr, verbose=0)
    
    return {
        "predictions": predictions.tolist(),
        "count": len(predictions),
    }


def _evaluate(
    X: List[List[float]],
    y: List[float],
    model_name: str = "default",
) -> Dict[str, Any]:
    """Evaluate model."""
    if tf is None:
        raise ImportError("tensorflow is not installed")
    
    if model_name not in _model_cache:
        return {"error": f"Model '{model_name}' not found"}
    
    model = _model_cache[model_name]
    X_arr = np.array(X) if np else X
    y_arr = np.array(y) if np else y
    
    results = model.evaluate(X_arr, y_arr, verbose=0)
    metrics = model.metrics_names
    
    return {
        "metrics": dict(zip(metrics, [float(r) for r in results])),
    }


def _save_model(
    model_name: str = "default",
    file_path: str = "model.keras",
) -> Dict[str, Any]:
    """Save model."""
    if tf is None:
        raise ImportError("tensorflow is not installed")
    
    if model_name not in _model_cache:
        return {"error": f"Model '{model_name}' not found"}
    
    model = _model_cache[model_name]
    model.save(file_path)
    
    return {"saved_to": file_path}


def _load_model(
    file_path: str,
    model_name: str = "default",
) -> Dict[str, Any]:
    """Load model."""
    if tf is None:
        raise ImportError("tensorflow is not installed")
    
    from tensorflow import keras
    
    model = keras.models.load_model(file_path)
    _model_cache[model_name] = model
    
    return {
        "loaded": model_name,
        "num_layers": len(model.layers),
    }


def _summary(model_name: str = "default") -> Dict[str, Any]:
    """Get model summary."""
    if tf is None:
        raise ImportError("tensorflow is not installed")
    
    if model_name not in _model_cache:
        return {"error": f"Model '{model_name}' not found"}
    
    model = _model_cache[model_name]
    
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    
    return {
        "summary": "\n".join(summary_lines),
        "total_params": model.count_params(),
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run TensorFlow operations."""
    args = args or {}
    operation = args.get("operation", "create_model")
    
    if tf is None:
        return {
            "tool": "ml_tensorflow",
            "status": "error",
            "error": "tensorflow not installed. Run: pip install tensorflow",
        }
    
    try:
        if operation == "create_model":
            result = _create_model(
                layers=args.get("layers", []),
                model_name=args.get("model_name", "default"),
                optimizer=args.get("optimizer", "adam"),
                loss=args.get("loss", "sparse_categorical_crossentropy"),
                metrics=args.get("metrics"),
            )
        
        elif operation == "train":
            result = _train(
                X=args.get("X", []),
                y=args.get("y", []),
                model_name=args.get("model_name", "default"),
                epochs=args.get("epochs", 10),
                batch_size=args.get("batch_size", 32),
                validation_split=args.get("validation_split", 0.2),
                verbose=args.get("verbose", 0),
            )
        
        elif operation == "predict":
            result = _predict(
                X=args.get("X", []),
                model_name=args.get("model_name", "default"),
            )
        
        elif operation == "evaluate":
            result = _evaluate(
                X=args.get("X", []),
                y=args.get("y", []),
                model_name=args.get("model_name", "default"),
            )
        
        elif operation == "save_model":
            result = _save_model(
                model_name=args.get("model_name", "default"),
                file_path=args.get("file_path", "model.keras"),
            )
        
        elif operation == "load_model":
            result = _load_model(
                file_path=args.get("file_path", ""),
                model_name=args.get("model_name", "default"),
            )
        
        elif operation == "summary":
            result = _summary(model_name=args.get("model_name", "default"))
        
        else:
            return {"tool": "ml_tensorflow", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "ml_tensorflow", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "ml_tensorflow", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "create_model": {
            "operation": "create_model",
            "layers": [
                {"type": "input", "input_shape": [10]},
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": 3, "activation": "softmax"},
            ],
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
        },
        "train": {
            "operation": "train",
            "X": [[0.1] * 10 for _ in range(100)],
            "y": [0, 1, 2] * 33 + [0],
            "epochs": 5,
        },
    }
