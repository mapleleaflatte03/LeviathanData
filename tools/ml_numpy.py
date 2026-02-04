"""Tool: ml_numpy
NumPy numerical computing utilities.

Supported operations:
- array: Create arrays from data
- stats: Statistical operations (mean, std, median, etc.)
- linalg: Linear algebra operations
- transform: Array transformations (reshape, transpose, etc.)
- random: Random number generation
"""
from typing import Any, Dict, List, Optional, Union
import json


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


np = _optional_import("numpy")


def _to_list(arr: Any) -> Union[List, float, int]:
    """Convert numpy array to Python list for JSON serialization."""
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return arr


def _stats(data: List, operations: List[str]) -> Dict[str, Any]:
    """Compute statistical operations on data."""
    if np is None:
        raise ImportError("numpy is not installed")
    
    arr = np.array(data)
    results = {}
    
    stat_funcs = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
        "var": np.var,
        "min": np.min,
        "max": np.max,
        "sum": np.sum,
        "prod": np.prod,
        "percentile_25": lambda x: np.percentile(x, 25),
        "percentile_50": lambda x: np.percentile(x, 50),
        "percentile_75": lambda x: np.percentile(x, 75),
        "percentile_90": lambda x: np.percentile(x, 90),
        "percentile_99": lambda x: np.percentile(x, 99),
    }
    
    for op in operations:
        if op in stat_funcs:
            try:
                results[op] = _to_list(stat_funcs[op](arr))
            except Exception as e:
                results[op] = f"error: {str(e)}"
        elif op.startswith("percentile_"):
            try:
                p = int(op.split("_")[1])
                results[op] = _to_list(np.percentile(arr, p))
            except Exception as e:
                results[op] = f"error: {str(e)}"
    
    return results


def _linalg(data: List, operation: str, **kwargs) -> Dict[str, Any]:
    """Perform linear algebra operations."""
    if np is None:
        raise ImportError("numpy is not installed")
    
    arr = np.array(data)
    
    if operation == "dot":
        other = np.array(kwargs.get("other", []))
        result = np.dot(arr, other)
        return {"result": _to_list(result)}
    
    elif operation == "matmul":
        other = np.array(kwargs.get("other", []))
        result = np.matmul(arr, other)
        return {"result": _to_list(result)}
    
    elif operation == "inv":
        result = np.linalg.inv(arr)
        return {"result": _to_list(result)}
    
    elif operation == "det":
        result = np.linalg.det(arr)
        return {"result": _to_list(result)}
    
    elif operation == "eig":
        eigenvalues, eigenvectors = np.linalg.eig(arr)
        return {
            "eigenvalues": _to_list(eigenvalues),
            "eigenvectors": _to_list(eigenvectors),
        }
    
    elif operation == "svd":
        u, s, vh = np.linalg.svd(arr)
        return {
            "u": _to_list(u),
            "s": _to_list(s),
            "vh": _to_list(vh),
        }
    
    elif operation == "norm":
        ord_val = kwargs.get("ord")
        result = np.linalg.norm(arr, ord=ord_val)
        return {"result": _to_list(result)}
    
    elif operation == "solve":
        b = np.array(kwargs.get("b", []))
        result = np.linalg.solve(arr, b)
        return {"result": _to_list(result)}
    
    else:
        raise ValueError(f"Unknown linalg operation: {operation}")


def _transform(data: List, operations: List[Dict]) -> Any:
    """Apply transformations to array."""
    if np is None:
        raise ImportError("numpy is not installed")
    
    arr = np.array(data)
    
    for op in operations:
        op_type = op.get("type")
        
        if op_type == "reshape":
            shape = tuple(op.get("shape", [-1]))
            arr = arr.reshape(shape)
        
        elif op_type == "transpose":
            axes = op.get("axes")
            arr = np.transpose(arr, axes=axes)
        
        elif op_type == "flatten":
            arr = arr.flatten()
        
        elif op_type == "squeeze":
            arr = np.squeeze(arr)
        
        elif op_type == "expand_dims":
            axis = op.get("axis", 0)
            arr = np.expand_dims(arr, axis=axis)
        
        elif op_type == "normalize":
            arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
        
        elif op_type == "minmax":
            arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
        
        elif op_type == "clip":
            min_val = op.get("min", None)
            max_val = op.get("max", None)
            arr = np.clip(arr, min_val, max_val)
        
        elif op_type == "round":
            decimals = op.get("decimals", 0)
            arr = np.round(arr, decimals=decimals)
    
    return arr


def _random(operation: str, **kwargs) -> Any:
    """Generate random numbers."""
    if np is None:
        raise ImportError("numpy is not installed")
    
    seed = kwargs.get("seed")
    if seed is not None:
        np.random.seed(seed)
    
    shape = tuple(kwargs.get("shape", [10]))
    
    if operation == "uniform":
        low = kwargs.get("low", 0.0)
        high = kwargs.get("high", 1.0)
        return np.random.uniform(low, high, shape)
    
    elif operation == "normal":
        loc = kwargs.get("loc", 0.0)
        scale = kwargs.get("scale", 1.0)
        return np.random.normal(loc, scale, shape)
    
    elif operation == "randint":
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 100)
        return np.random.randint(low, high, shape)
    
    elif operation == "choice":
        a = kwargs.get("a", [])
        size = kwargs.get("size", 1)
        replace = kwargs.get("replace", True)
        return np.random.choice(a, size=size, replace=replace)
    
    elif operation == "permutation":
        x = kwargs.get("x", 10)
        return np.random.permutation(x)
    
    else:
        raise ValueError(f"Unknown random operation: {operation}")


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run numpy operations.
    
    Args:
        args: Dictionary with:
            - operation: "stats", "linalg", "transform", "random"
            - data: Input data array (not needed for random)
            - stats_ops: List of stats to compute
            - linalg_op: Linear algebra operation name
            - transforms: List of transformation operations
            - random_op: Random generation operation
    
    Returns:
        Result dictionary with status and computed values
    """
    args = args or {}
    operation = args.get("operation", "stats")
    data = args.get("data", [])
    
    if np is None:
        return {"tool": "ml_numpy", "status": "error", "error": "numpy not installed"}
    
    try:
        if operation == "stats":
            stats_ops = args.get("stats_ops", ["mean", "std", "min", "max"])
            results = _stats(data, stats_ops)
            return {
                "tool": "ml_numpy",
                "status": "ok",
                "statistics": results,
            }
        
        elif operation == "linalg":
            linalg_op = args.get("linalg_op", "dot")
            results = _linalg(data, linalg_op, **args)
            return {
                "tool": "ml_numpy",
                "status": "ok",
                **results,
            }
        
        elif operation == "transform":
            transforms = args.get("transforms", [])
            result = _transform(data, transforms)
            return {
                "tool": "ml_numpy",
                "status": "ok",
                "shape": list(result.shape),
                "result": _to_list(result),
            }
        
        elif operation == "random":
            random_op = args.get("random_op", "uniform")
            result = _random(random_op, **args)
            return {
                "tool": "ml_numpy",
                "status": "ok",
                "result": _to_list(result),
            }
        
        elif operation == "array":
            arr = np.array(data)
            return {
                "tool": "ml_numpy",
                "status": "ok",
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "result": _to_list(arr),
            }
        
        else:
            return {"tool": "ml_numpy", "status": "error", "error": f"Unknown operation: {operation}"}
    
    except Exception as e:
        return {"tool": "ml_numpy", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "stats": {
            "operation": "stats",
            "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "stats_ops": ["mean", "std", "median", "percentile_90"],
        },
        "linalg": {
            "operation": "linalg",
            "data": [[1, 2], [3, 4]],
            "linalg_op": "eig",
        },
        "transform": {
            "operation": "transform",
            "data": [[1, 2, 3], [4, 5, 6]],
            "transforms": [
                {"type": "transpose"},
                {"type": "normalize"},
            ],
        },
        "random": {
            "operation": "random",
            "random_op": "normal",
            "loc": 0,
            "scale": 1,
            "shape": [100],
            "seed": 42,
        },
    }
