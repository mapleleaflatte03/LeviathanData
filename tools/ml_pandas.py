"""Tool: ml_pandas
Pandas data processing utilities.

Supported operations:
- read: Read data from CSV, JSON, Excel, Parquet
- describe: Get statistical summary of data
- filter: Filter rows based on conditions
- transform: Apply transformations (groupby, pivot, merge)
- write: Write data to various formats
"""
from typing import Any, Dict, Optional
import json


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


pd = _optional_import("pandas")


def _read_data(file_path: str, file_type: Optional[str] = None) -> Any:
    """Read data from file into DataFrame."""
    if pd is None:
        raise ImportError("pandas is not installed. Run: pip install pandas")
    
    if file_type is None:
        file_type = file_path.split(".")[-1].lower()
    
    readers = {
        "csv": pd.read_csv,
        "json": pd.read_json,
        "xlsx": pd.read_excel,
        "xls": pd.read_excel,
        "parquet": pd.read_parquet,
        "feather": pd.read_feather,
    }
    
    reader = readers.get(file_type)
    if reader is None:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return reader(file_path)


def _describe(df: Any) -> Dict[str, Any]:
    """Get statistical description of DataFrame."""
    desc = df.describe(include="all").to_dict()
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "statistics": desc,
        "null_counts": df.isnull().sum().to_dict(),
    }


def _filter(df: Any, conditions: list) -> Any:
    """Filter DataFrame based on conditions."""
    for cond in conditions:
        col = cond.get("column")
        op = cond.get("op", "==")
        val = cond.get("value")
        
        if op == "==":
            df = df[df[col] == val]
        elif op == "!=":
            df = df[df[col] != val]
        elif op == ">":
            df = df[df[col] > val]
        elif op == ">=":
            df = df[df[col] >= val]
        elif op == "<":
            df = df[df[col] < val]
        elif op == "<=":
            df = df[df[col] <= val]
        elif op == "in":
            df = df[df[col].isin(val)]
        elif op == "contains":
            df = df[df[col].str.contains(val, na=False)]
    
    return df


def _transform(df: Any, operations: list) -> Any:
    """Apply transformations to DataFrame."""
    for op in operations:
        op_type = op.get("type")
        
        if op_type == "groupby":
            by = op.get("by")
            agg = op.get("agg", "sum")
            df = df.groupby(by).agg(agg).reset_index()
        
        elif op_type == "sort":
            by = op.get("by")
            ascending = op.get("ascending", True)
            df = df.sort_values(by=by, ascending=ascending)
        
        elif op_type == "select":
            columns = op.get("columns")
            df = df[columns]
        
        elif op_type == "rename":
            mapping = op.get("mapping")
            df = df.rename(columns=mapping)
        
        elif op_type == "dropna":
            subset = op.get("subset")
            df = df.dropna(subset=subset)
        
        elif op_type == "fillna":
            value = op.get("value")
            df = df.fillna(value)
        
        elif op_type == "head":
            n = op.get("n", 10)
            df = df.head(n)
        
        elif op_type == "tail":
            n = op.get("n", 10)
            df = df.tail(n)
    
    return df


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run pandas operations.
    
    Args:
        args: Dictionary with:
            - operation: "read", "describe", "filter", "transform", "write"
            - file_path: Path to input file
            - file_type: Optional file type override
            - conditions: List of filter conditions (for filter op)
            - transforms: List of transformations (for transform op)
            - output_path: Path for output file (for write op)
            - output_format: Format for output (csv, json, parquet)
    
    Returns:
        Result dictionary with status and data/preview
    """
    args = args or {}
    operation = args.get("operation", "describe")
    file_path = args.get("file_path")
    
    if pd is None:
        return {"tool": "ml_pandas", "status": "error", "error": "pandas not installed"}
    
    try:
        if operation == "read":
            df = _read_data(file_path, args.get("file_type"))
            return {
                "tool": "ml_pandas",
                "status": "ok",
                "shape": list(df.shape),
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient="records"),
            }
        
        elif operation == "describe":
            df = _read_data(file_path, args.get("file_type"))
            return {
                "tool": "ml_pandas",
                "status": "ok",
                "description": _describe(df),
            }
        
        elif operation == "filter":
            df = _read_data(file_path, args.get("file_type"))
            conditions = args.get("conditions", [])
            df = _filter(df, conditions)
            return {
                "tool": "ml_pandas",
                "status": "ok",
                "shape": list(df.shape),
                "preview": df.head(10).to_dict(orient="records"),
            }
        
        elif operation == "transform":
            df = _read_data(file_path, args.get("file_type"))
            transforms = args.get("transforms", [])
            df = _transform(df, transforms)
            return {
                "tool": "ml_pandas",
                "status": "ok",
                "shape": list(df.shape),
                "preview": df.head(10).to_dict(orient="records"),
            }
        
        elif operation == "write":
            df = _read_data(file_path, args.get("file_type"))
            
            if args.get("conditions"):
                df = _filter(df, args["conditions"])
            if args.get("transforms"):
                df = _transform(df, args["transforms"])
            
            output_path = args.get("output_path")
            output_format = args.get("output_format", "csv")
            
            if output_format == "csv":
                df.to_csv(output_path, index=False)
            elif output_format == "json":
                df.to_json(output_path, orient="records", indent=2)
            elif output_format == "parquet":
                df.to_parquet(output_path, index=False)
            elif output_format == "excel":
                df.to_excel(output_path, index=False)
            
            return {
                "tool": "ml_pandas",
                "status": "ok",
                "output_path": output_path,
                "rows_written": len(df),
            }
        
        else:
            return {"tool": "ml_pandas", "status": "error", "error": f"Unknown operation: {operation}"}
    
    except Exception as e:
        return {"tool": "ml_pandas", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "describe": {
            "operation": "describe",
            "file_path": "data.csv",
        },
        "filter": {
            "operation": "filter",
            "file_path": "data.csv",
            "conditions": [
                {"column": "age", "op": ">", "value": 25},
                {"column": "city", "op": "in", "value": ["NYC", "LA"]},
            ],
        },
        "transform": {
            "operation": "transform",
            "file_path": "data.csv",
            "transforms": [
                {"type": "groupby", "by": ["category"], "agg": "sum"},
                {"type": "sort", "by": "total", "ascending": False},
            ],
        },
    }
