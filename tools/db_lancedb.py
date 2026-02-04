"""Tool: db_lancedb
LanceDB vector database for efficient embeddings storage.

Supported operations:
- create_table: Create a new table
- drop_table: Drop a table
- list_tables: List all tables
- insert: Insert records
- search: Vector similarity search
- delete: Delete records
- update: Update records
- create_index: Create an index
"""
from typing import Any, Dict, List, Optional, Union
import json
import os


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


lancedb = _optional_import("lancedb")
pa = _optional_import("pyarrow")

# Connection cache
_conn_cache: Dict[str, Any] = {}


def _get_connection(uri: str = "./lancedb") -> Any:
    """Get or create LanceDB connection."""
    if lancedb is None:
        raise ImportError("lancedb is not installed. Run: pip install lancedb")
    
    if uri not in _conn_cache:
        _conn_cache[uri] = lancedb.connect(uri)
    
    return _conn_cache[uri]


def _create_table(
    table_name: str,
    data: List[Dict[str, Any]],
    mode: str = "create",
    uri: str = "./lancedb",
) -> Dict[str, Any]:
    """Create a new table with initial data."""
    db = _get_connection(uri)
    
    table = db.create_table(table_name, data, mode=mode)
    
    return {
        "created": table_name,
        "rows": len(data),
        "schema": str(table.schema) if hasattr(table, "schema") else None,
    }


def _drop_table(table_name: str, uri: str = "./lancedb") -> Dict[str, Any]:
    """Drop a table."""
    db = _get_connection(uri)
    db.drop_table(table_name)
    return {"dropped": table_name}


def _list_tables(uri: str = "./lancedb") -> Dict[str, Any]:
    """List all tables."""
    db = _get_connection(uri)
    tables = db.table_names()
    return {"tables": list(tables)}


def _insert(
    table_name: str,
    data: List[Dict[str, Any]],
    uri: str = "./lancedb",
) -> Dict[str, Any]:
    """Insert records into a table."""
    db = _get_connection(uri)
    table = db.open_table(table_name)
    table.add(data)
    
    return {
        "inserted": len(data),
        "table": table_name,
    }


def _search(
    table_name: str,
    query_vector: List[float],
    limit: int = 10,
    columns: Optional[List[str]] = None,
    filter_sql: Optional[str] = None,
    uri: str = "./lancedb",
    metric: str = "L2",
    nprobes: int = 20,
) -> Dict[str, Any]:
    """Vector similarity search."""
    db = _get_connection(uri)
    table = db.open_table(table_name)
    
    search = table.search(query_vector)
    
    if limit:
        search = search.limit(limit)
    
    if columns:
        search = search.select(columns)
    
    if filter_sql:
        search = search.where(filter_sql)
    
    search = search.metric(metric)
    search = search.nprobes(nprobes)
    
    results = search.to_list()
    
    return {
        "results": results,
        "count": len(results),
    }


def _search_text(
    table_name: str,
    query: str,
    limit: int = 10,
    columns: Optional[List[str]] = None,
    uri: str = "./lancedb",
) -> Dict[str, Any]:
    """Full-text search (requires FTS index)."""
    db = _get_connection(uri)
    table = db.open_table(table_name)
    
    search = table.search(query, query_type="fts")
    
    if limit:
        search = search.limit(limit)
    
    if columns:
        search = search.select(columns)
    
    results = search.to_list()
    
    return {
        "results": results,
        "count": len(results),
        "query_type": "fts",
    }


def _delete(
    table_name: str,
    filter_sql: str,
    uri: str = "./lancedb",
) -> Dict[str, Any]:
    """Delete records matching filter."""
    db = _get_connection(uri)
    table = db.open_table(table_name)
    table.delete(filter_sql)
    
    return {
        "deleted": True,
        "filter": filter_sql,
    }


def _update(
    table_name: str,
    updates: Dict[str, Any],
    filter_sql: Optional[str] = None,
    uri: str = "./lancedb",
) -> Dict[str, Any]:
    """Update records."""
    db = _get_connection(uri)
    table = db.open_table(table_name)
    
    if filter_sql:
        table.update(where=filter_sql, values=updates)
    else:
        table.update(values=updates)
    
    return {
        "updated": True,
        "filter": filter_sql,
    }


def _create_index(
    table_name: str,
    index_type: str = "IVF_PQ",
    num_partitions: int = 256,
    num_sub_vectors: int = 96,
    column: str = "vector",
    uri: str = "./lancedb",
) -> Dict[str, Any]:
    """Create an index on the table."""
    db = _get_connection(uri)
    table = db.open_table(table_name)
    
    table.create_index(
        index_type=index_type,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        column=column,
    )
    
    return {
        "indexed": table_name,
        "index_type": index_type,
        "column": column,
    }


def _create_fts_index(
    table_name: str,
    columns: List[str],
    uri: str = "./lancedb",
) -> Dict[str, Any]:
    """Create a full-text search index."""
    db = _get_connection(uri)
    table = db.open_table(table_name)
    
    table.create_fts_index(columns)
    
    return {
        "fts_indexed": table_name,
        "columns": columns,
    }


def _get_table_info(table_name: str, uri: str = "./lancedb") -> Dict[str, Any]:
    """Get table information."""
    db = _get_connection(uri)
    table = db.open_table(table_name)
    
    return {
        "name": table_name,
        "schema": str(table.schema) if hasattr(table, "schema") else None,
        "count": table.count_rows() if hasattr(table, "count_rows") else len(table),
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run LanceDB operations.
    
    Args:
        args: Dictionary with:
            - operation: LanceDB operation to perform
            - table_name: Name of the table
            - uri: Database URI (default: ./lancedb)
            - Various operation-specific parameters
    
    Returns:
        Result dictionary with operation output
    """
    args = args or {}
    operation = args.get("operation", "list_tables")
    uri = args.get("uri", "./lancedb")
    
    if lancedb is None:
        return {
            "tool": "db_lancedb",
            "status": "error",
            "error": "lancedb not installed. Run: pip install lancedb",
        }
    
    try:
        if operation == "create_table":
            result = _create_table(
                table_name=args.get("table_name", ""),
                data=args.get("data", []),
                mode=args.get("mode", "create"),
                uri=uri,
            )
        
        elif operation == "drop_table":
            result = _drop_table(
                table_name=args.get("table_name", ""),
                uri=uri,
            )
        
        elif operation == "list_tables":
            result = _list_tables(uri=uri)
        
        elif operation == "info":
            result = _get_table_info(
                table_name=args.get("table_name", ""),
                uri=uri,
            )
        
        elif operation == "insert":
            result = _insert(
                table_name=args.get("table_name", ""),
                data=args.get("data", []),
                uri=uri,
            )
        
        elif operation == "search":
            result = _search(
                table_name=args.get("table_name", ""),
                query_vector=args.get("query_vector", []),
                limit=args.get("limit", 10),
                columns=args.get("columns"),
                filter_sql=args.get("filter"),
                uri=uri,
                metric=args.get("metric", "L2"),
                nprobes=args.get("nprobes", 20),
            )
        
        elif operation == "search_text":
            result = _search_text(
                table_name=args.get("table_name", ""),
                query=args.get("query", ""),
                limit=args.get("limit", 10),
                columns=args.get("columns"),
                uri=uri,
            )
        
        elif operation == "delete":
            result = _delete(
                table_name=args.get("table_name", ""),
                filter_sql=args.get("filter", ""),
                uri=uri,
            )
        
        elif operation == "update":
            result = _update(
                table_name=args.get("table_name", ""),
                updates=args.get("updates", {}),
                filter_sql=args.get("filter"),
                uri=uri,
            )
        
        elif operation == "create_index":
            result = _create_index(
                table_name=args.get("table_name", ""),
                index_type=args.get("index_type", "IVF_PQ"),
                num_partitions=args.get("num_partitions", 256),
                num_sub_vectors=args.get("num_sub_vectors", 96),
                column=args.get("column", "vector"),
                uri=uri,
            )
        
        elif operation == "create_fts_index":
            result = _create_fts_index(
                table_name=args.get("table_name", ""),
                columns=args.get("columns", []),
                uri=uri,
            )
        
        else:
            return {
                "tool": "db_lancedb",
                "status": "error",
                "error": f"Unknown operation: {operation}",
            }
        
        return {"tool": "db_lancedb", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "db_lancedb", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "create_table": {
            "operation": "create_table",
            "table_name": "documents",
            "data": [
                {"id": "1", "text": "Hello world", "vector": [0.1] * 384},
                {"id": "2", "text": "Goodbye world", "vector": [0.2] * 384},
            ],
        },
        "search": {
            "operation": "search",
            "table_name": "documents",
            "query_vector": [0.15] * 384,
            "limit": 5,
            "columns": ["id", "text"],
        },
        "insert": {
            "operation": "insert",
            "table_name": "documents",
            "data": [
                {"id": "3", "text": "New document", "vector": [0.3] * 384},
            ],
        },
        "create_index": {
            "operation": "create_index",
            "table_name": "documents",
            "index_type": "IVF_PQ",
            "column": "vector",
        },
    }
