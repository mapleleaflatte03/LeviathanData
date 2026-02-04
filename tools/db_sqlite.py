"""Tool: db_sqlite
SQLite database operations.

Supported operations:
- query: Execute SELECT queries
- execute: Execute INSERT/UPDATE/DELETE statements
- create_table: Create new tables
- describe: Get table schema information
- tables: List all tables in database
"""
from typing import Any, Dict, List, Optional
import json


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


sqlite3 = _optional_import("sqlite3")


def _connect(db_path: str):
    """Create database connection."""
    if sqlite3 is None:
        raise ImportError("sqlite3 is not available")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _query(db_path: str, sql: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
    """Execute SELECT query and return results."""
    params = params or []
    with _connect(db_path) as conn:
        cursor = conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]


def _execute(db_path: str, sql: str, params: Optional[List] = None) -> Dict[str, Any]:
    """Execute non-SELECT statement."""
    params = params or []
    with _connect(db_path) as conn:
        cursor = conn.execute(sql, params)
        conn.commit()
        return {
            "rowcount": cursor.rowcount,
            "lastrowid": cursor.lastrowid,
        }


def _execute_many(db_path: str, sql: str, params_list: List[List]) -> Dict[str, Any]:
    """Execute statement with multiple parameter sets."""
    with _connect(db_path) as conn:
        cursor = conn.executemany(sql, params_list)
        conn.commit()
        return {
            "rowcount": cursor.rowcount,
        }


def _create_table(db_path: str, table_name: str, columns: List[Dict[str, str]], if_not_exists: bool = True) -> Dict[str, Any]:
    """Create a new table."""
    col_defs = []
    for col in columns:
        col_def = f"{col['name']} {col['type']}"
        if col.get("primary_key"):
            col_def += " PRIMARY KEY"
        if col.get("autoincrement"):
            col_def += " AUTOINCREMENT"
        if col.get("not_null"):
            col_def += " NOT NULL"
        if col.get("unique"):
            col_def += " UNIQUE"
        if col.get("default") is not None:
            col_def += f" DEFAULT {col['default']}"
        col_defs.append(col_def)
    
    exists_clause = "IF NOT EXISTS " if if_not_exists else ""
    sql = f"CREATE TABLE {exists_clause}{table_name} ({', '.join(col_defs)})"
    
    with _connect(db_path) as conn:
        conn.execute(sql)
        conn.commit()
    
    return {"created": table_name, "sql": sql}


def _describe_table(db_path: str, table_name: str) -> Dict[str, Any]:
    """Get table schema information."""
    with _connect(db_path) as conn:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns = []
        for row in cursor.fetchall():
            columns.append({
                "cid": row[0],
                "name": row[1],
                "type": row[2],
                "notnull": bool(row[3]),
                "default": row[4],
                "pk": bool(row[5]),
            })
        
        cursor = conn.execute(f"PRAGMA index_list({table_name})")
        indexes = []
        for row in cursor.fetchall():
            indexes.append({
                "seq": row[0],
                "name": row[1],
                "unique": bool(row[2]),
            })
        
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
    
    return {
        "table": table_name,
        "columns": columns,
        "indexes": indexes,
        "row_count": row_count,
    }


def _list_tables(db_path: str) -> List[str]:
    """List all tables in database."""
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [row[0] for row in cursor.fetchall()]


def _drop_table(db_path: str, table_name: str, if_exists: bool = True) -> Dict[str, Any]:
    """Drop a table."""
    exists_clause = "IF EXISTS " if if_exists else ""
    sql = f"DROP TABLE {exists_clause}{table_name}"
    
    with _connect(db_path) as conn:
        conn.execute(sql)
        conn.commit()
    
    return {"dropped": table_name}


def _vacuum(db_path: str) -> Dict[str, Any]:
    """Vacuum the database to reclaim space."""
    with _connect(db_path) as conn:
        conn.execute("VACUUM")
    return {"vacuumed": True}


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run SQLite operations.
    
    Args:
        args: Dictionary with:
            - operation: "query", "execute", "execute_many", "create_table", 
                        "describe", "tables", "drop_table", "vacuum"
            - db_path: Path to SQLite database file
            - sql: SQL statement (for query/execute)
            - params: Parameters for SQL (list)
            - params_list: List of parameter lists (for execute_many)
            - table_name: Table name (for create_table/describe/drop_table)
            - columns: Column definitions (for create_table)
    
    Returns:
        Result dictionary with status and data
    """
    args = args or {}
    operation = args.get("operation", "tables")
    db_path = args.get("db_path", ":memory:")
    
    if sqlite3 is None:
        return {"tool": "db_sqlite", "status": "error", "error": "sqlite3 not available"}
    
    try:
        if operation == "query":
            sql = args.get("sql", "")
            params = args.get("params", [])
            rows = _query(db_path, sql, params)
            return {
                "tool": "db_sqlite",
                "status": "ok",
                "rows": rows,
                "count": len(rows),
            }
        
        elif operation == "execute":
            sql = args.get("sql", "")
            params = args.get("params", [])
            result = _execute(db_path, sql, params)
            return {
                "tool": "db_sqlite",
                "status": "ok",
                **result,
            }
        
        elif operation == "execute_many":
            sql = args.get("sql", "")
            params_list = args.get("params_list", [])
            result = _execute_many(db_path, sql, params_list)
            return {
                "tool": "db_sqlite",
                "status": "ok",
                **result,
            }
        
        elif operation == "create_table":
            table_name = args.get("table_name", "")
            columns = args.get("columns", [])
            if_not_exists = args.get("if_not_exists", True)
            result = _create_table(db_path, table_name, columns, if_not_exists)
            return {
                "tool": "db_sqlite",
                "status": "ok",
                **result,
            }
        
        elif operation == "describe":
            table_name = args.get("table_name", "")
            result = _describe_table(db_path, table_name)
            return {
                "tool": "db_sqlite",
                "status": "ok",
                **result,
            }
        
        elif operation == "tables":
            tables = _list_tables(db_path)
            return {
                "tool": "db_sqlite",
                "status": "ok",
                "tables": tables,
            }
        
        elif operation == "drop_table":
            table_name = args.get("table_name", "")
            if_exists = args.get("if_exists", True)
            result = _drop_table(db_path, table_name, if_exists)
            return {
                "tool": "db_sqlite",
                "status": "ok",
                **result,
            }
        
        elif operation == "vacuum":
            result = _vacuum(db_path)
            return {
                "tool": "db_sqlite",
                "status": "ok",
                **result,
            }
        
        else:
            return {"tool": "db_sqlite", "status": "error", "error": f"Unknown operation: {operation}"}
    
    except Exception as e:
        return {"tool": "db_sqlite", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "create_table": {
            "operation": "create_table",
            "db_path": "data.db",
            "table_name": "users",
            "columns": [
                {"name": "id", "type": "INTEGER", "primary_key": True, "autoincrement": True},
                {"name": "name", "type": "TEXT", "not_null": True},
                {"name": "email", "type": "TEXT", "unique": True},
                {"name": "created_at", "type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"},
            ],
        },
        "insert": {
            "operation": "execute",
            "db_path": "data.db",
            "sql": "INSERT INTO users (name, email) VALUES (?, ?)",
            "params": ["John Doe", "john@example.com"],
        },
        "query": {
            "operation": "query",
            "db_path": "data.db",
            "sql": "SELECT * FROM users WHERE name LIKE ?",
            "params": ["%John%"],
        },
        "describe": {
            "operation": "describe",
            "db_path": "data.db",
            "table_name": "users",
        },
    }
