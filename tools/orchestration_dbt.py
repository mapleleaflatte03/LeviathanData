"""Tool: orchestration_dbt
dbt data transformation CLI integration.

Supported operations:
- run: Run dbt models
- test: Run dbt tests
- build: Build (run + test)
- compile: Compile models
- docs: Generate documentation
- seed: Load seed data
- snapshot: Run snapshots
- list: List resources
- init: Initialize project
"""
from typing import Any, Dict, List, Optional
import json
import subprocess
import shutil
import os


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None


def _check_dbt() -> bool:
    """Check if dbt is installed."""
    return shutil.which("dbt") is not None


def _run_dbt(
    command: str,
    args: List[str],
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run dbt command."""
    cmd = ["dbt", command]
    cmd.extend(args)
    
    if project_dir:
        cmd.extend(["--project-dir", project_dir])
    
    if profiles_dir:
        cmd.extend(["--profiles-dir", profiles_dir])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _run_models(
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    models: Optional[List[str]] = None,
    select: Optional[str] = None,
    exclude: Optional[str] = None,
    full_refresh: bool = False,
) -> Dict[str, Any]:
    """Run dbt models."""
    args = []
    
    if models:
        args.extend(["--models"] + models)
    elif select:
        args.extend(["--select", select])
    
    if exclude:
        args.extend(["--exclude", exclude])
    
    if full_refresh:
        args.append("--full-refresh")
    
    return _run_dbt("run", args, project_dir, profiles_dir)


def _run_tests(
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    select: Optional[str] = None,
) -> Dict[str, Any]:
    """Run dbt tests."""
    args = []
    
    if select:
        args.extend(["--select", select])
    
    return _run_dbt("test", args, project_dir, profiles_dir)


def _build(
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    select: Optional[str] = None,
) -> Dict[str, Any]:
    """Run dbt build (run + test)."""
    args = []
    
    if select:
        args.extend(["--select", select])
    
    return _run_dbt("build", args, project_dir, profiles_dir)


def _compile(
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    select: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile dbt models."""
    args = []
    
    if select:
        args.extend(["--select", select])
    
    return _run_dbt("compile", args, project_dir, profiles_dir)


def _generate_docs(
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate dbt documentation."""
    return _run_dbt("docs", ["generate"], project_dir, profiles_dir)


def _seed(
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    select: Optional[str] = None,
    full_refresh: bool = False,
) -> Dict[str, Any]:
    """Load seed data."""
    args = []
    
    if select:
        args.extend(["--select", select])
    
    if full_refresh:
        args.append("--full-refresh")
    
    return _run_dbt("seed", args, project_dir, profiles_dir)


def _snapshot(
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    select: Optional[str] = None,
) -> Dict[str, Any]:
    """Run snapshots."""
    args = []
    
    if select:
        args.extend(["--select", select])
    
    return _run_dbt("snapshot", args, project_dir, profiles_dir)


def _list_resources(
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    resource_type: Optional[str] = None,
    output_format: str = "json",
) -> Dict[str, Any]:
    """List dbt resources."""
    args = ["--output", output_format]
    
    if resource_type:
        args.extend(["--resource-type", resource_type])
    
    result = _run_dbt("list", args, project_dir, profiles_dir)
    
    if result["success"] and output_format == "json":
        try:
            lines = result["stdout"].strip().split("\n")
            resources = [json.loads(line) for line in lines if line]
            result["resources"] = resources
        except json.JSONDecodeError:
            result["resources"] = result["stdout"].strip().split("\n")
    
    return result


def _init(
    project_name: str,
    directory: Optional[str] = None,
) -> Dict[str, Any]:
    """Initialize dbt project."""
    args = [project_name]
    
    cwd = directory or os.getcwd()
    
    cmd = ["dbt", "init", project_name]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=60,
        )
        
        return {
            "success": result.returncode == 0,
            "project_name": project_name,
            "path": os.path.join(cwd, project_name),
            "stdout": result.stdout,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _create_model(
    project_dir: str,
    model_name: str,
    sql: str,
    schema: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a dbt model file."""
    models_dir = os.path.join(project_dir, "models")
    
    if schema:
        models_dir = os.path.join(models_dir, schema)
    
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f"{model_name}.sql")
    
    with open(model_path, "w") as f:
        f.write(sql)
    
    return {
        "created": True,
        "path": model_path,
        "model_name": model_name,
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run dbt operations."""
    args = args or {}
    operation = args.get("operation", "list")
    
    if not _check_dbt():
        return {
            "tool": "orchestration_dbt",
            "status": "error",
            "error": "dbt not found. Install with: pip install dbt-core dbt-postgres",
        }
    
    project_dir = args.get("project_dir")
    profiles_dir = args.get("profiles_dir")
    
    try:
        if operation == "run":
            result = _run_models(
                project_dir=project_dir,
                profiles_dir=profiles_dir,
                models=args.get("models"),
                select=args.get("select"),
                exclude=args.get("exclude"),
                full_refresh=args.get("full_refresh", False),
            )
        
        elif operation == "test":
            result = _run_tests(
                project_dir=project_dir,
                profiles_dir=profiles_dir,
                select=args.get("select"),
            )
        
        elif operation == "build":
            result = _build(
                project_dir=project_dir,
                profiles_dir=profiles_dir,
                select=args.get("select"),
            )
        
        elif operation == "compile":
            result = _compile(
                project_dir=project_dir,
                profiles_dir=profiles_dir,
                select=args.get("select"),
            )
        
        elif operation == "docs":
            result = _generate_docs(
                project_dir=project_dir,
                profiles_dir=profiles_dir,
            )
        
        elif operation == "seed":
            result = _seed(
                project_dir=project_dir,
                profiles_dir=profiles_dir,
                select=args.get("select"),
                full_refresh=args.get("full_refresh", False),
            )
        
        elif operation == "snapshot":
            result = _snapshot(
                project_dir=project_dir,
                profiles_dir=profiles_dir,
                select=args.get("select"),
            )
        
        elif operation == "list":
            result = _list_resources(
                project_dir=project_dir,
                profiles_dir=profiles_dir,
                resource_type=args.get("resource_type"),
            )
        
        elif operation == "init":
            result = _init(
                project_name=args.get("project_name", "my_dbt_project"),
                directory=args.get("directory"),
            )
        
        elif operation == "create_model":
            result = _create_model(
                project_dir=args.get("project_dir", "."),
                model_name=args.get("model_name", "new_model"),
                sql=args.get("sql", "SELECT 1 as id"),
                schema=args.get("schema"),
            )
        
        else:
            return {"tool": "orchestration_dbt", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "orchestration_dbt", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "orchestration_dbt", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "run": {
            "operation": "run",
            "project_dir": "./my_dbt_project",
            "select": "staging.*",
        },
        "test": {
            "operation": "test",
            "project_dir": "./my_dbt_project",
        },
        "create_model": {
            "operation": "create_model",
            "project_dir": "./my_dbt_project",
            "model_name": "dim_customers",
            "sql": """{{ config(materialized='table') }}

SELECT
    id,
    name,
    email,
    created_at
FROM {{ source('raw', 'customers') }}""",
            "schema": "marts",
        },
    }
