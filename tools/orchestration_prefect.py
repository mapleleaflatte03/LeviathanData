"""Tool: orchestration_prefect
Prefect workflow orchestration.

Supported operations:
- create_flow: Create a Prefect flow
- run_flow: Execute a flow
- list_flows: List flows
- list_runs: List flow runs
- get_run: Get run details
- create_deployment: Create deployment
- schedule: Schedule a deployment
"""
from typing import Any, Callable, Dict, List, Optional
import json
import asyncio
from functools import wraps


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


prefect = _optional_import("prefect")


# Store for dynamically created flows
_flows: Dict[str, Any] = {}
_results: Dict[str, Any] = {}


def _create_flow(
    name: str,
    tasks: List[Dict[str, Any]],
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a Prefect flow from task definitions."""
    if prefect is None:
        raise ImportError("prefect not installed. Run: pip install prefect")
    
    from prefect import flow, task
    
    # Create task functions
    task_funcs = {}
    
    for task_def in tasks:
        task_name = task_def.get("name", "task")
        task_type = task_def.get("type", "python")
        code = task_def.get("code", "pass")
        
        # Create task function dynamically
        exec_globals = {"__builtins__": __builtins__}
        exec(f"def {task_name}():\n    {code}", exec_globals)
        task_funcs[task_name] = task(name=task_name)(exec_globals[task_name])
    
    # Create flow
    @flow(name=name, description=description)
    def dynamic_flow():
        results = {}
        for task_name, task_func in task_funcs.items():
            results[task_name] = task_func()
        return results
    
    _flows[name] = dynamic_flow
    
    return {
        "created": True,
        "name": name,
        "task_count": len(tasks),
    }


def _run_flow(
    name: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a flow."""
    if prefect is None:
        raise ImportError("prefect not installed")
    
    if name not in _flows:
        raise ValueError(f"Flow '{name}' not found")
    
    flow_func = _flows[name]
    
    # Run the flow
    result = flow_func()
    
    return {
        "executed": True,
        "name": name,
        "result": result,
    }


def _list_flows() -> Dict[str, Any]:
    """List all registered flows."""
    if prefect is None:
        return {"flows": list(_flows.keys()), "source": "local"}
    
    # Try to list from Prefect server
    try:
        from prefect.client import get_client
        
        async def _get_flows():
            async with get_client() as client:
                flows = await client.read_flows()
                return [f.name for f in flows]
        
        flow_names = asyncio.run(_get_flows())
        return {"flows": flow_names, "source": "server"}
    except Exception:
        return {"flows": list(_flows.keys()), "source": "local"}


def _list_runs(
    flow_name: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """List flow runs."""
    if prefect is None:
        raise ImportError("prefect not installed")
    
    from prefect.client import get_client
    
    async def _get_runs():
        async with get_client() as client:
            runs = await client.read_flow_runs(limit=limit)
            return [
                {
                    "id": str(r.id),
                    "name": r.name,
                    "state": r.state.type.value if r.state else None,
                    "created": str(r.created) if r.created else None,
                }
                for r in runs
            ]
    
    try:
        runs = asyncio.run(_get_runs())
        return {"runs": runs}
    except Exception as e:
        return {"error": str(e), "runs": []}


def _get_run(run_id: str) -> Dict[str, Any]:
    """Get flow run details."""
    if prefect is None:
        raise ImportError("prefect not installed")
    
    from prefect.client import get_client
    import uuid
    
    async def _get():
        async with get_client() as client:
            run = await client.read_flow_run(uuid.UUID(run_id))
            return {
                "id": str(run.id),
                "name": run.name,
                "state": run.state.type.value if run.state else None,
                "parameters": run.parameters,
            }
    
    try:
        return asyncio.run(_get())
    except Exception as e:
        return {"error": str(e)}


def _create_deployment(
    flow_name: str,
    deployment_name: str,
    schedule: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a deployment."""
    if prefect is None:
        raise ImportError("prefect not installed")
    
    if flow_name not in _flows:
        raise ValueError(f"Flow '{flow_name}' not found")
    
    from prefect.deployments import Deployment
    
    flow_func = _flows[flow_name]
    
    deployment = Deployment.build_from_flow(
        flow=flow_func,
        name=deployment_name,
        parameters=parameters or {},
    )
    
    # Apply deployment
    deployment_id = deployment.apply()
    
    return {
        "created": True,
        "deployment_id": str(deployment_id),
        "deployment_name": deployment_name,
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Prefect operations."""
    args = args or {}
    operation = args.get("operation", "list_flows")
    
    try:
        if operation == "create_flow":
            result = _create_flow(
                name=args.get("name", "my_flow"),
                tasks=args.get("tasks", []),
                description=args.get("description"),
            )
        
        elif operation == "run_flow":
            result = _run_flow(
                name=args.get("name", ""),
                parameters=args.get("parameters"),
            )
        
        elif operation == "list_flows":
            result = _list_flows()
        
        elif operation == "list_runs":
            result = _list_runs(
                flow_name=args.get("flow_name"),
                limit=args.get("limit", 10),
            )
        
        elif operation == "get_run":
            result = _get_run(run_id=args.get("run_id", ""))
        
        elif operation == "create_deployment":
            result = _create_deployment(
                flow_name=args.get("flow_name", ""),
                deployment_name=args.get("deployment_name", "default"),
                schedule=args.get("schedule"),
                parameters=args.get("parameters"),
            )
        
        else:
            return {"tool": "orchestration_prefect", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "orchestration_prefect", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "orchestration_prefect", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "create_flow": {
            "operation": "create_flow",
            "name": "etl_pipeline",
            "description": "ETL pipeline for data processing",
            "tasks": [
                {"name": "extract", "code": "return {'data': [1, 2, 3]}"},
                {"name": "transform", "code": "return {'transformed': True}"},
                {"name": "load", "code": "return {'loaded': True}"},
            ],
        },
        "run_flow": {
            "operation": "run_flow",
            "name": "etl_pipeline",
        },
        "list_flows": {
            "operation": "list_flows",
        },
    }
