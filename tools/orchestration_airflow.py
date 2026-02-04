"""Tool: orchestration_airflow
Apache Airflow DAG orchestration via REST API.

Supported operations:
- list_dags: List all DAGs
- get_dag: Get DAG details
- trigger_dag: Trigger a DAG run
- list_dag_runs: List DAG runs
- get_dag_run: Get DAG run status
- pause_dag: Pause a DAG
- unpause_dag: Unpause a DAG
- list_tasks: List tasks in a DAG
- get_task_instances: Get task instance status
"""
from typing import Any, Dict, List, Optional
import json
import os
from datetime import datetime


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


requests = _optional_import("requests")

# Default Airflow configuration
DEFAULT_AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")
DEFAULT_AIRFLOW_USER = os.getenv("AIRFLOW_USER", "airflow")
DEFAULT_AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")


class AirflowClient:
    """Client for Airflow REST API."""
    
    def __init__(
        self,
        base_url: str = DEFAULT_AIRFLOW_URL,
        username: str = DEFAULT_AIRFLOW_USER,
        password: str = DEFAULT_AIRFLOW_PASSWORD,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/v1"
        self.auth = (username, password)
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make API request."""
        if requests is None:
            raise ImportError("requests library not installed")
        
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        
        response = requests.request(
            method=method,
            url=url,
            auth=self.auth,
            headers=headers,
            json=data,
            params=params,
            timeout=30,
        )
        
        response.raise_for_status()
        return response.json() if response.text else {}
    
    def list_dags(
        self,
        limit: int = 100,
        offset: int = 0,
        only_active: bool = True,
    ) -> Dict[str, Any]:
        """List all DAGs."""
        params = {
            "limit": limit,
            "offset": offset,
            "only_active": str(only_active).lower(),
        }
        return self._request("GET", "dags", params=params)
    
    def get_dag(self, dag_id: str) -> Dict[str, Any]:
        """Get DAG details."""
        return self._request("GET", f"dags/{dag_id}")
    
    def trigger_dag(
        self,
        dag_id: str,
        conf: Optional[Dict] = None,
        logical_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger a DAG run."""
        data = {}
        if conf:
            data["conf"] = conf
        if logical_date:
            data["logical_date"] = logical_date
        
        return self._request("POST", f"dags/{dag_id}/dagRuns", data=data or None)
    
    def list_dag_runs(
        self,
        dag_id: str,
        limit: int = 25,
        offset: int = 0,
        state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List DAG runs."""
        params = {"limit": limit, "offset": offset}
        if state:
            params["state"] = state
        return self._request("GET", f"dags/{dag_id}/dagRuns", params=params)
    
    def get_dag_run(self, dag_id: str, dag_run_id: str) -> Dict[str, Any]:
        """Get DAG run details."""
        return self._request("GET", f"dags/{dag_id}/dagRuns/{dag_run_id}")
    
    def pause_dag(self, dag_id: str) -> Dict[str, Any]:
        """Pause a DAG."""
        return self._request("PATCH", f"dags/{dag_id}", data={"is_paused": True})
    
    def unpause_dag(self, dag_id: str) -> Dict[str, Any]:
        """Unpause a DAG."""
        return self._request("PATCH", f"dags/{dag_id}", data={"is_paused": False})
    
    def list_tasks(self, dag_id: str) -> Dict[str, Any]:
        """List tasks in a DAG."""
        return self._request("GET", f"dags/{dag_id}/tasks")
    
    def get_task_instances(
        self,
        dag_id: str,
        dag_run_id: str,
    ) -> Dict[str, Any]:
        """Get task instances for a DAG run."""
        return self._request("GET", f"dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances")
    
    def get_task_instance(
        self,
        dag_id: str,
        dag_run_id: str,
        task_id: str,
    ) -> Dict[str, Any]:
        """Get specific task instance."""
        return self._request(
            "GET",
            f"dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}"
        )
    
    def get_task_logs(
        self,
        dag_id: str,
        dag_run_id: str,
        task_id: str,
        task_try_number: int = 1,
    ) -> str:
        """Get task logs."""
        if requests is None:
            raise ImportError("requests library not installed")
        
        url = f"{self.api_url}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{task_try_number}"
        headers = {"Accept": "text/plain"}
        
        response = requests.get(
            url,
            auth=self.auth,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.text


def _get_client(args: Dict[str, Any]) -> AirflowClient:
    """Get Airflow client from args."""
    return AirflowClient(
        base_url=args.get("airflow_url", DEFAULT_AIRFLOW_URL),
        username=args.get("username", DEFAULT_AIRFLOW_USER),
        password=args.get("password", DEFAULT_AIRFLOW_PASSWORD),
    )


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Airflow operations.
    
    Args:
        args: Dictionary with:
            - operation: API operation to perform
            - airflow_url: Airflow base URL
            - username: Airflow username
            - password: Airflow password
            - dag_id: DAG identifier
            - dag_run_id: DAG run identifier
            - Various operation-specific parameters
    
    Returns:
        Result dictionary with operation output
    """
    args = args or {}
    operation = args.get("operation", "list_dags")
    
    if requests is None:
        return {
            "tool": "orchestration_airflow",
            "status": "error",
            "error": "requests library not installed",
        }
    
    try:
        client = _get_client(args)
        
        if operation == "list_dags":
            result = client.list_dags(
                limit=args.get("limit", 100),
                offset=args.get("offset", 0),
                only_active=args.get("only_active", True),
            )
        
        elif operation == "get_dag":
            result = client.get_dag(dag_id=args.get("dag_id", ""))
        
        elif operation == "trigger_dag":
            result = client.trigger_dag(
                dag_id=args.get("dag_id", ""),
                conf=args.get("conf"),
                logical_date=args.get("logical_date"),
            )
        
        elif operation == "list_dag_runs":
            result = client.list_dag_runs(
                dag_id=args.get("dag_id", ""),
                limit=args.get("limit", 25),
                offset=args.get("offset", 0),
                state=args.get("state"),
            )
        
        elif operation == "get_dag_run":
            result = client.get_dag_run(
                dag_id=args.get("dag_id", ""),
                dag_run_id=args.get("dag_run_id", ""),
            )
        
        elif operation == "pause_dag":
            result = client.pause_dag(dag_id=args.get("dag_id", ""))
        
        elif operation == "unpause_dag":
            result = client.unpause_dag(dag_id=args.get("dag_id", ""))
        
        elif operation == "list_tasks":
            result = client.list_tasks(dag_id=args.get("dag_id", ""))
        
        elif operation == "get_task_instances":
            result = client.get_task_instances(
                dag_id=args.get("dag_id", ""),
                dag_run_id=args.get("dag_run_id", ""),
            )
        
        elif operation == "get_task_instance":
            result = client.get_task_instance(
                dag_id=args.get("dag_id", ""),
                dag_run_id=args.get("dag_run_id", ""),
                task_id=args.get("task_id", ""),
            )
        
        elif operation == "get_task_logs":
            logs = client.get_task_logs(
                dag_id=args.get("dag_id", ""),
                dag_run_id=args.get("dag_run_id", ""),
                task_id=args.get("task_id", ""),
                task_try_number=args.get("task_try_number", 1),
            )
            result = {"logs": logs}
        
        else:
            return {
                "tool": "orchestration_airflow",
                "status": "error",
                "error": f"Unknown operation: {operation}",
            }
        
        return {"tool": "orchestration_airflow", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "orchestration_airflow", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "list_dags": {
            "operation": "list_dags",
            "airflow_url": "http://localhost:8080",
            "only_active": True,
        },
        "trigger_dag": {
            "operation": "trigger_dag",
            "dag_id": "example_dag",
            "conf": {"param1": "value1", "param2": "value2"},
        },
        "get_dag_run_status": {
            "operation": "get_dag_run",
            "dag_id": "example_dag",
            "dag_run_id": "manual__2024-01-01T00:00:00+00:00",
        },
        "get_task_logs": {
            "operation": "get_task_logs",
            "dag_id": "example_dag",
            "dag_run_id": "manual__2024-01-01T00:00:00+00:00",
            "task_id": "my_task",
            "task_try_number": 1,
        },
    }
