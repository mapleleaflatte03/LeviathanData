"""Tool: orchestration_mage
Mage AI data pipeline management.

Supported operations:
- list_pipelines: List all pipelines
- get_pipeline: Get pipeline details
- run_pipeline: Trigger pipeline run
- list_blocks: List blocks in pipeline
- create_pipeline: Create new pipeline
- create_block: Create block in pipeline
- get_runs: Get pipeline runs
"""
from typing import Any, Dict, List, Optional
import json


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


requests = _optional_import("requests")


class MageClient:
    """Mage AI API client."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:6789",
        api_key: Optional[str] = None,
    ):
        if requests is None:
            raise ImportError("requests not installed")
        
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        
        self.session.headers["Content-Type"] = "application/json"
    
    def _get(self, endpoint: str) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/api/{endpoint}")
        response.raise_for_status()
        return response.json()
    
    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/api/{endpoint}",
            json=data,
        )
        response.raise_for_status()
        return response.json()
    
    def _put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.put(
            f"{self.base_url}/api/{endpoint}",
            json=data,
        )
        response.raise_for_status()
        return response.json()
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipelines."""
        data = self._get("pipelines")
        return data.get("pipelines", [])
    
    def get_pipeline(self, pipeline_uuid: str) -> Dict[str, Any]:
        """Get pipeline details."""
        return self._get(f"pipelines/{pipeline_uuid}")
    
    def create_pipeline(
        self,
        name: str,
        pipeline_type: str = "python",
    ) -> Dict[str, Any]:
        """Create new pipeline."""
        return self._post(
            "pipelines",
            {
                "pipeline": {
                    "name": name,
                    "type": pipeline_type,
                },
            },
        )
    
    def list_blocks(self, pipeline_uuid: str) -> List[Dict[str, Any]]:
        """List blocks in pipeline."""
        data = self._get(f"pipelines/{pipeline_uuid}")
        return data.get("pipeline", {}).get("blocks", [])
    
    def create_block(
        self,
        pipeline_uuid: str,
        name: str,
        block_type: str,
        language: str = "python",
        content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create block in pipeline."""
        block_data = {
            "block": {
                "name": name,
                "type": block_type,
                "language": language,
            },
        }
        
        if content:
            block_data["block"]["content"] = content
        
        return self._post(f"pipelines/{pipeline_uuid}/blocks", block_data)
    
    def run_pipeline(
        self,
        pipeline_uuid: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Trigger pipeline run."""
        run_data = {
            "pipeline_run": {
                "pipeline_uuid": pipeline_uuid,
                "variables": variables or {},
            },
        }
        return self._post("pipeline_runs", run_data)
    
    def get_runs(self, pipeline_uuid: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get pipeline runs."""
        endpoint = "pipeline_runs"
        if pipeline_uuid:
            endpoint = f"pipelines/{pipeline_uuid}/pipeline_runs"
        
        data = self._get(endpoint)
        return data.get("pipeline_runs", [])
    
    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Get specific run details."""
        return self._get(f"pipeline_runs/{run_id}")


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Mage operations."""
    args = args or {}
    operation = args.get("operation", "list_pipelines")
    
    try:
        client = MageClient(
            base_url=args.get("base_url", "http://localhost:6789"),
            api_key=args.get("api_key"),
        )
        
        if operation == "list_pipelines":
            pipelines = client.list_pipelines()
            result = {"pipelines": pipelines}
        
        elif operation == "get_pipeline":
            data = client.get_pipeline(args.get("pipeline_uuid", ""))
            result = {"pipeline": data}
        
        elif operation == "create_pipeline":
            data = client.create_pipeline(
                name=args.get("name", "new_pipeline"),
                pipeline_type=args.get("pipeline_type", "python"),
            )
            result = {"created": True, "pipeline": data}
        
        elif operation == "list_blocks":
            blocks = client.list_blocks(args.get("pipeline_uuid", ""))
            result = {"blocks": blocks}
        
        elif operation == "create_block":
            data = client.create_block(
                pipeline_uuid=args.get("pipeline_uuid", ""),
                name=args.get("name", "new_block"),
                block_type=args.get("block_type", "data_loader"),
                language=args.get("language", "python"),
                content=args.get("content"),
            )
            result = {"created": True, "block": data}
        
        elif operation == "run_pipeline":
            data = client.run_pipeline(
                pipeline_uuid=args.get("pipeline_uuid", ""),
                variables=args.get("variables"),
            )
            result = {"triggered": True, "run": data}
        
        elif operation == "get_runs":
            runs = client.get_runs(args.get("pipeline_uuid"))
            result = {"runs": runs}
        
        elif operation == "get_run":
            data = client.get_run(args.get("run_id", ""))
            result = {"run": data}
        
        else:
            return {"tool": "orchestration_mage", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "orchestration_mage", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "orchestration_mage", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "list_pipelines": {
            "operation": "list_pipelines",
            "base_url": "http://localhost:6789",
        },
        "create_pipeline": {
            "operation": "create_pipeline",
            "name": "etl_pipeline",
            "pipeline_type": "python",
        },
        "create_block": {
            "operation": "create_block",
            "pipeline_uuid": "etl_pipeline",
            "name": "load_data",
            "block_type": "data_loader",
            "content": """import pandas as pd

@data_loader
def load_data():
    return pd.read_csv('data.csv')""",
        },
        "run_pipeline": {
            "operation": "run_pipeline",
            "pipeline_uuid": "etl_pipeline",
            "variables": {"date": "2024-01-01"},
        },
    }
