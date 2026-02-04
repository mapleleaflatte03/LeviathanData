"""Tool: orchestration_nifi
Apache NiFi flow management via REST API.

Supported operations:
- list_process_groups: List process groups
- get_process_group: Get process group details
- list_processors: List processors
- start_processor: Start a processor
- stop_processor: Stop a processor
- list_connections: List connections
- get_flow_status: Get flow status
- upload_template: Upload template
- instantiate_template: Create flow from template
"""
from typing import Any, Dict, List, Optional
import json
import base64


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


requests = _optional_import("requests")


class NiFiClient:
    """NiFi REST API client."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080/nifi-api",
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
    ):
        if requests is None:
            raise ImportError("requests not installed")
        
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        elif username and password:
            # Get access token
            auth_response = self.session.post(
                f"{self.base_url}/access/token",
                data={"username": username, "password": password},
            )
            if auth_response.status_code == 201:
                self.session.headers["Authorization"] = f"Bearer {auth_response.text}"
    
    def _get(self, endpoint: str) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/{endpoint}")
        response.raise_for_status()
        return response.json()
    
    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/{endpoint}",
            json=data,
        )
        response.raise_for_status()
        return response.json()
    
    def _put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.put(
            f"{self.base_url}/{endpoint}",
            json=data,
        )
        response.raise_for_status()
        return response.json()
    
    def get_root_process_group(self) -> Dict[str, Any]:
        """Get root process group."""
        return self._get("flow/process-groups/root")
    
    def list_process_groups(self, parent_id: str = "root") -> List[Dict[str, Any]]:
        """List process groups."""
        data = self._get(f"flow/process-groups/{parent_id}")
        groups = data.get("processGroupFlow", {}).get("flow", {}).get("processGroups", [])
        return [
            {
                "id": g.get("id"),
                "name": g.get("component", {}).get("name"),
                "running_count": g.get("runningCount", 0),
                "stopped_count": g.get("stoppedCount", 0),
            }
            for g in groups
        ]
    
    def get_process_group(self, group_id: str) -> Dict[str, Any]:
        """Get process group details."""
        return self._get(f"process-groups/{group_id}")
    
    def list_processors(self, group_id: str = "root") -> List[Dict[str, Any]]:
        """List processors in group."""
        data = self._get(f"flow/process-groups/{group_id}")
        processors = data.get("processGroupFlow", {}).get("flow", {}).get("processors", [])
        return [
            {
                "id": p.get("id"),
                "name": p.get("component", {}).get("name"),
                "type": p.get("component", {}).get("type"),
                "state": p.get("component", {}).get("state"),
            }
            for p in processors
        ]
    
    def get_processor(self, processor_id: str) -> Dict[str, Any]:
        """Get processor details."""
        return self._get(f"processors/{processor_id}")
    
    def update_processor_state(self, processor_id: str, state: str) -> Dict[str, Any]:
        """Update processor state (RUNNING/STOPPED)."""
        processor = self.get_processor(processor_id)
        revision = processor.get("revision", {})
        
        return self._put(
            f"processors/{processor_id}/run-status",
            {
                "revision": revision,
                "state": state,
            },
        )
    
    def list_connections(self, group_id: str = "root") -> List[Dict[str, Any]]:
        """List connections in group."""
        data = self._get(f"flow/process-groups/{group_id}")
        connections = data.get("processGroupFlow", {}).get("flow", {}).get("connections", [])
        return [
            {
                "id": c.get("id"),
                "source_id": c.get("sourceId"),
                "destination_id": c.get("destinationId"),
                "queued_count": c.get("status", {}).get("aggregateSnapshot", {}).get("queuedCount"),
            }
            for c in connections
        ]
    
    def get_flow_status(self) -> Dict[str, Any]:
        """Get overall flow status."""
        data = self._get("flow/status")
        status = data.get("controllerStatus", {})
        return {
            "active_threads": status.get("activeThreadCount", 0),
            "running_components": status.get("runningCount", 0),
            "stopped_components": status.get("stoppedCount", 0),
            "invalid_components": status.get("invalidCount", 0),
            "queued_count": status.get("flowFilesQueued", 0),
        }
    
    def upload_template(self, group_id: str, template_xml: str) -> Dict[str, Any]:
        """Upload template XML."""
        response = self.session.post(
            f"{self.base_url}/process-groups/{group_id}/templates/upload",
            files={"template": ("template.xml", template_xml, "application/xml")},
        )
        response.raise_for_status()
        return response.json()


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run NiFi operations."""
    args = args or {}
    operation = args.get("operation", "get_flow_status")
    
    try:
        client = NiFiClient(
            base_url=args.get("base_url", "http://localhost:8080/nifi-api"),
            username=args.get("username"),
            password=args.get("password"),
            token=args.get("token"),
        )
        
        if operation == "list_process_groups":
            groups = client.list_process_groups(args.get("parent_id", "root"))
            result = {"process_groups": groups}
        
        elif operation == "get_process_group":
            data = client.get_process_group(args.get("group_id", "root"))
            result = {"process_group": data}
        
        elif operation == "list_processors":
            processors = client.list_processors(args.get("group_id", "root"))
            result = {"processors": processors}
        
        elif operation == "start_processor":
            data = client.update_processor_state(args.get("processor_id", ""), "RUNNING")
            result = {"started": True, "processor": data}
        
        elif operation == "stop_processor":
            data = client.update_processor_state(args.get("processor_id", ""), "STOPPED")
            result = {"stopped": True, "processor": data}
        
        elif operation == "list_connections":
            connections = client.list_connections(args.get("group_id", "root"))
            result = {"connections": connections}
        
        elif operation == "get_flow_status":
            status = client.get_flow_status()
            result = {"status": status}
        
        elif operation == "upload_template":
            data = client.upload_template(
                args.get("group_id", "root"),
                args.get("template_xml", ""),
            )
            result = {"uploaded": True, "template": data}
        
        else:
            return {"tool": "orchestration_nifi", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "orchestration_nifi", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "orchestration_nifi", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "get_flow_status": {
            "operation": "get_flow_status",
            "base_url": "http://localhost:8080/nifi-api",
        },
        "list_processors": {
            "operation": "list_processors",
            "group_id": "root",
        },
        "start_processor": {
            "operation": "start_processor",
            "processor_id": "abc-123-def",
        },
    }
