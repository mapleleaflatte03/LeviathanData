"""Tool: viz_superset
Apache Superset API integration.

Supported operations:
- login: Authenticate with Superset
- list_dashboards: List all dashboards
- get_dashboard: Get dashboard details
- list_charts: List all charts
- get_chart: Get chart details
- export_dashboard: Export dashboard
- query: Execute SQL query
- list_databases: List databases
"""
from typing import Any, Dict, List, Optional
import json


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


requests = _optional_import("requests")

# Session cache
_session_cache: Dict[str, Any] = {}


class SupersetClient:
    """Superset API client."""
    
    def __init__(
        self,
        base_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        access_token: Optional[str] = None,
    ):
        if requests is None:
            raise ImportError("requests is not installed")
        
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.access_token = access_token
        
        if access_token:
            self.session.headers["Authorization"] = f"Bearer {access_token}"
        elif username and password:
            self._login(username, password)
    
    def _login(self, username: str, password: str) -> None:
        """Login to get access token."""
        login_url = f"{self.base_url}/api/v1/security/login"
        payload = {
            "username": username,
            "password": password,
            "provider": "db",
        }
        
        response = self.session.post(login_url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        self.access_token = data.get("access_token")
        self.session.headers["Authorization"] = f"Bearer {self.access_token}"
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request."""
        url = f"{self.base_url}/api/v1/{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def _post(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make POST request."""
        url = f"{self.base_url}/api/v1/{endpoint}"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def list_dashboards(self, page: int = 0, page_size: int = 20) -> Dict:
        """List all dashboards."""
        return self._get("dashboard/", {"q": json.dumps({
            "page": page,
            "page_size": page_size,
        })})
    
    def get_dashboard(self, dashboard_id: int) -> Dict:
        """Get dashboard details."""
        return self._get(f"dashboard/{dashboard_id}")
    
    def list_charts(self, page: int = 0, page_size: int = 20) -> Dict:
        """List all charts."""
        return self._get("chart/", {"q": json.dumps({
            "page": page,
            "page_size": page_size,
        })})
    
    def get_chart(self, chart_id: int) -> Dict:
        """Get chart details."""
        return self._get(f"chart/{chart_id}")
    
    def get_chart_data(self, chart_id: int) -> Dict:
        """Get chart data."""
        return self._get(f"chart/{chart_id}/data/")
    
    def list_databases(self) -> Dict:
        """List all databases."""
        return self._get("database/")
    
    def execute_query(
        self,
        database_id: int,
        sql: str,
        schema: Optional[str] = None,
    ) -> Dict:
        """Execute SQL query."""
        payload = {
            "database_id": database_id,
            "sql": sql,
        }
        if schema:
            payload["schema"] = schema
        
        return self._post("sqllab/execute/", payload)
    
    def export_dashboard(self, dashboard_id: int) -> bytes:
        """Export dashboard as JSON."""
        url = f"{self.base_url}/api/v1/dashboard/export/"
        response = self.session.get(url, params={"q": json.dumps([dashboard_id])})
        response.raise_for_status()
        return response.content


def _get_client(
    base_url: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    access_token: Optional[str] = None,
) -> SupersetClient:
    """Get or create Superset client."""
    cache_key = base_url
    
    if cache_key not in _session_cache:
        _session_cache[cache_key] = SupersetClient(
            base_url=base_url,
            username=username,
            password=password,
            access_token=access_token,
        )
    
    return _session_cache[cache_key]


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Superset operations."""
    args = args or {}
    operation = args.get("operation", "list_dashboards")
    
    if requests is None:
        return {
            "tool": "viz_superset",
            "status": "error",
            "error": "requests not installed. Run: pip install requests",
        }
    
    base_url = args.get("base_url", "http://localhost:8088")
    
    try:
        client = _get_client(
            base_url=base_url,
            username=args.get("username"),
            password=args.get("password"),
            access_token=args.get("access_token"),
        )
        
        if operation == "login":
            return {
                "tool": "viz_superset",
                "status": "ok",
                "logged_in": True,
                "access_token": client.access_token,
            }
        
        elif operation == "list_dashboards":
            result = client.list_dashboards(
                page=args.get("page", 0),
                page_size=args.get("page_size", 20),
            )
            return {"tool": "viz_superset", "status": "ok", **result}
        
        elif operation == "get_dashboard":
            result = client.get_dashboard(args.get("dashboard_id", 0))
            return {"tool": "viz_superset", "status": "ok", **result}
        
        elif operation == "list_charts":
            result = client.list_charts(
                page=args.get("page", 0),
                page_size=args.get("page_size", 20),
            )
            return {"tool": "viz_superset", "status": "ok", **result}
        
        elif operation == "get_chart":
            result = client.get_chart(args.get("chart_id", 0))
            return {"tool": "viz_superset", "status": "ok", **result}
        
        elif operation == "get_chart_data":
            result = client.get_chart_data(args.get("chart_id", 0))
            return {"tool": "viz_superset", "status": "ok", **result}
        
        elif operation == "list_databases":
            result = client.list_databases()
            return {"tool": "viz_superset", "status": "ok", **result}
        
        elif operation == "query":
            result = client.execute_query(
                database_id=args.get("database_id", 0),
                sql=args.get("sql", ""),
                schema=args.get("schema"),
            )
            return {"tool": "viz_superset", "status": "ok", **result}
        
        elif operation == "export_dashboard":
            data = client.export_dashboard(args.get("dashboard_id", 0))
            return {
                "tool": "viz_superset",
                "status": "ok",
                "exported": True,
                "size": len(data),
            }
        
        else:
            return {"tool": "viz_superset", "status": "error", "error": f"Unknown operation: {operation}"}
    
    except Exception as e:
        return {"tool": "viz_superset", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "login": {
            "operation": "login",
            "base_url": "http://localhost:8088",
            "username": "admin",
            "password": "admin",
        },
        "list_dashboards": {
            "operation": "list_dashboards",
            "base_url": "http://localhost:8088",
            "access_token": "YOUR_TOKEN",
        },
        "query": {
            "operation": "query",
            "base_url": "http://localhost:8088",
            "database_id": 1,
            "sql": "SELECT * FROM table LIMIT 10",
        },
    }
