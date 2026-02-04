"""Tool: viz_metabase
Metabase API integration.

Supported operations:
- login: Authenticate with Metabase
- list_dashboards: List all dashboards
- get_dashboard: Get dashboard details
- list_questions: List all questions/cards
- get_question: Get question details
- run_question: Run a saved question
- query: Execute native query
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


class MetabaseClient:
    """Metabase API client."""
    
    def __init__(
        self,
        base_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        session_token: Optional[str] = None,
    ):
        if requests is None:
            raise ImportError("requests is not installed")
        
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session_token = session_token
        
        if session_token:
            self.session.headers["X-Metabase-Session"] = session_token
        elif username and password:
            self._login(username, password)
    
    def _login(self, username: str, password: str) -> None:
        """Login to get session token."""
        login_url = f"{self.base_url}/api/session"
        payload = {"username": username, "password": password}
        
        response = self.session.post(login_url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        self.session_token = data.get("id")
        self.session.headers["X-Metabase-Session"] = self.session_token
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request."""
        url = f"{self.base_url}/api/{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def _post(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make POST request."""
        url = f"{self.base_url}/api/{endpoint}"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def list_dashboards(self) -> List[Dict]:
        """List all dashboards."""
        return self._get("dashboard")
    
    def get_dashboard(self, dashboard_id: int) -> Dict:
        """Get dashboard details."""
        return self._get(f"dashboard/{dashboard_id}")
    
    def list_questions(self) -> List[Dict]:
        """List all questions/cards."""
        return self._get("card")
    
    def get_question(self, question_id: int) -> Dict:
        """Get question details."""
        return self._get(f"card/{question_id}")
    
    def run_question(
        self,
        question_id: int,
        parameters: Optional[Dict] = None,
    ) -> Dict:
        """Run a saved question."""
        endpoint = f"card/{question_id}/query"
        payload = {}
        if parameters:
            payload["parameters"] = parameters
        return self._post(endpoint, payload)
    
    def list_databases(self) -> List[Dict]:
        """List all databases."""
        return self._get("database")
    
    def get_database(self, database_id: int) -> Dict:
        """Get database details."""
        return self._get(f"database/{database_id}")
    
    def execute_query(
        self,
        database_id: int,
        query: str,
    ) -> Dict:
        """Execute native SQL query."""
        payload = {
            "database": database_id,
            "type": "native",
            "native": {"query": query},
        }
        return self._post("dataset", payload)
    
    def get_collections(self) -> List[Dict]:
        """List all collections."""
        return self._get("collection")
    
    def create_question(
        self,
        name: str,
        database_id: int,
        query: str,
        collection_id: Optional[int] = None,
        visualization: str = "table",
    ) -> Dict:
        """Create a new question."""
        payload = {
            "name": name,
            "dataset_query": {
                "database": database_id,
                "type": "native",
                "native": {"query": query},
            },
            "display": visualization,
            "visualization_settings": {},
        }
        if collection_id:
            payload["collection_id"] = collection_id
        
        return self._post("card", payload)


def _get_client(
    base_url: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    session_token: Optional[str] = None,
) -> MetabaseClient:
    """Get or create Metabase client."""
    cache_key = base_url
    
    if cache_key not in _session_cache:
        _session_cache[cache_key] = MetabaseClient(
            base_url=base_url,
            username=username,
            password=password,
            session_token=session_token,
        )
    
    return _session_cache[cache_key]


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Metabase operations."""
    args = args or {}
    operation = args.get("operation", "list_dashboards")
    
    if requests is None:
        return {
            "tool": "viz_metabase",
            "status": "error",
            "error": "requests not installed. Run: pip install requests",
        }
    
    base_url = args.get("base_url", "http://localhost:3000")
    
    try:
        client = _get_client(
            base_url=base_url,
            username=args.get("username"),
            password=args.get("password"),
            session_token=args.get("session_token"),
        )
        
        if operation == "login":
            return {
                "tool": "viz_metabase",
                "status": "ok",
                "logged_in": True,
                "session_token": client.session_token,
            }
        
        elif operation == "list_dashboards":
            dashboards = client.list_dashboards()
            return {
                "tool": "viz_metabase",
                "status": "ok",
                "dashboards": dashboards,
                "count": len(dashboards),
            }
        
        elif operation == "get_dashboard":
            result = client.get_dashboard(args.get("dashboard_id", 0))
            return {"tool": "viz_metabase", "status": "ok", **result}
        
        elif operation == "list_questions":
            questions = client.list_questions()
            return {
                "tool": "viz_metabase",
                "status": "ok",
                "questions": questions,
                "count": len(questions),
            }
        
        elif operation == "get_question":
            result = client.get_question(args.get("question_id", 0))
            return {"tool": "viz_metabase", "status": "ok", **result}
        
        elif operation == "run_question":
            result = client.run_question(
                question_id=args.get("question_id", 0),
                parameters=args.get("parameters"),
            )
            return {"tool": "viz_metabase", "status": "ok", **result}
        
        elif operation == "list_databases":
            databases = client.list_databases()
            return {
                "tool": "viz_metabase",
                "status": "ok",
                "databases": databases,
            }
        
        elif operation == "query":
            result = client.execute_query(
                database_id=args.get("database_id", 0),
                query=args.get("query", ""),
            )
            return {"tool": "viz_metabase", "status": "ok", **result}
        
        elif operation == "collections":
            collections = client.get_collections()
            return {
                "tool": "viz_metabase",
                "status": "ok",
                "collections": collections,
            }
        
        elif operation == "create_question":
            result = client.create_question(
                name=args.get("name", "New Question"),
                database_id=args.get("database_id", 0),
                query=args.get("query", ""),
                collection_id=args.get("collection_id"),
                visualization=args.get("visualization", "table"),
            )
            return {"tool": "viz_metabase", "status": "ok", **result}
        
        else:
            return {"tool": "viz_metabase", "status": "error", "error": f"Unknown operation: {operation}"}
    
    except Exception as e:
        return {"tool": "viz_metabase", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "login": {
            "operation": "login",
            "base_url": "http://localhost:3000",
            "username": "admin@example.com",
            "password": "password",
        },
        "query": {
            "operation": "query",
            "base_url": "http://localhost:3000",
            "database_id": 1,
            "query": "SELECT * FROM users LIMIT 10",
        },
        "run_question": {
            "operation": "run_question",
            "question_id": 1,
            "parameters": {"date": "2024-01-01"},
        },
    }
