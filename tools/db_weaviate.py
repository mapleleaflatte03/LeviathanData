"""Tool: db_weaviate
Weaviate vector database for semantic search.

Supported operations:
- create_class: Create a schema class
- delete_class: Delete a schema class
- list_classes: List all schema classes
- insert: Insert objects
- query: Semantic search
- get_by_id: Get object by ID
- update: Update an object
- delete: Delete objects
- batch_insert: Batch insert multiple objects
"""
from typing import Any, Dict, List, Optional
import json
import uuid


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


weaviate = _optional_import("weaviate")

# Client cache
_client_cache: Dict[str, Any] = {}


def _get_client(
    url: str = "http://localhost:8080",
    api_key: Optional[str] = None,
    additional_headers: Optional[Dict] = None,
) -> Any:
    """Get or create Weaviate client."""
    if weaviate is None:
        raise ImportError("weaviate-client is not installed. Run: pip install weaviate-client")
    
    cache_key = url
    
    if cache_key not in _client_cache:
        auth_config = None
        if api_key:
            auth_config = weaviate.auth.AuthApiKey(api_key=api_key)
        
        _client_cache[cache_key] = weaviate.Client(
            url=url,
            auth_client_secret=auth_config,
            additional_headers=additional_headers or {},
        )
    
    return _client_cache[cache_key]


def _create_class(
    class_name: str,
    properties: List[Dict[str, str]],
    vectorizer: str = "none",
    description: Optional[str] = None,
    **client_kwargs,
) -> Dict[str, Any]:
    """Create a schema class."""
    client = _get_client(**client_kwargs)
    
    class_obj = {
        "class": class_name,
        "properties": [
            {
                "name": p["name"],
                "dataType": [p.get("dataType", "text")],
                "description": p.get("description", ""),
            }
            for p in properties
        ],
        "vectorizer": vectorizer,
    }
    
    if description:
        class_obj["description"] = description
    
    client.schema.create_class(class_obj)
    
    return {"created": class_name, "properties": len(properties)}


def _delete_class(class_name: str, **client_kwargs) -> Dict[str, Any]:
    """Delete a schema class."""
    client = _get_client(**client_kwargs)
    client.schema.delete_class(class_name)
    return {"deleted": class_name}


def _list_classes(**client_kwargs) -> Dict[str, Any]:
    """List all schema classes."""
    client = _get_client(**client_kwargs)
    schema = client.schema.get()
    
    return {
        "classes": [
            {
                "name": c["class"],
                "properties": [p["name"] for p in c.get("properties", [])],
                "vectorizer": c.get("vectorizer", "none"),
            }
            for c in schema.get("classes", [])
        ]
    }


def _insert(
    class_name: str,
    properties: Dict[str, Any],
    vector: Optional[List[float]] = None,
    object_id: Optional[str] = None,
    **client_kwargs,
) -> Dict[str, Any]:
    """Insert a single object."""
    client = _get_client(**client_kwargs)
    
    object_id = object_id or str(uuid.uuid4())
    
    kwargs = {
        "data_object": properties,
        "class_name": class_name,
        "uuid": object_id,
    }
    
    if vector:
        kwargs["vector"] = vector
    
    result_id = client.data_object.create(**kwargs)
    
    return {"id": result_id, "class": class_name}


def _batch_insert(
    class_name: str,
    objects: List[Dict[str, Any]],
    **client_kwargs,
) -> Dict[str, Any]:
    """Batch insert multiple objects."""
    client = _get_client(**client_kwargs)
    
    with client.batch as batch:
        for obj in objects:
            properties = obj.get("properties", {})
            vector = obj.get("vector")
            object_id = obj.get("id") or str(uuid.uuid4())
            
            batch.add_data_object(
                data_object=properties,
                class_name=class_name,
                uuid=object_id,
                vector=vector,
            )
    
    return {"inserted": len(objects), "class": class_name}


def _query(
    class_name: str,
    query_vector: Optional[List[float]] = None,
    query_text: Optional[str] = None,
    properties: Optional[List[str]] = None,
    limit: int = 10,
    filters: Optional[Dict] = None,
    **client_kwargs,
) -> Dict[str, Any]:
    """Semantic search query."""
    client = _get_client(**client_kwargs)
    
    query = client.query.get(class_name, properties or ["*"])
    
    if query_vector:
        query = query.with_near_vector({"vector": query_vector})
    elif query_text:
        query = query.with_near_text({"concepts": [query_text]})
    
    query = query.with_limit(limit)
    query = query.with_additional(["id", "distance", "certainty"])
    
    if filters:
        where_filter = {"operator": "And", "operands": []}
        for field, value in filters.items():
            where_filter["operands"].append({
                "path": [field],
                "operator": "Equal",
                "valueText": value if isinstance(value, str) else None,
                "valueNumber": value if isinstance(value, (int, float)) else None,
            })
        if where_filter["operands"]:
            query = query.with_where(where_filter)
    
    result = query.do()
    
    objects = result.get("data", {}).get("Get", {}).get(class_name, [])
    
    return {
        "results": [
            {
                "id": obj.get("_additional", {}).get("id"),
                "distance": obj.get("_additional", {}).get("distance"),
                "certainty": obj.get("_additional", {}).get("certainty"),
                "properties": {k: v for k, v in obj.items() if k != "_additional"},
            }
            for obj in objects
        ],
        "count": len(objects),
    }


def _get_by_id(class_name: str, object_id: str, **client_kwargs) -> Dict[str, Any]:
    """Get object by ID."""
    client = _get_client(**client_kwargs)
    
    result = client.data_object.get_by_id(
        uuid=object_id,
        class_name=class_name,
        with_vector=True,
    )
    
    if result:
        return {
            "id": result.get("id"),
            "class": result.get("class"),
            "properties": result.get("properties", {}),
            "vector": result.get("vector"),
        }
    return {"error": "Object not found"}


def _update(
    class_name: str,
    object_id: str,
    properties: Dict[str, Any],
    vector: Optional[List[float]] = None,
    **client_kwargs,
) -> Dict[str, Any]:
    """Update an object."""
    client = _get_client(**client_kwargs)
    
    kwargs = {
        "data_object": properties,
        "class_name": class_name,
        "uuid": object_id,
    }
    
    if vector:
        kwargs["vector"] = vector
    
    client.data_object.update(**kwargs)
    
    return {"updated": object_id}


def _delete(class_name: str, object_id: str, **client_kwargs) -> Dict[str, Any]:
    """Delete an object."""
    client = _get_client(**client_kwargs)
    client.data_object.delete(uuid=object_id, class_name=class_name)
    return {"deleted": object_id}


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Weaviate operations.
    
    Args:
        args: Dictionary with:
            - operation: Weaviate operation to perform
            - class_name: Name of the class
            - url: Weaviate URL (default: http://localhost:8080)
            - api_key: Optional API key
            - Various operation-specific parameters
    
    Returns:
        Result dictionary with operation output
    """
    args = args or {}
    operation = args.get("operation", "list_classes")
    
    if weaviate is None:
        return {
            "tool": "db_weaviate",
            "status": "error",
            "error": "weaviate-client not installed. Run: pip install weaviate-client",
        }
    
    client_kwargs = {}
    for key in ["url", "api_key", "additional_headers"]:
        if key in args:
            client_kwargs[key] = args[key]
    
    try:
        if operation == "create_class":
            result = _create_class(
                class_name=args.get("class_name", ""),
                properties=args.get("properties", []),
                vectorizer=args.get("vectorizer", "none"),
                description=args.get("description"),
                **client_kwargs,
            )
        
        elif operation == "delete_class":
            result = _delete_class(
                class_name=args.get("class_name", ""),
                **client_kwargs,
            )
        
        elif operation == "list_classes":
            result = _list_classes(**client_kwargs)
        
        elif operation == "insert":
            result = _insert(
                class_name=args.get("class_name", ""),
                properties=args.get("properties", {}),
                vector=args.get("vector"),
                object_id=args.get("id"),
                **client_kwargs,
            )
        
        elif operation == "batch_insert":
            result = _batch_insert(
                class_name=args.get("class_name", ""),
                objects=args.get("objects", []),
                **client_kwargs,
            )
        
        elif operation == "query":
            result = _query(
                class_name=args.get("class_name", ""),
                query_vector=args.get("query_vector"),
                query_text=args.get("query_text"),
                properties=args.get("properties"),
                limit=args.get("limit", 10),
                filters=args.get("filters"),
                **client_kwargs,
            )
        
        elif operation == "get":
            result = _get_by_id(
                class_name=args.get("class_name", ""),
                object_id=args.get("id", ""),
                **client_kwargs,
            )
        
        elif operation == "update":
            result = _update(
                class_name=args.get("class_name", ""),
                object_id=args.get("id", ""),
                properties=args.get("properties", {}),
                vector=args.get("vector"),
                **client_kwargs,
            )
        
        elif operation == "delete":
            result = _delete(
                class_name=args.get("class_name", ""),
                object_id=args.get("id", ""),
                **client_kwargs,
            )
        
        else:
            return {
                "tool": "db_weaviate",
                "status": "error",
                "error": f"Unknown operation: {operation}",
            }
        
        return {"tool": "db_weaviate", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "db_weaviate", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "create_class": {
            "operation": "create_class",
            "class_name": "Article",
            "properties": [
                {"name": "title", "dataType": "text"},
                {"name": "content", "dataType": "text"},
                {"name": "category", "dataType": "text"},
            ],
            "vectorizer": "none",
        },
        "insert": {
            "operation": "insert",
            "class_name": "Article",
            "properties": {
                "title": "Introduction to Weaviate",
                "content": "Weaviate is a vector database...",
                "category": "tech",
            },
            "vector": [0.1] * 384,
        },
        "query": {
            "operation": "query",
            "class_name": "Article",
            "query_vector": [0.1] * 384,
            "limit": 5,
            "properties": ["title", "content"],
        },
    }
