"""Tool: db_qdrant
Qdrant vector database for similarity search.

Supported operations:
- create_collection: Create a new collection
- delete_collection: Delete a collection
- list_collections: List all collections
- upsert: Insert or update points
- search: Similarity search
- scroll: Iterate through points
- delete_points: Delete points by ID or filter
- get_points: Get points by ID
"""
from typing import Any, Dict, List, Optional, Union
import json
import uuid


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


qdrant_client = _optional_import("qdrant_client")

# Client cache
_client_cache: Dict[str, Any] = {}


def _get_client(
    host: str = "localhost",
    port: int = 6333,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    path: Optional[str] = None,
) -> Any:
    """Get or create Qdrant client."""
    if qdrant_client is None:
        raise ImportError("qdrant-client is not installed. Run: pip install qdrant-client")
    
    from qdrant_client import QdrantClient
    
    if path:
        cache_key = f"local:{path}"
    elif url:
        cache_key = f"url:{url}"
    else:
        cache_key = f"{host}:{port}"
    
    if cache_key not in _client_cache:
        if path:
            # Local persistent storage
            _client_cache[cache_key] = QdrantClient(path=path)
        elif url:
            _client_cache[cache_key] = QdrantClient(url=url, api_key=api_key)
        else:
            _client_cache[cache_key] = QdrantClient(host=host, port=port, api_key=api_key)
    
    return _client_cache[cache_key]


def _create_collection(
    collection_name: str,
    vector_size: int,
    distance: str = "Cosine",
    on_disk: bool = False,
    **client_kwargs,
) -> Dict[str, Any]:
    """Create a new collection."""
    from qdrant_client.models import Distance, VectorParams
    
    client = _get_client(**client_kwargs)
    
    distance_map = {
        "cosine": Distance.COSINE,
        "euclid": Distance.EUCLID,
        "dot": Distance.DOT,
    }
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance_map.get(distance.lower(), Distance.COSINE),
            on_disk=on_disk,
        ),
    )
    
    return {
        "created": collection_name,
        "vector_size": vector_size,
        "distance": distance,
    }


def _delete_collection(collection_name: str, **client_kwargs) -> Dict[str, Any]:
    """Delete a collection."""
    client = _get_client(**client_kwargs)
    client.delete_collection(collection_name=collection_name)
    return {"deleted": collection_name}


def _list_collections(**client_kwargs) -> Dict[str, Any]:
    """List all collections."""
    client = _get_client(**client_kwargs)
    collections = client.get_collections()
    
    return {
        "collections": [
            {
                "name": c.name,
            }
            for c in collections.collections
        ]
    }


def _get_collection_info(collection_name: str, **client_kwargs) -> Dict[str, Any]:
    """Get collection info."""
    client = _get_client(**client_kwargs)
    info = client.get_collection(collection_name=collection_name)
    
    return {
        "name": collection_name,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": info.status.value if hasattr(info.status, "value") else str(info.status),
        "config": {
            "size": info.config.params.vectors.size if hasattr(info.config.params.vectors, "size") else None,
            "distance": str(info.config.params.vectors.distance) if hasattr(info.config.params.vectors, "distance") else None,
        },
    }


def _upsert(
    collection_name: str,
    points: List[Dict[str, Any]],
    **client_kwargs,
) -> Dict[str, Any]:
    """Insert or update points."""
    from qdrant_client.models import PointStruct
    
    client = _get_client(**client_kwargs)
    
    point_structs = []
    for p in points:
        point_id = p.get("id") or str(uuid.uuid4())
        if isinstance(point_id, str) and not point_id.isdigit():
            # Use UUID for string IDs
            try:
                point_id = uuid.UUID(point_id).hex
            except ValueError:
                point_id = uuid.uuid5(uuid.NAMESPACE_DNS, point_id).hex
        
        point_structs.append(PointStruct(
            id=point_id,
            vector=p["vector"],
            payload=p.get("payload", {}),
        ))
    
    result = client.upsert(
        collection_name=collection_name,
        points=point_structs,
    )
    
    return {
        "upserted": len(point_structs),
        "status": str(result.status) if hasattr(result, "status") else "ok",
    }


def _search(
    collection_name: str,
    query_vector: List[float],
    limit: int = 10,
    score_threshold: Optional[float] = None,
    filter_conditions: Optional[Dict] = None,
    with_payload: bool = True,
    with_vectors: bool = False,
    **client_kwargs,
) -> Dict[str, Any]:
    """Similarity search."""
    client = _get_client(**client_kwargs)
    
    search_kwargs = {
        "collection_name": collection_name,
        "query_vector": query_vector,
        "limit": limit,
        "with_payload": with_payload,
        "with_vectors": with_vectors,
    }
    
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold
    
    if filter_conditions:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        conditions = []
        for field, value in filter_conditions.items():
            conditions.append(FieldCondition(
                key=field,
                match=MatchValue(value=value),
            ))
        search_kwargs["query_filter"] = Filter(must=conditions)
    
    results = client.search(**search_kwargs)
    
    return {
        "results": [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload,
                "vector": r.vector if with_vectors else None,
            }
            for r in results
        ],
        "count": len(results),
    }


def _scroll(
    collection_name: str,
    limit: int = 10,
    offset: Optional[str] = None,
    with_payload: bool = True,
    with_vectors: bool = False,
    filter_conditions: Optional[Dict] = None,
    **client_kwargs,
) -> Dict[str, Any]:
    """Scroll through points."""
    client = _get_client(**client_kwargs)
    
    scroll_kwargs = {
        "collection_name": collection_name,
        "limit": limit,
        "with_payload": with_payload,
        "with_vectors": with_vectors,
    }
    
    if offset:
        scroll_kwargs["offset"] = offset
    
    points, next_offset = client.scroll(**scroll_kwargs)
    
    return {
        "points": [
            {
                "id": str(p.id),
                "payload": p.payload,
                "vector": p.vector if with_vectors else None,
            }
            for p in points
        ],
        "next_offset": str(next_offset) if next_offset else None,
    }


def _delete_points(
    collection_name: str,
    point_ids: Optional[List[str]] = None,
    filter_conditions: Optional[Dict] = None,
    **client_kwargs,
) -> Dict[str, Any]:
    """Delete points by ID or filter."""
    client = _get_client(**client_kwargs)
    
    if point_ids:
        from qdrant_client.models import PointIdsList
        client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=point_ids),
        )
        return {"deleted_ids": point_ids}
    
    elif filter_conditions:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
        
        conditions = []
        for field, value in filter_conditions.items():
            conditions.append(FieldCondition(
                key=field,
                match=MatchValue(value=value),
            ))
        
        client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=Filter(must=conditions)),
        )
        return {"deleted_by_filter": True}
    
    return {"deleted": False, "error": "No point_ids or filter provided"}


def _get_points(
    collection_name: str,
    point_ids: List[str],
    with_payload: bool = True,
    with_vectors: bool = False,
    **client_kwargs,
) -> Dict[str, Any]:
    """Get points by ID."""
    client = _get_client(**client_kwargs)
    
    points = client.retrieve(
        collection_name=collection_name,
        ids=point_ids,
        with_payload=with_payload,
        with_vectors=with_vectors,
    )
    
    return {
        "points": [
            {
                "id": str(p.id),
                "payload": p.payload,
                "vector": p.vector if with_vectors else None,
            }
            for p in points
        ],
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Qdrant operations.
    
    Args:
        args: Dictionary with:
            - operation: Qdrant operation to perform
            - collection_name: Name of collection
            - host, port, url, api_key: Connection parameters
            - Various operation-specific parameters
    
    Returns:
        Result dictionary with operation output
    """
    args = args or {}
    operation = args.get("operation", "list_collections")
    
    if qdrant_client is None:
        return {
            "tool": "db_qdrant",
            "status": "error",
            "error": "qdrant-client not installed. Run: pip install qdrant-client",
        }
    
    # Extract client kwargs
    client_kwargs = {}
    for key in ["host", "port", "url", "api_key", "path"]:
        if key in args:
            client_kwargs[key] = args[key]
    
    try:
        if operation == "create_collection":
            result = _create_collection(
                collection_name=args.get("collection_name", ""),
                vector_size=args.get("vector_size", 384),
                distance=args.get("distance", "Cosine"),
                on_disk=args.get("on_disk", False),
                **client_kwargs,
            )
        
        elif operation == "delete_collection":
            result = _delete_collection(
                collection_name=args.get("collection_name", ""),
                **client_kwargs,
            )
        
        elif operation == "list_collections":
            result = _list_collections(**client_kwargs)
        
        elif operation == "info":
            result = _get_collection_info(
                collection_name=args.get("collection_name", ""),
                **client_kwargs,
            )
        
        elif operation == "upsert":
            result = _upsert(
                collection_name=args.get("collection_name", ""),
                points=args.get("points", []),
                **client_kwargs,
            )
        
        elif operation == "search":
            result = _search(
                collection_name=args.get("collection_name", ""),
                query_vector=args.get("query_vector", []),
                limit=args.get("limit", 10),
                score_threshold=args.get("score_threshold"),
                filter_conditions=args.get("filter"),
                with_payload=args.get("with_payload", True),
                with_vectors=args.get("with_vectors", False),
                **client_kwargs,
            )
        
        elif operation == "scroll":
            result = _scroll(
                collection_name=args.get("collection_name", ""),
                limit=args.get("limit", 10),
                offset=args.get("offset"),
                with_payload=args.get("with_payload", True),
                with_vectors=args.get("with_vectors", False),
                **client_kwargs,
            )
        
        elif operation == "delete":
            result = _delete_points(
                collection_name=args.get("collection_name", ""),
                point_ids=args.get("point_ids"),
                filter_conditions=args.get("filter"),
                **client_kwargs,
            )
        
        elif operation == "get":
            result = _get_points(
                collection_name=args.get("collection_name", ""),
                point_ids=args.get("point_ids", []),
                with_payload=args.get("with_payload", True),
                with_vectors=args.get("with_vectors", False),
                **client_kwargs,
            )
        
        else:
            return {
                "tool": "db_qdrant",
                "status": "error",
                "error": f"Unknown operation: {operation}",
            }
        
        return {"tool": "db_qdrant", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "db_qdrant", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "create_collection": {
            "operation": "create_collection",
            "collection_name": "documents",
            "vector_size": 384,
            "distance": "Cosine",
        },
        "upsert": {
            "operation": "upsert",
            "collection_name": "documents",
            "points": [
                {
                    "id": "doc1",
                    "vector": [0.1] * 384,
                    "payload": {"title": "Document 1", "category": "tech"},
                },
                {
                    "id": "doc2",
                    "vector": [0.2] * 384,
                    "payload": {"title": "Document 2", "category": "science"},
                },
            ],
        },
        "search": {
            "operation": "search",
            "collection_name": "documents",
            "query_vector": [0.15] * 384,
            "limit": 5,
            "filter": {"category": "tech"},
        },
    }
