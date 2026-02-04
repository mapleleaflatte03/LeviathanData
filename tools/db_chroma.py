"""Tool: db_chroma
Chroma vector database for embeddings and semantic search.

Supported operations:
- create_collection: Create a new collection
- add: Add documents with embeddings
- query: Semantic similarity search
- get: Get documents by ID
- update: Update documents
- delete: Delete documents
- list_collections: List all collections
"""
from typing import Any, Dict, List, Optional
import json
import uuid


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


chromadb = _optional_import("chromadb")

# Global client cache
_client_cache = {}


def _get_client(persist_directory: Optional[str] = None) -> Any:
    """Get or create Chroma client."""
    if chromadb is None:
        raise ImportError("chromadb is not installed. Run: pip install chromadb")
    
    cache_key = persist_directory or "ephemeral"
    
    if cache_key not in _client_cache:
        if persist_directory:
            _client_cache[cache_key] = chromadb.PersistentClient(path=persist_directory)
        else:
            _client_cache[cache_key] = chromadb.Client()
    
    return _client_cache[cache_key]


def _create_collection(
    name: str,
    persist_directory: Optional[str] = None,
    metadata: Optional[Dict] = None,
    embedding_function: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new collection."""
    client = _get_client(persist_directory)
    
    kwargs = {}
    if metadata:
        kwargs["metadata"] = metadata
    
    # Use default embedding function or specify one
    if embedding_function == "sentence_transformer":
        from chromadb.utils import embedding_functions
        kwargs["embedding_function"] = embedding_functions.SentenceTransformerEmbeddingFunction()
    elif embedding_function == "openai":
        from chromadb.utils import embedding_functions
        import os
        kwargs["embedding_function"] = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
    
    collection = client.get_or_create_collection(name=name, **kwargs)
    
    return {
        "name": collection.name,
        "count": collection.count(),
        "metadata": collection.metadata,
    }


def _add_documents(
    collection_name: str,
    documents: List[str],
    metadatas: Optional[List[Dict]] = None,
    ids: Optional[List[str]] = None,
    embeddings: Optional[List[List[float]]] = None,
    persist_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """Add documents to collection."""
    client = _get_client(persist_directory)
    collection = client.get_collection(name=collection_name)
    
    # Generate IDs if not provided
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in documents]
    
    kwargs = {
        "documents": documents,
        "ids": ids,
    }
    
    if metadatas:
        kwargs["metadatas"] = metadatas
    if embeddings:
        kwargs["embeddings"] = embeddings
    
    collection.add(**kwargs)
    
    return {
        "added": len(documents),
        "ids": ids,
        "collection_count": collection.count(),
    }


def _query(
    collection_name: str,
    query_texts: Optional[List[str]] = None,
    query_embeddings: Optional[List[List[float]]] = None,
    n_results: int = 10,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    include: Optional[List[str]] = None,
    persist_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """Query collection for similar documents."""
    client = _get_client(persist_directory)
    collection = client.get_collection(name=collection_name)
    
    kwargs = {"n_results": n_results}
    
    if query_texts:
        kwargs["query_texts"] = query_texts
    if query_embeddings:
        kwargs["query_embeddings"] = query_embeddings
    if where:
        kwargs["where"] = where
    if where_document:
        kwargs["where_document"] = where_document
    if include:
        kwargs["include"] = include
    else:
        kwargs["include"] = ["documents", "metadatas", "distances"]
    
    results = collection.query(**kwargs)
    
    return {
        "ids": results.get("ids", []),
        "documents": results.get("documents", []),
        "metadatas": results.get("metadatas", []),
        "distances": results.get("distances", []),
        "embeddings": results.get("embeddings"),
    }


def _get_documents(
    collection_name: str,
    ids: Optional[List[str]] = None,
    where: Optional[Dict] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    include: Optional[List[str]] = None,
    persist_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """Get documents from collection."""
    client = _get_client(persist_directory)
    collection = client.get_collection(name=collection_name)
    
    kwargs = {}
    if ids:
        kwargs["ids"] = ids
    if where:
        kwargs["where"] = where
    if limit:
        kwargs["limit"] = limit
    if offset:
        kwargs["offset"] = offset
    if include:
        kwargs["include"] = include
    else:
        kwargs["include"] = ["documents", "metadatas"]
    
    results = collection.get(**kwargs)
    
    return {
        "ids": results.get("ids", []),
        "documents": results.get("documents", []),
        "metadatas": results.get("metadatas", []),
        "embeddings": results.get("embeddings"),
    }


def _update_documents(
    collection_name: str,
    ids: List[str],
    documents: Optional[List[str]] = None,
    metadatas: Optional[List[Dict]] = None,
    embeddings: Optional[List[List[float]]] = None,
    persist_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """Update documents in collection."""
    client = _get_client(persist_directory)
    collection = client.get_collection(name=collection_name)
    
    kwargs = {"ids": ids}
    if documents:
        kwargs["documents"] = documents
    if metadatas:
        kwargs["metadatas"] = metadatas
    if embeddings:
        kwargs["embeddings"] = embeddings
    
    collection.update(**kwargs)
    
    return {"updated": len(ids)}


def _delete_documents(
    collection_name: str,
    ids: Optional[List[str]] = None,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    persist_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """Delete documents from collection."""
    client = _get_client(persist_directory)
    collection = client.get_collection(name=collection_name)
    
    kwargs = {}
    if ids:
        kwargs["ids"] = ids
    if where:
        kwargs["where"] = where
    if where_document:
        kwargs["where_document"] = where_document
    
    collection.delete(**kwargs)
    
    return {"deleted": True, "collection_count": collection.count()}


def _list_collections(persist_directory: Optional[str] = None) -> Dict[str, Any]:
    """List all collections."""
    client = _get_client(persist_directory)
    collections = client.list_collections()
    
    return {
        "collections": [
            {"name": c.name, "count": c.count(), "metadata": c.metadata}
            for c in collections
        ]
    }


def _delete_collection(
    name: str,
    persist_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """Delete a collection."""
    client = _get_client(persist_directory)
    client.delete_collection(name=name)
    return {"deleted": name}


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Chroma vector database operations.
    
    Args:
        args: Dictionary with:
            - operation: "create_collection", "add", "query", "get", "update", "delete", "list", "delete_collection"
            - collection_name: Name of collection
            - persist_directory: Optional path for persistent storage
            - documents: List of document strings
            - ids: Document IDs
            - metadatas: Document metadata
            - query_texts: Text queries for similarity search
            - n_results: Number of results to return
    
    Returns:
        Result dictionary with operation output
    """
    args = args or {}
    operation = args.get("operation", "list")
    persist_directory = args.get("persist_directory")
    collection_name = args.get("collection_name", "default")
    
    if chromadb is None:
        return {"tool": "db_chroma", "status": "error", "error": "chromadb not installed"}
    
    try:
        if operation == "create_collection":
            result = _create_collection(
                name=collection_name,
                persist_directory=persist_directory,
                metadata=args.get("metadata"),
                embedding_function=args.get("embedding_function"),
            )
        
        elif operation == "add":
            result = _add_documents(
                collection_name=collection_name,
                documents=args.get("documents", []),
                metadatas=args.get("metadatas"),
                ids=args.get("ids"),
                embeddings=args.get("embeddings"),
                persist_directory=persist_directory,
            )
        
        elif operation == "query":
            result = _query(
                collection_name=collection_name,
                query_texts=args.get("query_texts"),
                query_embeddings=args.get("query_embeddings"),
                n_results=args.get("n_results", 10),
                where=args.get("where"),
                where_document=args.get("where_document"),
                include=args.get("include"),
                persist_directory=persist_directory,
            )
        
        elif operation == "get":
            result = _get_documents(
                collection_name=collection_name,
                ids=args.get("ids"),
                where=args.get("where"),
                limit=args.get("limit"),
                offset=args.get("offset"),
                include=args.get("include"),
                persist_directory=persist_directory,
            )
        
        elif operation == "update":
            result = _update_documents(
                collection_name=collection_name,
                ids=args.get("ids", []),
                documents=args.get("documents"),
                metadatas=args.get("metadatas"),
                embeddings=args.get("embeddings"),
                persist_directory=persist_directory,
            )
        
        elif operation == "delete":
            result = _delete_documents(
                collection_name=collection_name,
                ids=args.get("ids"),
                where=args.get("where"),
                where_document=args.get("where_document"),
                persist_directory=persist_directory,
            )
        
        elif operation == "list":
            result = _list_collections(persist_directory=persist_directory)
        
        elif operation == "delete_collection":
            result = _delete_collection(
                name=collection_name,
                persist_directory=persist_directory,
            )
        
        else:
            return {"tool": "db_chroma", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "db_chroma", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "db_chroma", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "create_collection": {
            "operation": "create_collection",
            "collection_name": "documents",
            "persist_directory": "./chroma_data",
            "metadata": {"description": "Document embeddings"},
        },
        "add_documents": {
            "operation": "add",
            "collection_name": "documents",
            "documents": [
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with many layers.",
                "Natural language processing deals with text data.",
            ],
            "metadatas": [
                {"source": "wiki", "topic": "ml"},
                {"source": "wiki", "topic": "dl"},
                {"source": "wiki", "topic": "nlp"},
            ],
        },
        "query": {
            "operation": "query",
            "collection_name": "documents",
            "query_texts": ["What is AI?"],
            "n_results": 3,
        },
        "filter_query": {
            "operation": "query",
            "collection_name": "documents",
            "query_texts": ["neural networks"],
            "where": {"topic": "dl"},
            "n_results": 5,
        },
    }
