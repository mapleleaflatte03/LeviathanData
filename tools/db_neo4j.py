"""Tool: db_neo4j
Neo4j graph database for knowledge graphs.

Supported operations:
- query: Execute Cypher query
- create_node: Create a node
- create_relationship: Create a relationship between nodes
- find_nodes: Find nodes by label and properties
- find_path: Find shortest path between nodes
- delete_node: Delete a node
- update_node: Update node properties
- get_schema: Get database schema info
"""
from typing import Any, Dict, List, Optional, Union
import json


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


neo4j = _optional_import("neo4j")

# Driver cache
_driver_cache: Dict[str, Any] = {}


def _get_driver(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
    database: str = "neo4j",
) -> Any:
    """Get or create Neo4j driver."""
    if neo4j is None:
        raise ImportError("neo4j is not installed. Run: pip install neo4j")
    
    from neo4j import GraphDatabase
    
    cache_key = f"{uri}:{user}:{database}"
    
    if cache_key not in _driver_cache:
        _driver_cache[cache_key] = {
            "driver": GraphDatabase.driver(uri, auth=(user, password)),
            "database": database,
        }
    
    return _driver_cache[cache_key]


def _serialize_value(value: Any) -> Any:
    """Serialize Neo4j values for JSON."""
    if hasattr(value, "_properties"):  # Node or Relationship
        return dict(value._properties)
    elif hasattr(value, "nodes"):  # Path
        return {
            "nodes": [dict(n._properties) for n in value.nodes],
            "relationships": [dict(r._properties) for r in value.relationships],
        }
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def _query(
    cypher: str,
    parameters: Optional[Dict[str, Any]] = None,
    **connection_kwargs,
) -> Dict[str, Any]:
    """Execute a Cypher query."""
    conn = _get_driver(**connection_kwargs)
    driver = conn["driver"]
    database = conn["database"]
    
    with driver.session(database=database) as session:
        result = session.run(cypher, parameters or {})
        records = []
        
        for record in result:
            row = {}
            for key in record.keys():
                row[key] = _serialize_value(record[key])
            records.append(row)
        
        summary = result.consume()
        
        return {
            "records": records,
            "count": len(records),
            "counters": {
                "nodes_created": summary.counters.nodes_created,
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_created": summary.counters.relationships_created,
                "relationships_deleted": summary.counters.relationships_deleted,
                "properties_set": summary.counters.properties_set,
            },
        }


def _create_node(
    label: str,
    properties: Dict[str, Any],
    **connection_kwargs,
) -> Dict[str, Any]:
    """Create a node with given label and properties."""
    props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
    cypher = f"CREATE (n:{label} {{{props_str}}}) RETURN n, id(n) as node_id"
    
    result = _query(cypher, properties, **connection_kwargs)
    
    if result["records"]:
        return {
            "node": result["records"][0].get("n"),
            "node_id": result["records"][0].get("node_id"),
            "label": label,
        }
    return {"error": "Failed to create node"}


def _create_relationship(
    from_label: str,
    from_match: Dict[str, Any],
    to_label: str,
    to_match: Dict[str, Any],
    rel_type: str,
    rel_properties: Optional[Dict[str, Any]] = None,
    **connection_kwargs,
) -> Dict[str, Any]:
    """Create a relationship between two nodes."""
    from_match_str = ", ".join([f"a.{k} = $from_{k}" for k in from_match.keys()])
    to_match_str = ", ".join([f"b.{k} = $to_{k}" for k in to_match.keys()])
    
    params = {f"from_{k}": v for k, v in from_match.items()}
    params.update({f"to_{k}": v for k, v in to_match.items()})
    
    if rel_properties:
        props_str = ", ".join([f"{k}: $rel_{k}" for k in rel_properties.keys()])
        params.update({f"rel_{k}": v for k, v in rel_properties.items()})
        cypher = f"""
        MATCH (a:{from_label}), (b:{to_label})
        WHERE {from_match_str} AND {to_match_str}
        CREATE (a)-[r:{rel_type} {{{props_str}}}]->(b)
        RETURN r, id(r) as rel_id
        """
    else:
        cypher = f"""
        MATCH (a:{from_label}), (b:{to_label})
        WHERE {from_match_str} AND {to_match_str}
        CREATE (a)-[r:{rel_type}]->(b)
        RETURN r, id(r) as rel_id
        """
    
    result = _query(cypher, params, **connection_kwargs)
    
    if result["records"]:
        return {
            "relationship": result["records"][0].get("r"),
            "rel_id": result["records"][0].get("rel_id"),
            "type": rel_type,
        }
    return {"error": "Failed to create relationship"}


def _find_nodes(
    label: str,
    match_properties: Optional[Dict[str, Any]] = None,
    limit: int = 100,
    **connection_kwargs,
) -> Dict[str, Any]:
    """Find nodes by label and optional properties."""
    if match_properties:
        where_str = " AND ".join([f"n.{k} = ${k}" for k in match_properties.keys()])
        cypher = f"MATCH (n:{label}) WHERE {where_str} RETURN n, id(n) as node_id LIMIT {limit}"
        params = match_properties
    else:
        cypher = f"MATCH (n:{label}) RETURN n, id(n) as node_id LIMIT {limit}"
        params = {}
    
    result = _query(cypher, params, **connection_kwargs)
    
    return {
        "nodes": [
            {"properties": r.get("n"), "id": r.get("node_id")}
            for r in result["records"]
        ],
        "count": len(result["records"]),
    }


def _find_path(
    from_label: str,
    from_match: Dict[str, Any],
    to_label: str,
    to_match: Dict[str, Any],
    max_depth: int = 5,
    rel_type: Optional[str] = None,
    **connection_kwargs,
) -> Dict[str, Any]:
    """Find shortest path between two nodes."""
    from_match_str = ", ".join([f"a.{k} = $from_{k}" for k in from_match.keys()])
    to_match_str = ", ".join([f"b.{k} = $to_{k}" for k in to_match.keys()])
    
    params = {f"from_{k}": v for k, v in from_match.items()}
    params.update({f"to_{k}": v for k, v in to_match.items()})
    
    rel_pattern = f":{rel_type}" if rel_type else ""
    
    cypher = f"""
    MATCH (a:{from_label}), (b:{to_label})
    WHERE {from_match_str} AND {to_match_str}
    MATCH p = shortestPath((a)-[{rel_pattern}*..{max_depth}]-(b))
    RETURN p, length(p) as path_length
    """
    
    result = _query(cypher, params, **connection_kwargs)
    
    if result["records"]:
        return {
            "path": result["records"][0].get("p"),
            "length": result["records"][0].get("path_length"),
        }
    return {"path": None, "message": "No path found"}


def _delete_node(
    label: str,
    match_properties: Dict[str, Any],
    detach: bool = True,
    **connection_kwargs,
) -> Dict[str, Any]:
    """Delete a node."""
    where_str = " AND ".join([f"n.{k} = ${k}" for k in match_properties.keys()])
    detach_str = "DETACH " if detach else ""
    cypher = f"MATCH (n:{label}) WHERE {where_str} {detach_str}DELETE n"
    
    result = _query(cypher, match_properties, **connection_kwargs)
    
    return {
        "deleted": result["counters"]["nodes_deleted"],
    }


def _update_node(
    label: str,
    match_properties: Dict[str, Any],
    set_properties: Dict[str, Any],
    **connection_kwargs,
) -> Dict[str, Any]:
    """Update node properties."""
    where_str = " AND ".join([f"n.{k} = $match_{k}" for k in match_properties.keys()])
    set_str = ", ".join([f"n.{k} = $set_{k}" for k in set_properties.keys()])
    
    params = {f"match_{k}": v for k, v in match_properties.items()}
    params.update({f"set_{k}": v for k, v in set_properties.items()})
    
    cypher = f"MATCH (n:{label}) WHERE {where_str} SET {set_str} RETURN n"
    
    result = _query(cypher, params, **connection_kwargs)
    
    return {
        "updated": result["counters"]["properties_set"],
        "nodes": [r.get("n") for r in result["records"]],
    }


def _get_schema(**connection_kwargs) -> Dict[str, Any]:
    """Get database schema information."""
    # Get labels
    labels_result = _query("CALL db.labels()", **connection_kwargs)
    labels = [r.get("label") for r in labels_result["records"]]
    
    # Get relationship types
    rel_result = _query("CALL db.relationshipTypes()", **connection_kwargs)
    rel_types = [r.get("relationshipType") for r in rel_result["records"]]
    
    # Get property keys
    props_result = _query("CALL db.propertyKeys()", **connection_kwargs)
    property_keys = [r.get("propertyKey") for r in props_result["records"]]
    
    return {
        "labels": labels,
        "relationship_types": rel_types,
        "property_keys": property_keys,
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Neo4j operations.
    
    Args:
        args: Dictionary with:
            - operation: Neo4j operation to perform
            - uri, user, password, database: Connection parameters
            - Various operation-specific parameters
    
    Returns:
        Result dictionary with operation output
    """
    args = args or {}
    operation = args.get("operation", "query")
    
    if neo4j is None:
        return {
            "tool": "db_neo4j",
            "status": "error",
            "error": "neo4j not installed. Run: pip install neo4j",
        }
    
    connection_kwargs = {}
    for key in ["uri", "user", "password", "database"]:
        if key in args:
            connection_kwargs[key] = args[key]
    
    try:
        if operation == "query":
            result = _query(
                cypher=args.get("cypher", ""),
                parameters=args.get("parameters"),
                **connection_kwargs,
            )
        
        elif operation == "create_node":
            result = _create_node(
                label=args.get("label", "Node"),
                properties=args.get("properties", {}),
                **connection_kwargs,
            )
        
        elif operation == "create_relationship":
            result = _create_relationship(
                from_label=args.get("from_label", "Node"),
                from_match=args.get("from_match", {}),
                to_label=args.get("to_label", "Node"),
                to_match=args.get("to_match", {}),
                rel_type=args.get("rel_type", "RELATES_TO"),
                rel_properties=args.get("rel_properties"),
                **connection_kwargs,
            )
        
        elif operation == "find_nodes":
            result = _find_nodes(
                label=args.get("label", "Node"),
                match_properties=args.get("match"),
                limit=args.get("limit", 100),
                **connection_kwargs,
            )
        
        elif operation == "find_path":
            result = _find_path(
                from_label=args.get("from_label", "Node"),
                from_match=args.get("from_match", {}),
                to_label=args.get("to_label", "Node"),
                to_match=args.get("to_match", {}),
                max_depth=args.get("max_depth", 5),
                rel_type=args.get("rel_type"),
                **connection_kwargs,
            )
        
        elif operation == "delete_node":
            result = _delete_node(
                label=args.get("label", "Node"),
                match_properties=args.get("match", {}),
                detach=args.get("detach", True),
                **connection_kwargs,
            )
        
        elif operation == "update_node":
            result = _update_node(
                label=args.get("label", "Node"),
                match_properties=args.get("match", {}),
                set_properties=args.get("set", {}),
                **connection_kwargs,
            )
        
        elif operation == "schema":
            result = _get_schema(**connection_kwargs)
        
        else:
            return {
                "tool": "db_neo4j",
                "status": "error",
                "error": f"Unknown operation: {operation}",
            }
        
        return {"tool": "db_neo4j", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "db_neo4j", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "cypher_query": {
            "operation": "query",
            "cypher": "MATCH (n:Person) WHERE n.name = $name RETURN n",
            "parameters": {"name": "Alice"},
        },
        "create_person": {
            "operation": "create_node",
            "label": "Person",
            "properties": {"name": "Alice", "age": 30, "city": "NYC"},
        },
        "create_friendship": {
            "operation": "create_relationship",
            "from_label": "Person",
            "from_match": {"name": "Alice"},
            "to_label": "Person",
            "to_match": {"name": "Bob"},
            "rel_type": "KNOWS",
            "rel_properties": {"since": 2020},
        },
        "find_path": {
            "operation": "find_path",
            "from_label": "Person",
            "from_match": {"name": "Alice"},
            "to_label": "Person",
            "to_match": {"name": "Charlie"},
            "max_depth": 4,
        },
    }
