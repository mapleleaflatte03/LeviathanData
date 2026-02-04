"""Tool: orchestration_langgraph
LangGraph graph-based agent orchestration.

Supported operations:
- create_graph: Define a state graph with nodes and edges
- run_graph: Execute the graph with input state
- compile: Compile graph with checkpointer
- stream: Stream graph execution
"""
from typing import Any, Dict, List, Optional, Callable, TypedDict
import json


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


langgraph = _optional_import("langgraph")

# Graph registry for storing created graphs
_graph_registry: Dict[str, Any] = {}


class GraphState(TypedDict, total=False):
    """Generic state for LangGraph workflows."""
    messages: List[Dict[str, Any]]
    data: Dict[str, Any]
    current_node: str
    history: List[str]
    result: Any


def _create_node_function(node_config: Dict[str, Any]) -> Callable:
    """Create a node function from configuration."""
    node_type = node_config.get("type", "passthrough")
    
    if node_type == "passthrough":
        def passthrough(state: GraphState) -> GraphState:
            state["history"] = state.get("history", []) + [node_config.get("name", "unknown")]
            return state
        return passthrough
    
    elif node_type == "transform":
        transform_fn = node_config.get("transform", "")
        def transform(state: GraphState) -> GraphState:
            state["history"] = state.get("history", []) + [node_config.get("name", "unknown")]
            # Apply simple transformations
            if transform_fn == "uppercase" and "data" in state:
                if "text" in state["data"]:
                    state["data"]["text"] = state["data"]["text"].upper()
            elif transform_fn == "lowercase" and "data" in state:
                if "text" in state["data"]:
                    state["data"]["text"] = state["data"]["text"].lower()
            return state
        return transform
    
    elif node_type == "conditional":
        def conditional(state: GraphState) -> str:
            condition = node_config.get("condition", {})
            field = condition.get("field", "")
            operator = condition.get("operator", "eq")
            value = condition.get("value")
            true_branch = condition.get("true_branch", "end")
            false_branch = condition.get("false_branch", "end")
            
            actual_value = state.get("data", {}).get(field)
            
            if operator == "eq" and actual_value == value:
                return true_branch
            elif operator == "neq" and actual_value != value:
                return true_branch
            elif operator == "gt" and actual_value is not None and actual_value > value:
                return true_branch
            elif operator == "lt" and actual_value is not None and actual_value < value:
                return true_branch
            elif operator == "contains" and value in str(actual_value):
                return true_branch
            else:
                return false_branch
        return conditional
    
    elif node_type == "aggregate":
        def aggregate(state: GraphState) -> GraphState:
            state["history"] = state.get("history", []) + [node_config.get("name", "unknown")]
            # Aggregate messages into result
            if "messages" in state:
                state["result"] = {
                    "message_count": len(state["messages"]),
                    "aggregated": True,
                }
            return state
        return aggregate
    
    # Default passthrough
    def default(state: GraphState) -> GraphState:
        state["history"] = state.get("history", []) + [node_config.get("name", "unknown")]
        return state
    return default


def _create_graph(
    graph_id: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    entry_point: str = "start",
) -> Dict[str, Any]:
    """Create a state graph from configuration."""
    if langgraph is None:
        # Simulate graph creation without langgraph
        graph_config = {
            "id": graph_id,
            "nodes": {n["name"]: n for n in nodes},
            "edges": edges,
            "entry_point": entry_point,
            "compiled": False,
        }
        _graph_registry[graph_id] = graph_config
        return {
            "graph_id": graph_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "entry_point": entry_point,
            "simulated": True,
        }
    
    from langgraph.graph import StateGraph, END
    
    # Create the graph
    graph = StateGraph(GraphState)
    
    # Add nodes
    for node_config in nodes:
        node_name = node_config["name"]
        if node_name != "end":
            node_fn = _create_node_function(node_config)
            graph.add_node(node_name, node_fn)
    
    # Add edges
    for edge in edges:
        from_node = edge["from"]
        to_node = edge["to"]
        condition = edge.get("condition")
        
        if to_node == "end":
            to_node = END
        
        if condition:
            # Conditional edge
            condition_fn = _create_node_function({"type": "conditional", "condition": condition})
            graph.add_conditional_edges(from_node, condition_fn)
        else:
            graph.add_edge(from_node, to_node)
    
    # Set entry point
    graph.set_entry_point(entry_point)
    
    # Compile and store
    compiled = graph.compile()
    _graph_registry[graph_id] = {
        "graph": graph,
        "compiled": compiled,
        "config": {"nodes": nodes, "edges": edges, "entry_point": entry_point},
    }
    
    return {
        "graph_id": graph_id,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "entry_point": entry_point,
    }


def _run_graph(
    graph_id: str,
    input_state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a graph with input state."""
    if graph_id not in _graph_registry:
        raise ValueError(f"Graph not found: {graph_id}")
    
    graph_data = _graph_registry[graph_id]
    
    # Check if simulated mode
    if isinstance(graph_data, dict) and graph_data.get("simulated") or "compiled" not in graph_data:
        # Simulate execution
        nodes = graph_data.get("nodes", {})
        edges = graph_data.get("edges", [])
        entry = graph_data.get("entry_point", "start")
        
        state = GraphState(
            messages=input_state.get("messages", []),
            data=input_state.get("data", {}),
            current_node=entry,
            history=[],
            result=None,
        )
        
        # Simple simulation: follow edges from entry to end
        visited = []
        current = entry
        max_steps = 100
        
        while current and current != "end" and len(visited) < max_steps:
            visited.append(current)
            state["history"].append(current)
            state["current_node"] = current
            
            # Find next node
            next_node = None
            for edge in edges:
                if edge["from"] == current:
                    next_node = edge["to"]
                    break
            current = next_node
        
        return {
            "final_state": dict(state),
            "steps": visited,
            "simulated": True,
        }
    
    # Real execution with langgraph
    compiled = graph_data["compiled"]
    
    initial_state = GraphState(
        messages=input_state.get("messages", []),
        data=input_state.get("data", {}),
        current_node="",
        history=[],
        result=None,
    )
    
    result = compiled.invoke(initial_state, config=config)
    
    return {
        "final_state": dict(result),
        "steps": result.get("history", []),
    }


def _stream_graph(
    graph_id: str,
    input_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Stream graph execution (returns all intermediate states)."""
    if graph_id not in _graph_registry:
        raise ValueError(f"Graph not found: {graph_id}")
    
    graph_data = _graph_registry[graph_id]
    
    # Simulated streaming
    if isinstance(graph_data, dict) and (graph_data.get("simulated") or "compiled" not in graph_data):
        result = _run_graph(graph_id, input_state)
        return {
            "stream": [{"step": i, "node": node} for i, node in enumerate(result["steps"])],
            "final_state": result["final_state"],
            "simulated": True,
        }
    
    # Real streaming
    compiled = graph_data["compiled"]
    
    initial_state = GraphState(
        messages=input_state.get("messages", []),
        data=input_state.get("data", {}),
        current_node="",
        history=[],
        result=None,
    )
    
    stream_results = []
    for i, state in enumerate(compiled.stream(initial_state)):
        stream_results.append({"step": i, "state": dict(state)})
    
    return {
        "stream": stream_results,
        "final_state": stream_results[-1]["state"] if stream_results else {},
    }


def _list_graphs() -> Dict[str, Any]:
    """List all registered graphs."""
    return {
        "graphs": [
            {
                "id": gid,
                "node_count": len(g.get("config", {}).get("nodes", g.get("nodes", {}))),
                "simulated": g.get("simulated", False),
            }
            for gid, g in _graph_registry.items()
        ]
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run LangGraph operations.
    
    Args:
        args: Dictionary with:
            - operation: "create_graph", "run_graph", "stream", "list"
            - graph_id: Identifier for the graph
            - nodes: List of node configurations
            - edges: List of edge configurations
            - input_state: Initial state for execution
    
    Returns:
        Result dictionary with operation output
    """
    args = args or {}
    operation = args.get("operation", "list")
    
    try:
        if operation == "create_graph":
            result = _create_graph(
                graph_id=args.get("graph_id", "default"),
                nodes=args.get("nodes", []),
                edges=args.get("edges", []),
                entry_point=args.get("entry_point", "start"),
            )
        
        elif operation == "run_graph":
            result = _run_graph(
                graph_id=args.get("graph_id", "default"),
                input_state=args.get("input_state", {}),
                config=args.get("config"),
            )
        
        elif operation == "stream":
            result = _stream_graph(
                graph_id=args.get("graph_id", "default"),
                input_state=args.get("input_state", {}),
            )
        
        elif operation == "list":
            result = _list_graphs()
        
        else:
            return {"tool": "orchestration_langgraph", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "orchestration_langgraph", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "orchestration_langgraph", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "create_simple_graph": {
            "operation": "create_graph",
            "graph_id": "data_pipeline",
            "nodes": [
                {"name": "ingest", "type": "passthrough"},
                {"name": "transform", "type": "transform", "transform": "uppercase"},
                {"name": "validate", "type": "passthrough"},
                {"name": "output", "type": "aggregate"},
            ],
            "edges": [
                {"from": "ingest", "to": "transform"},
                {"from": "transform", "to": "validate"},
                {"from": "validate", "to": "output"},
                {"from": "output", "to": "end"},
            ],
            "entry_point": "ingest",
        },
        "run_graph": {
            "operation": "run_graph",
            "graph_id": "data_pipeline",
            "input_state": {
                "data": {"text": "hello world"},
                "messages": [{"role": "user", "content": "process this"}],
            },
        },
    }
