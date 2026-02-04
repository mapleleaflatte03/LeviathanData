"""Tool: orchestration_openclaw
OpenClaw-like tool orchestration for Leviathan.

This module provides a lightweight tool orchestration system
similar to OpenClaw for managing and executing tools.

Supported operations:
- register_tool: Register a tool
- list_tools: List registered tools
- execute: Execute a tool
- create_workflow: Create tool workflow
- run_workflow: Execute workflow
- get_history: Get execution history
"""
from typing import Any, Callable, Dict, List, Optional
import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


@dataclass
class ToolDefinition:
    """Tool definition."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Optional[Callable] = None
    module: Optional[str] = None


@dataclass
class ExecutionResult:
    """Tool execution result."""
    tool: str
    success: bool
    result: Any
    error: Optional[str] = None
    duration_ms: float = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# Global registries
_tools: Dict[str, ToolDefinition] = {}
_workflows: Dict[str, List[Dict[str, Any]]] = {}
_history: List[ExecutionResult] = []


def _register_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    handler: Optional[Callable] = None,
    module: Optional[str] = None,
) -> Dict[str, Any]:
    """Register a tool."""
    tool = ToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
        handler=handler,
        module=module,
    )
    _tools[name] = tool
    
    return {
        "registered": True,
        "name": name,
        "description": description,
    }


def _list_tools() -> Dict[str, Any]:
    """List all registered tools."""
    tools = []
    for name, tool in _tools.items():
        tools.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "has_handler": tool.handler is not None,
            "module": tool.module,
        })
    
    return {"tools": tools, "count": len(tools)}


def _execute_tool(
    name: str,
    args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a registered tool."""
    if name not in _tools:
        raise ValueError(f"Tool '{name}' not registered")
    
    tool = _tools[name]
    args = args or {}
    
    start_time = time.time()
    
    try:
        if tool.handler:
            result = tool.handler(args)
        elif tool.module:
            # Dynamic import and execution
            module = __import__(tool.module, fromlist=["run"])
            result = module.run(args)
        else:
            raise ValueError(f"Tool '{name}' has no handler or module")
        
        duration_ms = (time.time() - start_time) * 1000
        
        exec_result = ExecutionResult(
            tool=name,
            success=True,
            result=result,
            duration_ms=duration_ms,
        )
        _history.append(exec_result)
        
        return {
            "success": True,
            "result": result,
            "duration_ms": duration_ms,
        }
    
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        exec_result = ExecutionResult(
            tool=name,
            success=False,
            result=None,
            error=str(e),
            duration_ms=duration_ms,
        )
        _history.append(exec_result)
        
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "duration_ms": duration_ms,
        }


def _create_workflow(
    name: str,
    steps: List[Dict[str, Any]],
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a workflow."""
    workflow = []
    
    for step in steps:
        tool_name = step.get("tool")
        if tool_name and tool_name not in _tools:
            raise ValueError(f"Tool '{tool_name}' not registered")
        
        workflow.append({
            "tool": tool_name,
            "args": step.get("args", {}),
            "condition": step.get("condition"),
            "on_error": step.get("on_error", "stop"),
        })
    
    _workflows[name] = workflow
    
    return {
        "created": True,
        "name": name,
        "step_count": len(workflow),
    }


def _run_workflow(
    name: str,
    initial_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a workflow."""
    if name not in _workflows:
        raise ValueError(f"Workflow '{name}' not found")
    
    workflow = _workflows[name]
    context = initial_context or {}
    results = []
    
    for i, step in enumerate(workflow):
        tool_name = step["tool"]
        args = step["args"].copy()
        
        # Replace context variables in args
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                var_name = value[1:]
                if var_name in context:
                    args[key] = context[var_name]
        
        # Check condition
        condition = step.get("condition")
        if condition:
            if not eval(condition, {"context": context}):
                results.append({
                    "step": i,
                    "tool": tool_name,
                    "skipped": True,
                    "reason": "condition not met",
                })
                continue
        
        # Execute tool
        result = _execute_tool(tool_name, args)
        results.append({
            "step": i,
            "tool": tool_name,
            **result,
        })
        
        # Update context with result
        if result.get("success"):
            context[f"step_{i}_result"] = result.get("result")
            context["last_result"] = result.get("result")
        
        # Handle errors
        if not result.get("success"):
            on_error = step.get("on_error", "stop")
            if on_error == "stop":
                break
    
    return {
        "workflow": name,
        "completed": True,
        "step_count": len(results),
        "results": results,
        "context": context,
    }


def _get_history(
    limit: int = 10,
    tool_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Get execution history."""
    history = _history
    
    if tool_name:
        history = [h for h in history if h.tool == tool_name]
    
    history = history[-limit:]
    
    return {
        "history": [
            {
                "tool": h.tool,
                "success": h.success,
                "duration_ms": h.duration_ms,
                "timestamp": h.timestamp,
                "error": h.error,
            }
            for h in history
        ],
        "count": len(history),
    }


def _auto_register_tools() -> Dict[str, Any]:
    """Auto-register tools from the tools directory."""
    import os
    import glob
    
    tools_dir = os.path.dirname(__file__)
    registered = []
    
    for filepath in glob.glob(os.path.join(tools_dir, "*.py")):
        filename = os.path.basename(filepath)
        if filename.startswith("__"):
            continue
        
        module_name = filename[:-3]
        tool_name = module_name
        
        if tool_name in _tools:
            continue
        
        try:
            module = __import__(f"tools.{module_name}", fromlist=["run", "example"])
            
            if hasattr(module, "run"):
                # Extract description from docstring
                description = module.__doc__ or f"Tool: {tool_name}"
                description = description.strip().split("\n")[0]
                
                # Get example for parameters
                parameters = {}
                if hasattr(module, "example"):
                    examples = module.example()
                    if isinstance(examples, dict):
                        for op, params in examples.items():
                            parameters[op] = params
                
                _register_tool(
                    name=tool_name,
                    description=description,
                    parameters=parameters,
                    module=f"tools.{module_name}",
                )
                registered.append(tool_name)
        
        except Exception as e:
            pass
    
    return {"registered": registered, "count": len(registered)}


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run OpenClaw operations."""
    args = args or {}
    operation = args.get("operation", "list_tools")
    
    try:
        if operation == "register_tool":
            result = _register_tool(
                name=args.get("name", ""),
                description=args.get("description", ""),
                parameters=args.get("parameters", {}),
                module=args.get("module"),
            )
        
        elif operation == "list_tools":
            result = _list_tools()
        
        elif operation == "execute":
            result = _execute_tool(
                name=args.get("tool_name", ""),
                args=args.get("tool_args", {}),
            )
        
        elif operation == "create_workflow":
            result = _create_workflow(
                name=args.get("name", ""),
                steps=args.get("steps", []),
                description=args.get("description"),
            )
        
        elif operation == "run_workflow":
            result = _run_workflow(
                name=args.get("name", ""),
                initial_context=args.get("context", {}),
            )
        
        elif operation == "get_history":
            result = _get_history(
                limit=args.get("limit", 10),
                tool_name=args.get("tool_name"),
            )
        
        elif operation == "auto_register":
            result = _auto_register_tools()
        
        else:
            return {"tool": "orchestration_openclaw", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "orchestration_openclaw", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "orchestration_openclaw", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "register_tool": {
            "operation": "register_tool",
            "name": "my_tool",
            "description": "My custom tool",
            "parameters": {"input": {"type": "string"}},
            "module": "tools.my_tool",
        },
        "execute": {
            "operation": "execute",
            "tool_name": "ml_pandas",
            "tool_args": {
                "operation": "read_csv",
                "path": "data.csv",
            },
        },
        "create_workflow": {
            "operation": "create_workflow",
            "name": "etl_pipeline",
            "steps": [
                {"tool": "ml_pandas", "args": {"operation": "read_csv", "path": "input.csv"}},
                {"tool": "ml_pandas", "args": {"operation": "filter", "condition": "value > 0"}},
                {"tool": "ml_pandas", "args": {"operation": "to_csv", "path": "output.csv"}},
            ],
        },
        "run_workflow": {
            "operation": "run_workflow",
            "name": "etl_pipeline",
            "context": {"input_path": "data.csv"},
        },
    }
