"""Tool: orchestration_semantickernel
Microsoft Semantic Kernel for AI orchestration.

Supported operations:
- create_kernel: Create Semantic Kernel instance
- add_plugin: Add plugin to kernel
- invoke_function: Invoke kernel function
- create_plan: Create execution plan
- run_plan: Execute plan
- add_memory: Add to semantic memory
- search_memory: Search semantic memory
"""
from typing import Any, Dict, List, Optional
import json
import asyncio


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


# Storage for kernels and functions
_kernels: Dict[str, Any] = {}
_plugins: Dict[str, Dict[str, Any]] = {}


def _create_kernel(
    name: str = "default",
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    service_id: str = "default",
) -> Dict[str, Any]:
    """Create a Semantic Kernel instance."""
    sk = _optional_import("semantic_kernel")
    
    if sk is None:
        raise ImportError("semantic_kernel not installed. Run: pip install semantic-kernel")
    
    kernel = sk.Kernel()
    
    # Add chat completion service
    if api_key:
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
        
        service = OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id=model,
            api_key=api_key,
            endpoint=base_url,
        )
        kernel.add_service(service)
    
    _kernels[name] = kernel
    
    return {
        "created": True,
        "name": name,
        "model": model,
    }


def _add_plugin(
    kernel_name: str,
    plugin_name: str,
    functions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Add a plugin with functions to the kernel."""
    sk = _optional_import("semantic_kernel")
    
    if sk is None:
        raise ImportError("semantic_kernel not installed")
    
    kernel = _kernels.get(kernel_name)
    if not kernel:
        raise ValueError(f"Kernel '{kernel_name}' not found")
    
    # Create plugin functions
    plugin_funcs = {}
    
    for func_def in functions:
        func_name = func_def.get("name", "function")
        func_type = func_def.get("type", "semantic")  # semantic or native
        
        if func_type == "semantic":
            # Create semantic function from prompt template
            prompt = func_def.get("prompt", "{{$input}}")
            description = func_def.get("description", "")
            
            from semantic_kernel.functions import KernelFunction
            from semantic_kernel.prompt_template import PromptTemplateConfig
            
            config = PromptTemplateConfig(
                template=prompt,
                description=description,
            )
            
            func = kernel.create_function_from_prompt(
                function_name=func_name,
                plugin_name=plugin_name,
                prompt_template_config=config,
            )
            plugin_funcs[func_name] = func
        
        elif func_type == "native":
            # Create native function from code
            code = func_def.get("code", "def func(input): return input")
            exec_globals = {}
            exec(code, exec_globals)
            native_func = exec_globals.get(func_name)
            
            if native_func:
                from semantic_kernel.functions import kernel_function
                decorated = kernel_function(name=func_name)(native_func)
                plugin_funcs[func_name] = decorated
    
    _plugins[plugin_name] = plugin_funcs
    
    return {
        "added": True,
        "plugin_name": plugin_name,
        "function_count": len(plugin_funcs),
    }


def _invoke_function(
    kernel_name: str,
    plugin_name: str,
    function_name: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Invoke a kernel function."""
    sk = _optional_import("semantic_kernel")
    
    if sk is None:
        raise ImportError("semantic_kernel not installed")
    
    kernel = _kernels.get(kernel_name)
    if not kernel:
        raise ValueError(f"Kernel '{kernel_name}' not found")
    
    async def _invoke():
        result = await kernel.invoke(
            plugin_name=plugin_name,
            function_name=function_name,
            arguments=arguments or {},
        )
        return str(result)
    
    result = asyncio.run(_invoke())
    
    return {
        "invoked": True,
        "plugin": plugin_name,
        "function": function_name,
        "result": result,
    }


def _create_plan(
    kernel_name: str,
    goal: str,
    planner_type: str = "sequential",
) -> Dict[str, Any]:
    """Create an execution plan."""
    sk = _optional_import("semantic_kernel")
    
    if sk is None:
        raise ImportError("semantic_kernel not installed")
    
    kernel = _kernels.get(kernel_name)
    if not kernel:
        raise ValueError(f"Kernel '{kernel_name}' not found")
    
    async def _plan():
        if planner_type == "sequential":
            from semantic_kernel.planners import SequentialPlanner
            planner = SequentialPlanner(kernel)
        else:
            from semantic_kernel.planners import ActionPlanner
            planner = ActionPlanner(kernel)
        
        plan = await planner.create_plan(goal)
        return plan
    
    plan = asyncio.run(_plan())
    
    # Store plan for later execution
    plan_id = f"plan_{len(_kernels)}"
    _kernels[plan_id] = plan
    
    return {
        "created": True,
        "plan_id": plan_id,
        "goal": goal,
        "steps": len(plan._steps) if hasattr(plan, "_steps") else 0,
    }


def _run_plan(
    kernel_name: str,
    plan_id: str,
) -> Dict[str, Any]:
    """Execute a plan."""
    sk = _optional_import("semantic_kernel")
    
    if sk is None:
        raise ImportError("semantic_kernel not installed")
    
    kernel = _kernels.get(kernel_name)
    plan = _kernels.get(plan_id)
    
    if not kernel or not plan:
        raise ValueError("Kernel or plan not found")
    
    async def _execute():
        result = await plan.invoke(kernel)
        return str(result)
    
    result = asyncio.run(_execute())
    
    return {
        "executed": True,
        "plan_id": plan_id,
        "result": result,
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Semantic Kernel operations."""
    args = args or {}
    operation = args.get("operation", "create_kernel")
    
    try:
        if operation == "create_kernel":
            result = _create_kernel(
                name=args.get("name", "default"),
                model=args.get("model", "gpt-4"),
                api_key=args.get("api_key"),
                base_url=args.get("base_url"),
            )
        
        elif operation == "add_plugin":
            result = _add_plugin(
                kernel_name=args.get("kernel_name", "default"),
                plugin_name=args.get("plugin_name", "plugin"),
                functions=args.get("functions", []),
            )
        
        elif operation == "invoke_function":
            result = _invoke_function(
                kernel_name=args.get("kernel_name", "default"),
                plugin_name=args.get("plugin_name", ""),
                function_name=args.get("function_name", ""),
                arguments=args.get("arguments"),
            )
        
        elif operation == "create_plan":
            result = _create_plan(
                kernel_name=args.get("kernel_name", "default"),
                goal=args.get("goal", ""),
                planner_type=args.get("planner_type", "sequential"),
            )
        
        elif operation == "run_plan":
            result = _run_plan(
                kernel_name=args.get("kernel_name", "default"),
                plan_id=args.get("plan_id", ""),
            )
        
        elif operation == "list_kernels":
            result = {"kernels": list(_kernels.keys())}
        
        else:
            return {"tool": "orchestration_semantickernel", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "orchestration_semantickernel", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "orchestration_semantickernel", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "create_kernel": {
            "operation": "create_kernel",
            "name": "my_kernel",
            "model": "gpt-4",
            "api_key": "sk-...",
        },
        "add_plugin": {
            "operation": "add_plugin",
            "kernel_name": "my_kernel",
            "plugin_name": "writer",
            "functions": [
                {
                    "name": "summarize",
                    "type": "semantic",
                    "prompt": "Summarize the following text:\n{{$input}}\n\nSummary:",
                    "description": "Summarizes text",
                },
            ],
        },
        "invoke_function": {
            "operation": "invoke_function",
            "kernel_name": "my_kernel",
            "plugin_name": "writer",
            "function_name": "summarize",
            "arguments": {"input": "Long text to summarize..."},
        },
    }
