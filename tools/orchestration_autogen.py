"""Tool: orchestration_autogen
Microsoft AutoGen multi-agent conversations.

Supported operations:
- create_agent: Create an agent
- create_group_chat: Create group chat
- run_conversation: Run agent conversation
- register_function: Register function for agent
- list_agents: List created agents
"""
from typing import Any, Callable, Dict, List, Optional
import json


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


autogen = _optional_import("autogen")


# Store for agents and configs
_agents: Dict[str, Any] = {}
_configs: Dict[str, Any] = {}
_conversations: List[Dict[str, Any]] = []


def _get_llm_config(
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Build LLM config."""
    config = {
        "model": model,
        "temperature": temperature,
    }
    
    if api_key:
        config["api_key"] = api_key
    
    if base_url:
        config["base_url"] = base_url
    
    return {"config_list": [config]}


def _create_agent(
    name: str,
    agent_type: str = "assistant",
    system_message: Optional[str] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    human_input_mode: str = "NEVER",
    code_execution_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create an AutoGen agent."""
    if autogen is None:
        raise ImportError("autogen not installed. Run: pip install pyautogen")
    
    if agent_type == "assistant":
        agent = autogen.AssistantAgent(
            name=name,
            system_message=system_message or "You are a helpful AI assistant.",
            llm_config=llm_config,
        )
    elif agent_type == "user_proxy":
        agent = autogen.UserProxyAgent(
            name=name,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config or {"use_docker": False},
        )
    elif agent_type == "conversable":
        agent = autogen.ConversableAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode=human_input_mode,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    _agents[name] = agent
    
    return {
        "created": True,
        "name": name,
        "type": agent_type,
    }


def _create_group_chat(
    name: str,
    agent_names: List[str],
    max_round: int = 10,
    speaker_selection_method: str = "auto",
) -> Dict[str, Any]:
    """Create a group chat."""
    if autogen is None:
        raise ImportError("autogen not installed")
    
    agents = [_agents[n] for n in agent_names if n in _agents]
    
    if not agents:
        raise ValueError("No valid agents found")
    
    group_chat = autogen.GroupChat(
        agents=agents,
        messages=[],
        max_round=max_round,
        speaker_selection_method=speaker_selection_method,
    )
    
    _agents[name] = group_chat
    
    return {
        "created": True,
        "name": name,
        "agent_count": len(agents),
    }


def _run_conversation(
    initiator_name: str,
    recipient_name: str,
    message: str,
    max_turns: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a conversation between agents."""
    if autogen is None:
        raise ImportError("autogen not installed")
    
    initiator = _agents.get(initiator_name)
    recipient = _agents.get(recipient_name)
    
    if not initiator or not recipient:
        raise ValueError("Initiator or recipient not found")
    
    # Run the conversation
    chat_result = initiator.initiate_chat(
        recipient,
        message=message,
        max_turns=max_turns,
    )
    
    # Extract conversation history
    messages = []
    if hasattr(chat_result, "chat_history"):
        messages = chat_result.chat_history
    
    conversation = {
        "initiator": initiator_name,
        "recipient": recipient_name,
        "message": message,
        "messages": messages,
    }
    _conversations.append(conversation)
    
    return {
        "completed": True,
        "message_count": len(messages),
        "messages": messages[-5:] if messages else [],  # Last 5 messages
    }


def _register_function(
    agent_name: str,
    function_name: str,
    function_code: str,
    description: str,
) -> Dict[str, Any]:
    """Register a function for an agent."""
    if autogen is None:
        raise ImportError("autogen not installed")
    
    agent = _agents.get(agent_name)
    if not agent:
        raise ValueError(f"Agent '{agent_name}' not found")
    
    # Create function from code
    exec_globals = {}
    exec(function_code, exec_globals)
    func = exec_globals.get(function_name)
    
    if not func:
        raise ValueError(f"Function '{function_name}' not found in code")
    
    # Register with agent
    if hasattr(agent, "register_function"):
        agent.register_function(
            function_map={function_name: func},
        )
    
    return {
        "registered": True,
        "agent": agent_name,
        "function": function_name,
    }


def _list_agents() -> Dict[str, Any]:
    """List all created agents."""
    agents = []
    for name, agent in _agents.items():
        agent_info = {
            "name": name,
            "type": type(agent).__name__,
        }
        if hasattr(agent, "system_message"):
            agent_info["system_message"] = agent.system_message[:100] + "..." if len(agent.system_message) > 100 else agent.system_message
        agents.append(agent_info)
    
    return {"agents": agents}


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run AutoGen operations."""
    args = args or {}
    operation = args.get("operation", "list_agents")
    
    try:
        if operation == "create_agent":
            llm_config = None
            if args.get("model") or args.get("api_key"):
                llm_config = _get_llm_config(
                    model=args.get("model", "gpt-4"),
                    api_key=args.get("api_key"),
                    base_url=args.get("base_url"),
                    temperature=args.get("temperature", 0.7),
                )
            
            result = _create_agent(
                name=args.get("name", "assistant"),
                agent_type=args.get("agent_type", "assistant"),
                system_message=args.get("system_message"),
                llm_config=llm_config,
                human_input_mode=args.get("human_input_mode", "NEVER"),
                code_execution_config=args.get("code_execution_config"),
            )
        
        elif operation == "create_group_chat":
            result = _create_group_chat(
                name=args.get("name", "group_chat"),
                agent_names=args.get("agent_names", []),
                max_round=args.get("max_round", 10),
                speaker_selection_method=args.get("speaker_selection_method", "auto"),
            )
        
        elif operation == "run_conversation":
            result = _run_conversation(
                initiator_name=args.get("initiator", ""),
                recipient_name=args.get("recipient", ""),
                message=args.get("message", "Hello"),
                max_turns=args.get("max_turns"),
            )
        
        elif operation == "register_function":
            result = _register_function(
                agent_name=args.get("agent_name", ""),
                function_name=args.get("function_name", ""),
                function_code=args.get("function_code", ""),
                description=args.get("description", ""),
            )
        
        elif operation == "list_agents":
            result = _list_agents()
        
        else:
            return {"tool": "orchestration_autogen", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "orchestration_autogen", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "orchestration_autogen", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "create_assistant": {
            "operation": "create_agent",
            "name": "coding_assistant",
            "agent_type": "assistant",
            "system_message": "You are a helpful coding assistant.",
            "model": "gpt-4",
            "api_key": "sk-...",
        },
        "create_user_proxy": {
            "operation": "create_agent",
            "name": "user",
            "agent_type": "user_proxy",
            "human_input_mode": "NEVER",
        },
        "run_conversation": {
            "operation": "run_conversation",
            "initiator": "user",
            "recipient": "coding_assistant",
            "message": "Write a Python function to calculate fibonacci numbers.",
            "max_turns": 5,
        },
    }
