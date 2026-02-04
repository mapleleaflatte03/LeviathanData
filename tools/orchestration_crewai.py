"""Tool: orchestration_crewai
CrewAI multi-agent collaboration framework.

Supported operations:
- create_crew: Create a crew with agents and tasks
- run_crew: Execute crew tasks
- create_agent: Define an agent with role and tools
- create_task: Define a task with description and agent
- list_crews: List all registered crews
"""
from typing import Any, Dict, List, Optional
import json
import uuid


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


crewai = _optional_import("crewai")

# Registry for crews, agents, tasks
_crew_registry: Dict[str, Any] = {}
_agent_registry: Dict[str, Any] = {}
_task_registry: Dict[str, Any] = {}


class SimulatedAgent:
    """Simulated agent when crewai is not installed."""
    
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: Optional[List[str]] = None,
        verbose: bool = False,
        allow_delegation: bool = False,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.verbose = verbose
        self.allow_delegation = allow_delegation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "tools": self.tools,
            "allow_delegation": self.allow_delegation,
        }


class SimulatedTask:
    """Simulated task when crewai is not installed."""
    
    def __init__(
        self,
        description: str,
        agent: SimulatedAgent,
        expected_output: str = "",
        context: Optional[List["SimulatedTask"]] = None,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.context = context or []
        self.output = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "agent_role": self.agent.role,
            "expected_output": self.expected_output,
        }


class SimulatedCrew:
    """Simulated crew when crewai is not installed."""
    
    def __init__(
        self,
        agents: List[SimulatedAgent],
        tasks: List[SimulatedTask],
        process: str = "sequential",
        verbose: bool = False,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.verbose = verbose
    
    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Simulate crew execution."""
        results = []
        context_data = inputs or {}
        
        for task in self.tasks:
            # Simulate task execution
            result = {
                "task_id": task.id,
                "agent": task.agent.role,
                "description": task.description,
                "output": f"[Simulated output for: {task.description[:50]}...]",
                "status": "completed",
            }
            results.append(result)
            task.output = result["output"]
        
        return {
            "crew_id": self.id,
            "process": self.process,
            "task_results": results,
            "final_output": results[-1]["output"] if results else None,
            "simulated": True,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_count": len(self.agents),
            "task_count": len(self.tasks),
            "process": self.process,
        }


def _create_agent(
    agent_id: str,
    role: str,
    goal: str,
    backstory: str,
    tools: Optional[List[str]] = None,
    verbose: bool = False,
    allow_delegation: bool = False,
    llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create an agent."""
    if crewai is None:
        # Create simulated agent
        agent = SimulatedAgent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools,
            verbose=verbose,
            allow_delegation=allow_delegation,
        )
        _agent_registry[agent_id] = agent
        return {"agent_id": agent_id, **agent.to_dict(), "simulated": True}
    
    from crewai import Agent
    
    kwargs = {
        "role": role,
        "goal": goal,
        "backstory": backstory,
        "verbose": verbose,
        "allow_delegation": allow_delegation,
    }
    
    if llm_config:
        # Configure LLM if provided
        pass
    
    agent = Agent(**kwargs)
    _agent_registry[agent_id] = agent
    
    return {
        "agent_id": agent_id,
        "role": role,
        "goal": goal,
    }


def _create_task(
    task_id: str,
    description: str,
    agent_id: str,
    expected_output: str = "",
    context_task_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a task."""
    if agent_id not in _agent_registry:
        raise ValueError(f"Agent not found: {agent_id}")
    
    agent = _agent_registry[agent_id]
    
    context_tasks = []
    if context_task_ids:
        for tid in context_task_ids:
            if tid in _task_registry:
                context_tasks.append(_task_registry[tid])
    
    if crewai is None:
        task = SimulatedTask(
            description=description,
            agent=agent,
            expected_output=expected_output,
            context=context_tasks,
        )
        _task_registry[task_id] = task
        return {"task_id": task_id, **task.to_dict(), "simulated": True}
    
    from crewai import Task
    
    task = Task(
        description=description,
        agent=agent,
        expected_output=expected_output,
        context=context_tasks if context_tasks else None,
    )
    _task_registry[task_id] = task
    
    return {
        "task_id": task_id,
        "description": description,
        "agent_id": agent_id,
    }


def _create_crew(
    crew_id: str,
    agent_ids: List[str],
    task_ids: List[str],
    process: str = "sequential",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Create a crew from agents and tasks."""
    agents = [_agent_registry[aid] for aid in agent_ids if aid in _agent_registry]
    tasks = [_task_registry[tid] for tid in task_ids if tid in _task_registry]
    
    if not agents:
        raise ValueError("No valid agents found")
    if not tasks:
        raise ValueError("No valid tasks found")
    
    if crewai is None:
        crew = SimulatedCrew(
            agents=agents,
            tasks=tasks,
            process=process,
            verbose=verbose,
        )
        _crew_registry[crew_id] = crew
        return {"crew_id": crew_id, **crew.to_dict(), "simulated": True}
    
    from crewai import Crew, Process
    
    process_type = Process.sequential if process == "sequential" else Process.hierarchical
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=process_type,
        verbose=verbose,
    )
    _crew_registry[crew_id] = crew
    
    return {
        "crew_id": crew_id,
        "agent_count": len(agents),
        "task_count": len(tasks),
        "process": process,
    }


def _run_crew(
    crew_id: str,
    inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a crew."""
    if crew_id not in _crew_registry:
        raise ValueError(f"Crew not found: {crew_id}")
    
    crew = _crew_registry[crew_id]
    result = crew.kickoff(inputs=inputs)
    
    if isinstance(result, dict):
        return result
    
    return {
        "crew_id": crew_id,
        "result": str(result),
    }


def _list_crews() -> Dict[str, Any]:
    """List all registered crews."""
    crews = []
    for cid, crew in _crew_registry.items():
        if hasattr(crew, "to_dict"):
            crews.append({"id": cid, **crew.to_dict()})
        else:
            crews.append({
                "id": cid,
                "agent_count": len(crew.agents) if hasattr(crew, "agents") else 0,
                "task_count": len(crew.tasks) if hasattr(crew, "tasks") else 0,
            })
    
    return {
        "crews": crews,
        "agent_count": len(_agent_registry),
        "task_count": len(_task_registry),
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run CrewAI operations.
    
    Args:
        args: Dictionary with:
            - operation: "create_agent", "create_task", "create_crew", "run_crew", "list"
            - Various operation-specific parameters
    
    Returns:
        Result dictionary with operation output
    """
    args = args or {}
    operation = args.get("operation", "list")
    
    try:
        if operation == "create_agent":
            result = _create_agent(
                agent_id=args.get("agent_id", str(uuid.uuid4())[:8]),
                role=args.get("role", "Assistant"),
                goal=args.get("goal", ""),
                backstory=args.get("backstory", ""),
                tools=args.get("tools"),
                verbose=args.get("verbose", False),
                allow_delegation=args.get("allow_delegation", False),
                llm_config=args.get("llm_config"),
            )
        
        elif operation == "create_task":
            result = _create_task(
                task_id=args.get("task_id", str(uuid.uuid4())[:8]),
                description=args.get("description", ""),
                agent_id=args.get("agent_id", ""),
                expected_output=args.get("expected_output", ""),
                context_task_ids=args.get("context_task_ids"),
            )
        
        elif operation == "create_crew":
            result = _create_crew(
                crew_id=args.get("crew_id", str(uuid.uuid4())[:8]),
                agent_ids=args.get("agent_ids", []),
                task_ids=args.get("task_ids", []),
                process=args.get("process", "sequential"),
                verbose=args.get("verbose", False),
            )
        
        elif operation == "run_crew":
            result = _run_crew(
                crew_id=args.get("crew_id", ""),
                inputs=args.get("inputs"),
            )
        
        elif operation == "list":
            result = _list_crews()
        
        else:
            return {"tool": "orchestration_crewai", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "orchestration_crewai", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "orchestration_crewai", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "create_research_crew": {
            "steps": [
                {
                    "operation": "create_agent",
                    "agent_id": "researcher",
                    "role": "Data Researcher",
                    "goal": "Find and analyze relevant data sources",
                    "backstory": "Expert data analyst with 10 years experience",
                },
                {
                    "operation": "create_agent",
                    "agent_id": "writer",
                    "role": "Report Writer",
                    "goal": "Create comprehensive reports from research",
                    "backstory": "Technical writer specialized in data reports",
                },
                {
                    "operation": "create_task",
                    "task_id": "research_task",
                    "description": "Research the latest trends in AI",
                    "agent_id": "researcher",
                    "expected_output": "Summary of AI trends with sources",
                },
                {
                    "operation": "create_task",
                    "task_id": "report_task",
                    "description": "Write a report based on the research",
                    "agent_id": "writer",
                    "expected_output": "Formatted report document",
                    "context_task_ids": ["research_task"],
                },
                {
                    "operation": "create_crew",
                    "crew_id": "research_crew",
                    "agent_ids": ["researcher", "writer"],
                    "task_ids": ["research_task", "report_task"],
                    "process": "sequential",
                },
            ]
        },
        "run_crew": {
            "operation": "run_crew",
            "crew_id": "research_crew",
            "inputs": {"topic": "artificial intelligence"},
        },
    }
