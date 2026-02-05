from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class PipelineRequest(BaseModel):
    runId: str
    fileId: Optional[str] = None
    filePath: Optional[str] = None


class PipelineResponse(BaseModel):
    message: str
    alert: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None


class ToolRequest(BaseModel):
    toolName: str
    args: Dict[str, Any] = Field(default_factory=dict)


class LLMChatRequest(BaseModel):
    messages: list
    stream: bool = False


class LLMChatResponse(BaseModel):
    text: str


class HuntRequest(BaseModel):
    """Request for autonomous data hunting."""
    prompt: str = Field(..., description="Natural language description of data needed")
    messages: Optional[List[Dict[str, str]]] = Field(default=None, description="Chat context")


class HuntResponse(BaseModel):
    """Response from autonomous data hunt."""
    items_collected: int = Field(default=0, description="Number of items collected")
    sources: List[str] = Field(default_factory=list, description="Sources used")
    data: List[Dict[str, Any]] = Field(default_factory=list, description="Collected data")
    summary: str = Field(default="", description="LLM-generated summary")
    chart_data: Optional[Dict[str, Any]] = Field(default=None, description="Chart visualization data")
    timestamp: str = Field(default="", description="Hunt completion timestamp")
