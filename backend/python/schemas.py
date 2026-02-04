from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


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
