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


class OpenClawAnalyzeRequest(BaseModel):
    """Request for OpenClaw full-stack company OSINT analysis."""
    company_name: str = Field(..., description="Target company name for OSINT analysis")
    prompt: Optional[str] = Field(default=None, description="Optional additional analysis prompt")
    language: str = Field(default="vi", description="Output language: vi or en")
    export_pdf: bool = Field(default=True, description="Export PDF report")
    export_html: bool = Field(default=True, description="Export HTML report")


class OpenClawAnalyzeResponse(BaseModel):
    """Response from OpenClaw company analysis - real data, NOT UI screenshots."""
    success: bool = Field(default=True, description="Analysis success status")
    company: str = Field(default="", description="Analyzed company name")
    
    # OSINT data
    osint_tools_used: List[str] = Field(default_factory=list, description="OSINT tools executed")
    osint_items_collected: int = Field(default=0, description="Total OSINT items collected")
    osint_data: Dict[str, Any] = Field(default_factory=dict, description="Raw OSINT data by source")
    
    # KPIs calculated from real data
    kpis: Dict[str, Any] = Field(default_factory=dict, description="Calculated KPIs: coverage, risk, transparency")
    
    # LLM analysis
    analysis: str = Field(default="", description="VN-tuned LLM analysis with insights")
    
    # Dashboard data for real chart binding
    dashboard_data: Dict[str, Any] = Field(default_factory=dict, description="Chart data for PowerBI-like dashboard")
    
    # Report paths - REAL PDF/HTML, not UI screenshots
    report_pdf_path: Optional[str] = Field(default=None, description="Path to generated PDF report")
    report_html_path: Optional[str] = Field(default=None, description="Path to generated HTML report")
    
    # Execution logs
    execution_log: List[str] = Field(default_factory=list, description="Step-by-step execution log")
    timestamp: str = Field(default="", description="Analysis completion timestamp")
