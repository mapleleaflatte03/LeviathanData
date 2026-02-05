from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import asyncio
import logging

from .config import CONFIG
from .schemas import PipelineRequest, PipelineResponse, ToolRequest, LLMChatRequest, LLMChatResponse, HuntRequest, HuntResponse
from .orchestrator import ingest, analyze, visualize, reflect
from .tool_registry import run_tool, list_tools
from .llm_client import chat_completion
from .storage import ensure_dirs
from .background import start_background_jobs
from .crawler import autonomous_hunt, get_background_hunter

logger = logging.getLogger("leviathan-python")
logging.basicConfig(level=CONFIG["log_level"].upper())

app = FastAPI(title="Leviathan Python Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

REQUEST_COUNT = Counter("leviathan_requests_total", "Total requests", ["endpoint"])


@app.on_event("startup")
def _startup():
    ensure_dirs()
    start_background_jobs(logger)


@app.get("/health")
def health():
    import datetime

    return {"ok": True, "ts": datetime.datetime.utcnow().isoformat() + "Z"}


@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/pipeline/ingest", response_model=PipelineResponse)
def pipeline_ingest(req: PipelineRequest):
    REQUEST_COUNT.labels("ingest").inc()
    return ingest(req.runId, req.filePath)


@app.post("/pipeline/analyze", response_model=PipelineResponse)
def pipeline_analyze(req: PipelineRequest):
    REQUEST_COUNT.labels("analyze").inc()
    return analyze(req.runId, req.filePath)


@app.post("/pipeline/visualize", response_model=PipelineResponse)
def pipeline_visualize(req: PipelineRequest):
    REQUEST_COUNT.labels("visualize").inc()
    return visualize(req.runId, req.filePath)


@app.post("/pipeline/reflect", response_model=PipelineResponse)
def pipeline_reflect(req: PipelineRequest):
    REQUEST_COUNT.labels("reflect").inc()
    return reflect(req.runId, req.filePath)


@app.post("/pipeline/run", response_model=PipelineResponse)
def pipeline_run(req: PipelineRequest):
    REQUEST_COUNT.labels("run").inc()
    _ = ingest(req.runId, req.filePath)
    _ = analyze(req.runId, req.filePath)
    _ = visualize(req.runId, req.filePath)
    return reflect(req.runId, req.filePath)


@app.post("/tools/run")
def tools_run(req: ToolRequest):
    REQUEST_COUNT.labels("tools_run").inc()
    try:
        result = run_tool(req.toolName, req.args)
        return {"ok": True, "result": result}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/tools")
def tools_list():
    return {"tools": list_tools()}


@app.post("/llm/chat", response_model=LLMChatResponse)
def llm_chat(req: LLMChatRequest):
    REQUEST_COUNT.labels("llm_chat").inc()
    try:
        text = chat_completion(req.messages)
        return LLMChatResponse(text=text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/hunt", response_model=HuntResponse)
async def hunt(req: HuntRequest):
    """Autonomous ethical data hunt endpoint."""
    REQUEST_COUNT.labels("hunt").inc()
    try:
        logger.info(f"Hunt request: {req.prompt[:100]}...")
        
        # Run autonomous hunt
        result = await autonomous_hunt(
            prompt=req.prompt,
            on_progress=lambda s, m: logger.info(f"[{s}] {m}"),
        )
        
        # Generate summary using LLM
        summary = ""
        if result["items_collected"] > 0:
            try:
                summary_prompt = f"""Summarize the hunt results:
- Collected {result['items_collected']} items
- Sources: {', '.join(result['sources'])}
- Sample data: {str(result['data'][:3])[:500]}

Provide a brief, helpful summary of what was found."""
                
                summary = chat_completion([{"role": "user", "content": summary_prompt}])
            except Exception as e:
                logger.warning(f"Summary generation failed: {e}")
                summary = f"Collected {result['items_collected']} items from {len(result['sources'])} sources."
        else:
            summary = "No data found. Try more specific queries like 'VN stock trends' or 'Kaggle Titanic dataset'."
        
        # Prepare chart data if we have numeric data
        chart_data = None
        if result["data"] and len(result["data"]) > 0:
            first_item = result["data"][0]
            if "close" in first_item or "value" in first_item:
                # Time-series data
                chart_data = {
                    "type": "line",
                    "labels": [d.get("date", str(i)) for i, d in enumerate(result["data"][:50])],
                    "values": [d.get("close") or d.get("value") or 0 for d in result["data"][:50]],
                    "title": f"Hunt Results: {req.prompt[:30]}..."
                }
        
        return HuntResponse(
            items_collected=result["items_collected"],
            sources=result["sources"],
            data=result["data"][:100],  # Limit returned data
            summary=summary,
            chart_data=chart_data,
            timestamp=result["timestamp"]
        )
        
    except Exception as exc:
        logger.error(f"Hunt failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/hunt/status")
def hunt_status():
    """Get background hunter status."""
    hunter = get_background_hunter()
    return {
        "running": hunter.running,
        "last_hunt": hunter.last_hunt.isoformat() if hunter.last_hunt else None,
        "hunt_count": len(hunter.hunt_results),
        "targets": hunter.hunt_targets,
    }
