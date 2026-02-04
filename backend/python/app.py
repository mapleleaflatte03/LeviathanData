from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import logging

from .config import CONFIG
from .schemas import PipelineRequest, PipelineResponse, ToolRequest, LLMChatRequest, LLMChatResponse
from .orchestrator import ingest, analyze, visualize, reflect
from .tool_registry import run_tool, list_tools
from .llm_client import chat_completion
from .storage import ensure_dirs
from .background import start_background_jobs

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
