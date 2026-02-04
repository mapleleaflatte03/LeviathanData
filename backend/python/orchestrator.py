from pathlib import Path
from typing import Dict, Any
from .schemas import PipelineResponse
from .llm_client import chat_completion


def ingest(run_id: str, file_path: str | None) -> PipelineResponse:
    message = "No file provided"
    meta: Dict[str, Any] = {}
    if file_path:
        path = Path(file_path)
        meta = {
            "name": path.name,
            "suffix": path.suffix,
            "size": path.stat().st_size if path.exists() else None,
        }
        message = f"Ingested {path.name}"
    return PipelineResponse(message=message, meta=meta)


def analyze(run_id: str, file_path: str | None) -> PipelineResponse:
    insights = {"summary": "No analysis performed"}
    if file_path and Path(file_path).exists():
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                content = handle.read(5000)
            words = len(content.split())
            lines = content.count("\n") + 1
            insights = {"words": words, "lines": lines}
        except Exception:
            insights = {"summary": "Binary or unreadable file"}
    return PipelineResponse(message="Analysis complete", meta=insights)


def visualize(run_id: str, file_path: str | None) -> PipelineResponse:
    return PipelineResponse(message="Visualization stage complete", meta={"chart": "placeholder"})


def reflect(run_id: str, file_path: str | None) -> PipelineResponse:
    prompt = [
        {"role": "system", "content": "You are Leviathan, an autonomous data co-worker."},
        {"role": "user", "content": "Summarize the pipeline run and suggest one next step."},
    ]
    try:
        text = chat_completion(prompt)
    except Exception:
        text = "Reflection unavailable (LLM not configured)."
    alert = {
        "level": "info",
        "message": "Proactive suggestion generated",
    }
    return PipelineResponse(message=text, alert=alert)
