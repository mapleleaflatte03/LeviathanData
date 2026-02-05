import time
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from .config import CONFIG

MAX_RETRIES = 5
BASE_DELAY_SEC = 0.5

logger = logging.getLogger("llm_client")

# LLM stats tracking
llm_stats = {
    "total_calls": 0,
    "total_tokens_in": 0,
    "total_tokens_out": 0,
    "last_call_time": None,
    "last_endpoint": None,
    "last_model": None,
    "errors": 0,
    "healthy": True
}

# Optional callback for log events (can be set by orchestrator)
_log_callback: Optional[Callable[[Dict[str, Any]], None]] = None

def set_log_callback(callback: Callable[[Dict[str, Any]], None]) -> None:
    global _log_callback
    _log_callback = callback

def get_llm_stats() -> Dict[str, Any]:
    return dict(llm_stats)

def _log_llm_call(log_type: str, data: Dict[str, Any]) -> None:
    log_entry = {"type": log_type, "ts": datetime.utcnow().isoformat() + "Z", **data}
    logger.info(f"[LLM:{log_type}] {json.dumps(data)}")
    if _log_callback:
        try:
            _log_callback(log_entry)
        except Exception:
            pass


def _headers(api_key: str | None) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_chat_once(base_url: str, api_key: str, payload: Dict[str, Any], is_fallback: bool = False) -> requests.Response:
    if not base_url:
        raise RuntimeError("LLM base URL not configured")
    url = f"{base_url.rstrip('/')}/api/v1/chat/completions"
    
    llm_stats["total_calls"] += 1
    llm_stats["last_call_time"] = datetime.utcnow().isoformat() + "Z"
    llm_stats["last_endpoint"] = url
    llm_stats["last_model"] = payload.get("model", "unknown")
    
    prompt_preview = ""
    if payload.get("messages"):
        last_msg = payload["messages"][-1].get("content", "")
        prompt_preview = last_msg[:200] if isinstance(last_msg, str) else str(last_msg)[:200]
    
    _log_llm_call("REQUEST", {
        "endpoint": url,
        "model": payload.get("model"),
        "messageCount": len(payload.get("messages", [])),
        "promptPreview": prompt_preview,
        "useFallback": is_fallback
    })
    
    start_time = time.time()
    res = requests.post(url, headers=_headers(api_key), json=payload, timeout=60)
    latency_ms = int((time.time() - start_time) * 1000)
    
    if not res.ok:
        llm_stats["errors"] += 1
        llm_stats["healthy"] = False
        _log_llm_call("ERROR", {"status": res.status_code, "error": res.text[:500], "latencyMs": latency_ms})
    
    return res


def _post_chat_with_retry(base_url: str, api_key: str, payload: Dict[str, Any], is_fallback: bool = False) -> requests.Response:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = _post_chat_once(base_url, api_key, payload, is_fallback)
            res.raise_for_status()
            llm_stats["healthy"] = True
            return res
        except Exception as err:
            last_err = err
            wait_sec = BASE_DELAY_SEC * (2 ** (attempt - 1))
            _log_llm_call("RETRY", {"attempt": attempt, "maxRetries": MAX_RETRIES, "waitSec": wait_sec, "error": str(err), "useFallback": is_fallback})
            if attempt < MAX_RETRIES:
                time.sleep(wait_sec)
    raise last_err  # type: ignore


def chat_completion(messages: List[Dict[str, Any]], log_response: bool = True) -> str:
    payload = {"model": CONFIG["llm_model"], "messages": messages, "stream": False}
    start_time = time.time()
    
    try:
        res = _post_chat_with_retry(CONFIG["llm_base_url"], CONFIG["llm_api_key"], payload, is_fallback=False)
        data = res.json()
    except Exception:
        if not CONFIG["llm_fallback_base_url"]:
            raise
        _log_llm_call("FALLBACK", {"reason": f"Primary exhausted after {MAX_RETRIES} attempts"})
        payload["model"] = CONFIG["llm_fallback_model"] or CONFIG["llm_model"]
        res = _post_chat_with_retry(CONFIG["llm_fallback_base_url"], CONFIG["llm_fallback_api_key"], payload, is_fallback=True)
        data = res.json()
    
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    
    llm_stats["total_tokens_in"] += usage.get("prompt_tokens", 0)
    llm_stats["total_tokens_out"] += usage.get("completion_tokens", 0)
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    if log_response:
        _log_llm_call("RESPONSE", {
            "model": payload.get("model"),
            "tokensIn": usage.get("prompt_tokens", 0),
            "tokensOut": usage.get("completion_tokens", 0),
            "totalTokens": usage.get("total_tokens", 0),
            "latencyMs": latency_ms,
            "responsePreview": content[:300] if content else ""
        })
    
    return content
