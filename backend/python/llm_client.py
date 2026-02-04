import requests
from typing import List, Dict, Any
from .config import CONFIG


def _headers(api_key: str | None) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_chat(base_url: str, api_key: str, payload: Dict[str, Any]) -> requests.Response:
    if not base_url:
        raise RuntimeError("LLM base URL not configured")
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    return requests.post(url, headers=_headers(api_key), json=payload, timeout=60)


def chat_completion(messages: List[Dict[str, Any]]) -> str:
    payload = {"model": CONFIG["llm_model"], "messages": messages, "stream": False}
    try:
        res = _post_chat(CONFIG["llm_base_url"], CONFIG["llm_api_key"], payload)
        res.raise_for_status()
        data = res.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        if not CONFIG["llm_fallback_base_url"]:
            raise
        payload["model"] = CONFIG["llm_fallback_model"] or CONFIG["llm_model"]
        res = _post_chat(CONFIG["llm_fallback_base_url"], CONFIG["llm_fallback_api_key"], payload)
        res.raise_for_status()
        data = res.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
