import time
import requests
from typing import List, Dict, Any
from .config import CONFIG

MAX_RETRIES = 5
BASE_DELAY_SEC = 0.5


def _headers(api_key: str | None) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_chat_once(base_url: str, api_key: str, payload: Dict[str, Any]) -> requests.Response:
    if not base_url:
        raise RuntimeError("LLM base URL not configured")
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    return requests.post(url, headers=_headers(api_key), json=payload, timeout=60)


def _post_chat_with_retry(base_url: str, api_key: str, payload: Dict[str, Any], is_fallback: bool = False) -> requests.Response:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = _post_chat_once(base_url, api_key, payload)
            res.raise_for_status()
            return res
        except Exception as err:
            last_err = err
            wait_sec = BASE_DELAY_SEC * (2 ** (attempt - 1))  # 0.5, 1, 2, 4, 8 seconds
            label = " (fallback)" if is_fallback else ""
            print(f"[LLM] Attempt {attempt}/{MAX_RETRIES} failed{label}: {err}. Retrying in {wait_sec}s...")
            if attempt < MAX_RETRIES:
                time.sleep(wait_sec)
    raise last_err  # type: ignore


def chat_completion(messages: List[Dict[str, Any]]) -> str:
    payload = {"model": CONFIG["llm_model"], "messages": messages, "stream": False}
    try:
        res = _post_chat_with_retry(CONFIG["llm_base_url"], CONFIG["llm_api_key"], payload, is_fallback=False)
        data = res.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        if not CONFIG["llm_fallback_base_url"]:
            raise
        print(f"[LLM] Primary exhausted after {MAX_RETRIES} attempts, trying fallback...")
        payload["model"] = CONFIG["llm_fallback_model"] or CONFIG["llm_model"]
        res = _post_chat_with_retry(CONFIG["llm_fallback_base_url"], CONFIG["llm_fallback_api_key"], payload, is_fallback=True)
        data = res.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
