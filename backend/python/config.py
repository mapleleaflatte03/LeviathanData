import os
from pathlib import Path


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "db_path": Path(os.getenv("DB_PATH", ROOT / "data" / "leviathan.db")),
    "upload_dir": Path(os.getenv("UPLOAD_DIR", ROOT / "data" / "uploads")),
    "report_dir": Path(os.getenv("REPORT_DIR", ROOT / "data" / "reports")),
    "llm_base_url": os.getenv("LLM_BASE_URL", ""),
    "llm_api_key": os.getenv("LLM_API_KEY", ""),
    "llm_model": os.getenv("LLM_MODEL", "qwen3-32b"),
    "llm_fallback_base_url": os.getenv("LLM_FALLBACK_BASE_URL", ""),
    "llm_fallback_api_key": os.getenv("LLM_FALLBACK_API_KEY", ""),
    "llm_fallback_model": os.getenv("LLM_FALLBACK_MODEL", ""),
    "log_level": os.getenv("LOG_LEVEL", "info"),
    "python_autostart": _bool(os.getenv("PYTHON_AUTOSTART"), True),
}
