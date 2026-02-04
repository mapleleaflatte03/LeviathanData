from pathlib import Path
from .config import CONFIG


def ensure_dirs():
    Path(CONFIG["upload_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["report_dir"]).mkdir(parents=True, exist_ok=True)
