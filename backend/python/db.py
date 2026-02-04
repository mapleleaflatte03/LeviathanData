import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional
from .config import CONFIG


def _connect():
    db_path = Path(CONFIG["db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def update_run_meta(run_id: str, meta: Dict[str, Any]):
    with _connect() as conn:
        conn.execute(
            "UPDATE runs SET meta_json = ? WHERE id = ?",
            (json_dumps(meta), run_id),
        )
        conn.commit()


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=True)
