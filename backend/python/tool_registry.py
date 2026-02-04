import importlib
from typing import Any, Dict

TOOL_MODULES = {
    "orchestration_openclaw": "tools.orchestration_openclaw",
    "orchestration_langgraph": "tools.orchestration_langgraph",
    "orchestration_crewai": "tools.orchestration_crewai",
    "orchestration_autogen": "tools.orchestration_autogen",
    "orchestration_semantickernel": "tools.orchestration_semantickernel",
    "orchestration_airflow": "tools.orchestration_airflow",
    "orchestration_prefect": "tools.orchestration_prefect",
    "orchestration_dbt": "tools.orchestration_dbt",
    "orchestration_nifi": "tools.orchestration_nifi",
    "orchestration_mage": "tools.orchestration_mage",
    "ml_pandas": "tools.ml_pandas",
    "ml_numpy": "tools.ml_numpy",
    "ml_sklearn": "tools.ml_sklearn",
    "ml_xgboost": "tools.ml_xgboost",
    "ml_lightgbm": "tools.ml_lightgbm",
    "ml_pytorch": "tools.ml_pytorch",
    "ml_tensorflow": "tools.ml_tensorflow",
    "ml_transformers": "tools.ml_transformers",
    "ml_flaml": "tools.ml_flaml",
    "viz_superset": "tools.viz_superset",
    "viz_metabase": "tools.viz_metabase",
    "viz_plotly_dash": "tools.viz_plotly_dash",
    "viz_streamlit": "tools.viz_streamlit",
    "viz_evidence": "tools.viz_evidence",
    "db_chroma": "tools.db_chroma",
    "db_weaviate": "tools.db_weaviate",
    "db_qdrant": "tools.db_qdrant",
    "db_neo4j": "tools.db_neo4j",
    "db_lancedb": "tools.db_lancedb",
    "db_sqlite": "tools.db_sqlite",
    "mm_whisper": "tools.mm_whisper",
    "mm_openclip": "tools.mm_openclip",
    "mm_tesseract": "tools.mm_tesseract",
    "mm_ffmpeg": "tools.mm_ffmpeg",
    "infra_docker": "tools.infra_docker",
    "infra_kubernetes": "tools.infra_kubernetes",
    "infra_traefik": "tools.infra_traefik",
    "infra_caddy": "tools.infra_caddy",
    "infra_prometheus_grafana": "tools.infra_prometheus_grafana",
    "browser_puppeteer": "tools.browser_puppeteer",
    "browser_playwright": "tools.browser_playwright",
}


def run_tool(tool_name: str, args: Dict[str, Any]):
    if tool_name not in TOOL_MODULES:
        raise KeyError(f"Unknown tool: {tool_name}")
    module = importlib.import_module(TOOL_MODULES[tool_name])
    if not hasattr(module, "run"):
        raise AttributeError(f"Tool {tool_name} missing run()")
    return module.run(args)


def list_tools():
    return sorted(TOOL_MODULES.keys())
