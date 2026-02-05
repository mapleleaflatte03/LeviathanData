from backend.python.orchestrator import ingest, analyze, visualize, reflect
from pathlib import Path


def test_orchestrator_pipeline(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello leviathan\nthis is a test")

    ingest_res = ingest("run1", str(file_path))
    assert "Ingested" in ingest_res.message

    analyze_res = analyze("run1", str(file_path))
    assert analyze_res.meta.get("ml")
    assert analyze_res.meta["ml"].get("primaryMetric") in {"accuracy", "r2", "quality"}

    viz_res = visualize("run1", str(file_path))
    assert viz_res.message == "Visualization stage complete"
    assert viz_res.meta.get("svgPath")

    reflect_res = reflect("run1", str(file_path))
    assert reflect_res.message
    assert reflect_res.meta.get("proactiveScan")
