#!/usr/bin/env python3
"""
Leviathan Pipeline Test Runner
Runs end-to-end pipeline on 6 selected datasets:
  titanic, creditcardfraud, dogs-vs-cats, imdb-reviews, stock-market, fake-news
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add backend package to path
backend_path = str(Path(__file__).parent.parent / "backend")
sys.path.insert(0, backend_path)

from python.orchestrator import ingest, analyze, visualize, reflect, _run_dir


DATASETS = [
    ("titanic", "data/uploads/titanic__train.csv"),
    ("creditcardfraud", "data/uploads/creditcardfraud__data.csv"),
    ("dogs-vs-cats", "data/uploads/dogs-vs-cats__cat1.jpg"),
    ("imdb-reviews", "data/uploads/imdb-reviews__data.csv"),
    ("stock-market", "data/uploads/stock-market__AAPL.csv"),
    ("fake-news", "data/uploads/fake-news__Fake.csv"),
]

REPORT_DIR = Path(__file__).parent.parent / "data" / "reports"


def run_pipeline(name: str, file_path: str) -> dict:
    """Run full pipeline on a single dataset."""
    run_id = f"test-{name}-{int(time.time())}"
    base_path = Path(__file__).parent.parent
    full_path = str(base_path / file_path)
    
    print(f"\n{'='*60}")
    print(f"🔄 PIPELINE: {name.upper()}")
    print(f"{'='*60}")
    
    results = {
        "runId": run_id,
        "dataset": name,
        "filePath": full_path,
        "startTime": datetime.utcnow().isoformat(),
        "stages": {},
        "success": False,
        "needsRefinement": False,
        "errors": [],
    }
    
    try:
        # Stage 1: Ingest
        print(f"  📥 Ingest...")
        ingest_result = ingest(run_id, full_path)
        results["stages"]["ingest"] = {
            "message": ingest_result.message,
            "meta": ingest_result.meta,
        }
        print(f"     ✓ {ingest_result.message}")
        
        # Stage 2: Analyze (includes cleaning, EDA, ML)
        print(f"  🔬 Analyze (clean → EDA → ML)...")
        analyze_result = analyze(run_id, full_path)
        results["stages"]["analyze"] = {
            "message": analyze_result.message,
            "meta": analyze_result.meta,
            "alert": analyze_result.alert,
        }
        
        ml_meta = analyze_result.meta.get("ml", {}) if analyze_result.meta else {}
        metric_name = ml_meta.get("primaryMetric", "quality")
        metric_value = ml_meta.get("primaryMetricValue", 0.0)
        task = ml_meta.get("task", "unknown")
        model = ml_meta.get("selectedModel", "unknown")
        refined = ml_meta.get("refined", False)
        needs_refinement = ml_meta.get("needsRefinement", False)
        
        results["needsRefinement"] = needs_refinement
        results["metric"] = {"name": metric_name, "value": metric_value}
        results["task"] = task
        results["model"] = model
        results["refined"] = refined
        
        status_icon = "⚠️" if needs_refinement else "✓"
        refined_note = " (auto-refined)" if refined else ""
        print(f"     {status_icon} {task} | {model}{refined_note} | {metric_name}={metric_value:.4f}")
        
        if analyze_result.alert:
            print(f"     ⚡ Alert: {analyze_result.alert.get('message', '')}")
        
        # Stage 3: Visualize (SVG generation)
        print(f"  📊 Visualize (SVG)...")
        viz_result = visualize(run_id, full_path)
        results["stages"]["visualize"] = {
            "message": viz_result.message,
            "meta": viz_result.meta,
        }
        svg_path = viz_result.meta.get("svgPath", "") if viz_result.meta else ""
        svg_size = viz_result.meta.get("svgSize", 0) if viz_result.meta else 0
        print(f"     ✓ Generated SVG ({svg_size} bytes)")
        results["svgPath"] = svg_path
        
        # Stage 4: Reflect (insights + summary)
        print(f"  💡 Reflect (insights)...")
        reflect_result = reflect(run_id, full_path)
        results["stages"]["reflect"] = {
            "message": reflect_result.message,
            "meta": reflect_result.meta,
            "alert": reflect_result.alert,
        }
        
        reflect_meta = reflect_result.meta or {}
        quality = reflect_meta.get("quality", "unknown")
        insights = reflect_meta.get("insights", [])
        
        print(f"     ✓ Quality: {quality}")
        for insight in insights[:3]:
            print(f"       • {insight[:80]}{'...' if len(insight) > 80 else ''}")
        
        results["quality"] = quality
        results["insights"] = insights
        results["success"] = True
        results["endTime"] = datetime.utcnow().isoformat()
        
    except Exception as e:
        results["errors"].append(str(e))
        results["success"] = False
        results["endTime"] = datetime.utcnow().isoformat()
        print(f"     ❌ Error: {e}")
    
    return results


def generate_pdf_report(all_results: list) -> str:
    """Generate a simple PDF-like text report."""
    report_path = REPORT_DIR / f"pipeline_test_report_{int(time.time())}.txt"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "=" * 70,
        "LEVIATHAN PIPELINE TEST REPORT",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "=" * 70,
        "",
    ]
    
    passed = sum(1 for r in all_results if r["success"])
    refined = sum(1 for r in all_results if r.get("refined", False))
    
    lines.append(f"SUMMARY: {passed}/{len(all_results)} datasets passed")
    lines.append(f"AUTO-REFINED: {refined} datasets")
    lines.append("")
    
    for r in all_results:
        lines.append("-" * 50)
        lines.append(f"Dataset: {r['dataset'].upper()}")
        lines.append(f"Run ID: {r['runId']}")
        lines.append(f"Status: {'✓ PASS' if r['success'] else '✗ FAIL'}")
        
        if r.get("metric"):
            lines.append(f"Metric: {r['metric']['name']} = {r['metric']['value']:.4f}")
        
        lines.append(f"Task: {r.get('task', 'N/A')}")
        lines.append(f"Model: {r.get('model', 'N/A')}")
        lines.append(f"Quality: {r.get('quality', 'N/A')}")
        
        if r.get("refined"):
            lines.append("⚡ Auto-refinement was applied")
        
        if r.get("needsRefinement"):
            lines.append("⚠️ Accuracy below threshold - further tuning recommended")
        
        if r.get("errors"):
            for err in r["errors"]:
                lines.append(f"ERROR: {err}")
        
        lines.append("")
    
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def main():
    print("\n" + "=" * 70)
    print("🐙 LEVIATHAN PIPELINE TEST - FULL E2E EXECUTION")
    print("   Ingest → Clean → EDA → ML → Visualize → Reflect")
    print("=" * 70)
    
    all_results = []
    
    for name, file_path in DATASETS:
        result = run_pipeline(name, file_path)
        all_results.append(result)
    
    # Generate report
    print("\n" + "=" * 60)
    print("📄 GENERATING REPORT")
    print("=" * 60)
    
    report_path = generate_pdf_report(all_results)
    print(f"  ✓ Report saved: {report_path}")
    
    # Save JSON results
    json_path = REPORT_DIR / f"pipeline_results_{int(time.time())}.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"  ✓ JSON results: {json_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    
    for r in all_results:
        status = "✓" if r["success"] else "✗"
        metric = r.get("metric", {})
        metric_str = f"{metric.get('name', 'N/A')}={metric.get('value', 0):.4f}" if metric else "N/A"
        refined_str = " [REFINED]" if r.get("refined") else ""
        warn_str = " ⚠️" if r.get("needsRefinement") else ""
        print(f"  {status} {r['dataset']:<18} | {r.get('task', 'N/A'):<15} | {metric_str}{refined_str}{warn_str}")
    
    passed = sum(1 for r in all_results if r["success"])
    print(f"\n  RESULT: {passed}/{len(all_results)} passed")
    
    # Return exit code
    return 0 if passed == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
