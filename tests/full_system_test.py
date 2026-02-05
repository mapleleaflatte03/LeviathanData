#!/usr/bin/env python3
"""
Leviathan Full System Test
Tests all components: HF ViT/BERT, pandas, sklearn, XGBoost, Plotly, pipelines
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test results tracking
RESULTS = {
    "passed": [],
    "failed": [],
    "warnings": [],
    "start_time": None,
    "end_time": None
}

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    symbol = {"INFO": "ℹ️", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(level, "•")
    print(f"[{timestamp}] {symbol} {msg}")

def test_passed(name, details=""):
    RESULTS["passed"].append({"name": name, "details": details})
    log(f"{name} - {details}", "PASS")

def test_failed(name, error):
    RESULTS["failed"].append({"name": name, "error": str(error)})
    log(f"{name} - {error}", "FAIL")

def test_warning(name, msg):
    RESULTS["warnings"].append({"name": name, "message": msg})
    log(f"{name} - {msg}", "WARN")

# ===== TEST 1: Core Imports =====
def test_core_imports():
    log("Testing core imports...")
    try:
        import numpy as np
        import pandas as pd
        test_passed("NumPy import", f"v{np.__version__}")
        test_passed("Pandas import", f"v{pd.__version__}")
    except Exception as e:
        test_failed("Core imports", e)
        return False
    return True

# ===== TEST 2: ML Libraries =====
def test_ml_libraries():
    log("Testing ML libraries...")
    
    # Scikit-learn
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import sklearn
        test_passed("Scikit-learn", f"v{sklearn.__version__}")
    except Exception as e:
        test_failed("Scikit-learn", e)
    
    # XGBoost
    try:
        import xgboost as xgb
        test_passed("XGBoost", f"v{xgb.__version__}")
    except Exception as e:
        test_warning("XGBoost", f"Not available: {e}")
    
    # LightGBM
    try:
        import lightgbm as lgb
        test_passed("LightGBM", f"v{lgb.__version__}")
    except Exception as e:
        test_warning("LightGBM", f"Not available: {e}")

# ===== TEST 3: Hugging Face =====
def test_huggingface():
    log("Testing Hugging Face Transformers...")
    
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        test_passed("Transformers import", "OK")
        test_passed("PyTorch import", f"v{torch.__version__}")
        
        # Test BERT spam detection
        log("Testing BERT spam classifier...")
        spam_clf = hf_pipeline('text-classification', 
                              model='mrm8488/bert-tiny-finetuned-sms-spam-detection',
                              device=-1)
        result = spam_clf("You won a FREE iPhone!")
        label = result[0]['label']
        score = result[0]['score']
        if score > 0.7:
            test_passed("BERT spam detection", f"Label={label}, Score={score:.3f}")
        else:
            test_warning("BERT spam detection", f"Low confidence: {score:.3f}")
        
        # Test ViT image classification (if image available)
        test_images = list(Path("/root/leviathan/data/uploads").glob("*.jpg"))
        if test_images:
            log("Testing ViT image classifier...")
            from transformers import ViTImageProcessor, ViTForImageClassification
            from PIL import Image
            
            processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            
            img = Image.open(test_images[0]).convert('RGB')
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            predicted_class = outputs.logits.argmax(-1).item()
            test_passed("ViT image classification", f"Class={predicted_class}")
        else:
            test_warning("ViT image classification", "No test images found")
            
    except Exception as e:
        test_failed("Hugging Face", f"{type(e).__name__}: {e}")

# ===== TEST 4: Orchestrator Pipeline =====
def test_orchestrator():
    log("Testing Orchestrator pipeline...")
    
    try:
        from backend.python.orchestrator import (
            ingest, analyze, visualize, reflect,
            HF_AVAILABLE
        )
        test_passed("Orchestrator import", f"HF_AVAILABLE={HF_AVAILABLE}")
        
        # Test with each test file
        test_files = [
            "/root/leviathan/data/uploads/test_csv_sales.csv",
            "/root/leviathan/data/uploads/test_timeseries.csv",
            "/root/leviathan/data/uploads/test_text_spam.csv",
            "/root/leviathan/data/uploads/test_mixed.csv",
        ]
        
        for test_file in test_files:
            if not os.path.exists(test_file):
                test_warning(f"Pipeline {Path(test_file).name}", "File not found")
                continue
                
            fname = Path(test_file).name
            run_id = f"test_{fname}_{int(time.time())}"
            
            try:
                log(f"  Ingesting {fname}...")
                ingest_result = ingest(run_id, test_file)
                
                log(f"  Analyzing {fname}...")
                analyze_result = analyze(run_id, test_file)
                
                log(f"  Visualizing {fname}...")
                viz_result = visualize(run_id, test_file)
                
                log(f"  Reflecting {fname}...")
                reflect_result = reflect(run_id, test_file)
                
                # Check accuracy if available
                accuracy = None
                if hasattr(analyze_result, 'payload') and isinstance(analyze_result.payload, dict):
                    accuracy = analyze_result.payload.get("accuracy") or analyze_result.payload.get("r2")
                elif isinstance(analyze_result, dict):
                    accuracy = analyze_result.get("accuracy") or analyze_result.get("r2")
                
                if accuracy and accuracy > 0.7:
                    test_passed(f"Pipeline {fname}", f"Accuracy={accuracy:.2%}")
                elif accuracy:
                    test_warning(f"Pipeline {fname}", f"Low accuracy={accuracy:.2%}")
                else:
                    test_passed(f"Pipeline {fname}", "Completed (no accuracy metric)")
                    
            except Exception as e:
                test_failed(f"Pipeline {fname}", f"{type(e).__name__}: {e}")
                
    except Exception as e:
        test_failed("Orchestrator", f"{type(e).__name__}: {e}")
        traceback.print_exc()

# ===== TEST 5: Plotly Charts =====
def test_plotly():
    log("Testing Plotly visualization...")
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        
        # Create test chart
        df = pd.DataFrame({
            "x": range(10),
            "y": [i**2 for i in range(10)],
            "category": ["A", "B"] * 5
        })
        
        fig = px.bar(df, x="x", y="y", color="category")
        html = fig.to_html(include_plotlyjs=False)
        
        if len(html) > 1000:
            test_passed("Plotly charts", f"Generated {len(html)} bytes HTML")
        else:
            test_warning("Plotly charts", "Output seems small")
            
    except Exception as e:
        test_failed("Plotly", e)

# ===== TEST 6: Chroma Memory =====
def test_chroma():
    log("Testing Chroma vector store...")
    
    try:
        import chromadb
        
        client = chromadb.Client()
        collection = client.get_or_create_collection("test_collection")
        
        # Add test documents
        collection.add(
            documents=["Test document 1", "Test document 2", "ML pipeline test"],
            ids=["id1", "id2", "id3"]
        )
        
        # Query
        results = collection.query(query_texts=["pipeline"], n_results=1)
        
        if results and results['documents']:
            test_passed("Chroma memory", f"Query returned: {results['documents'][0][:50]}")
        else:
            test_warning("Chroma memory", "Empty query results")
            
        # Cleanup
        client.delete_collection("test_collection")
        
    except Exception as e:
        test_warning("Chroma memory", f"Not available: {e}")

# ===== TEST 7: API Endpoints =====
def test_api_endpoints():
    log("Testing API endpoints...")
    
    try:
        import requests
        
        # Health check
        resp = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if resp.status_code == 200:
            test_passed("Python API /health", resp.json())
        else:
            test_failed("Python API /health", f"Status {resp.status_code}")
            
    except Exception as e:
        test_warning("Python API", f"Not reachable: {e}")
    
    try:
        import requests
        
        # Node health check
        resp = requests.get("http://127.0.0.1:3000/api/health", timeout=5)
        if resp.status_code == 200:
            test_passed("Node API /api/health", resp.json())
        else:
            test_failed("Node API /api/health", f"Status {resp.status_code}")
            
    except Exception as e:
        test_warning("Node API", f"Not reachable: {e}")

# ===== TEST 8: Tool Registry =====
def test_tool_registry():
    log("Testing Tool Registry...")
    
    try:
        from backend.python.tool_registry import TOOL_MODULES, run_tool
        
        tool_count = len(TOOL_MODULES)
        test_passed("Tool Registry", f"{tool_count} tool modules registered")
        
        # List some tools
        for name in list(TOOL_MODULES.keys())[:5]:
            log(f"  - {name}")
            
    except Exception as e:
        test_failed("Tool Registry", e)

# ===== TEST 9: Proactive Scanner =====
def test_proactive_scanner():
    log("Testing Proactive Scanner...")
    
    try:
        from backend.python.background import start_background_jobs
        
        # Just verify it can be imported
        test_passed("Proactive Scanner", "start_background_jobs available")
        
    except Exception as e:
        test_warning("Proactive Scanner", f"Not available: {e}")

# ===== MAIN =====
def main():
    RESULTS["start_time"] = datetime.now()
    
    print("\n" + "="*60)
    print("🐙 LEVIATHAN FULL SYSTEM TEST")
    print("="*60 + "\n")
    
    # Run all tests
    test_core_imports()
    test_ml_libraries()
    test_huggingface()
    test_plotly()
    test_chroma()
    test_api_endpoints()
    test_tool_registry()
    test_proactive_scanner()
    test_orchestrator()
    
    RESULTS["end_time"] = datetime.now()
    duration = (RESULTS["end_time"] - RESULTS["start_time"]).total_seconds()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = len(RESULTS["passed"])
    failed = len(RESULTS["failed"])
    warnings = len(RESULTS["warnings"])
    total = passed + failed + warnings
    
    print(f"\n✅ Passed:   {passed}")
    print(f"❌ Failed:   {failed}")
    print(f"⚠️  Warnings: {warnings}")
    print(f"⏱️  Duration: {duration:.1f}s")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    if failed > 0:
        print("\n❌ FAILURES:")
        for f in RESULTS["failed"]:
            print(f"  - {f['name']}: {f['error']}")
    
    print("\n" + "="*60)
    
    # Return success if >90%
    return success_rate >= 90

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
