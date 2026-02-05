#!/usr/bin/env python3
"""
Test script to verify real LLM API calls via Qwen endpoint.
Sends 5 sample prompts and logs raw request/response + token counts.
"""

import sys
import os
import time
import json

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.python.llm_client import chat_completion, get_llm_stats, set_log_callback

# Track all logs
all_logs = []

def log_callback(entry):
    all_logs.append(entry)
    print(f"  📡 {entry['type']}: {json.dumps({k: v for k, v in entry.items() if k != 'type' and k != 'ts'}, indent=2)[:500]}")

set_log_callback(log_callback)

TEST_PROMPTS = [
    {
        "name": "Summarize CSV",
        "messages": [
            {"role": "system", "content": "You are a data analysis assistant."},
            {"role": "user", "content": "Summarize this CSV data: date,product,quantity,revenue\\n2024-01-01,Widget A,100,5000\\n2024-01-02,Widget B,150,7500\\n2024-01-03,Widget A,120,6000"}
        ]
    },
    {
        "name": "Classify Image",
        "messages": [
            {"role": "system", "content": "You are an image classification assistant."},
            {"role": "user", "content": "Based on this description, classify the image: A photograph showing a cat sitting on a windowsill with sunlight streaming in."}
        ]
    },
    {
        "name": "Forecast Time-series",
        "messages": [
            {"role": "system", "content": "You are a time-series forecasting assistant."},
            {"role": "user", "content": "Given this time series: [100, 102, 98, 105, 110, 108, 115], predict the next 3 values and explain your reasoning."}
        ]
    },
    {
        "name": "Detect Anomaly",
        "messages": [
            {"role": "system", "content": "You are an anomaly detection assistant."},
            {"role": "user", "content": "Analyze this sequence for anomalies: [10, 11, 12, 10, 11, 500, 12, 11, 10]. Identify any outliers and explain why."}
        ]
    },
    {
        "name": "Generate Insights",
        "messages": [
            {"role": "system", "content": "You are a business intelligence assistant."},
            {"role": "user", "content": "Generate 3 key insights from this sales data: Q1: $50K, Q2: $65K, Q3: $45K, Q4: $80K. Consider seasonality and trends."}
        ]
    }
]


def run_tests():
    print("\n" + "=" * 60)
    print("🔬 LEVIATHAN LLM API TEST SCRIPT")
    print("=" * 60)
    print(f"Testing {len(TEST_PROMPTS)} prompts against Qwen endpoint...\n")
    
    results = []
    
    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'─' * 50}")
        print(f"📋 Test {i}/{len(TEST_PROMPTS)}: {test['name']}")
        print(f"{'─' * 50}")
        
        prompt_preview = test['messages'][-1]['content'][:100]
        print(f"  Prompt: {prompt_preview}...")
        
        start_time = time.time()
        try:
            response = chat_completion(test['messages'])
            elapsed = time.time() - start_time
            
            print(f"\n  ✅ SUCCESS ({elapsed:.2f}s)")
            print(f"  Response preview: {response[:200]}..." if len(response) > 200 else f"  Response: {response}")
            
            results.append({
                "name": test['name'],
                "success": True,
                "elapsed_sec": round(elapsed, 2),
                "response_length": len(response),
                "response_preview": response[:300]
            })
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n  ❌ FAILED ({elapsed:.2f}s): {e}")
            results.append({
                "name": test['name'],
                "success": False,
                "elapsed_sec": round(elapsed, 2),
                "error": str(e)
            })
        
        # Small delay between requests
        if i < len(TEST_PROMPTS):
            time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    stats = get_llm_stats()
    passed = sum(1 for r in results if r['success'])
    
    print(f"\n  Tests passed: {passed}/{len(results)}")
    print(f"  Total LLM calls: {stats['total_calls']}")
    print(f"  Total tokens in: {stats['total_tokens_in']}")
    print(f"  Total tokens out: {stats['total_tokens_out']}")
    print(f"  LLM healthy: {stats['healthy']}")
    print(f"  Last endpoint: {stats['last_endpoint']}")
    print(f"  Last model: {stats['last_model']}")
    print(f"  Errors: {stats['errors']}")
    
    print("\n  Individual results:")
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"    {status} {r['name']}: {r['elapsed_sec']}s")
    
    print("\n" + "=" * 60)
    
    # Return exit code based on results
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
