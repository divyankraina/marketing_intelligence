"""
Load Testing Script for Marketing Intelligence API
Tests system under concurrent load to validate performance and stability.
"""
import requests
import time
import concurrent.futures
import statistics
from typing import List, Dict
import json

BASE_URL = "http://localhost:8000"

def test_predict_discount():
    """Single prediction request"""
    payload = {
        "product_name": "Samsung Galaxy S23 Ultra",
        "category": "Electronics|Mobiles",
        "actual_price": 124999.0,
        "rating": 4.6,
        "rating_count": 5000,
        "about_product": "Flagship phone with 200MP camera",
        "review_content": "Great phone"
    }
    start = time.time()
    try:
        response = requests.post(f"{BASE_URL}/predict_discount", json=payload, timeout=30)
        latency = time.time() - start
        return {
            "success": response.status_code == 200,
            "latency": latency,
            "status_code": response.status_code
        }
    except Exception as e:
        return {
            "success": False,
            "latency": time.time() - start,
            "error": str(e)
        }

def test_answer_question():
    """Single RAG query request"""
    payload = {"query": "What is the highest rated product?"}
    start = time.time()
    try:
        response = requests.post(f"{BASE_URL}/answer_question", json=payload, timeout=60)
        latency = time.time() - start
        return {
            "success": response.status_code in [200, 503],  # 503 is OK if training
            "latency": latency,
            "status_code": response.status_code
        }
    except Exception as e:
        return {
            "success": False,
            "latency": time.time() - start,
            "error": str(e)
        }

def run_load_test(endpoint_func, num_requests: int = 100, concurrency: int = 10):
    """Run load test with specified concurrency"""
    print(f"\n🚀 Load Testing: {endpoint_func.__name__}")
    print(f"   Requests: {num_requests}, Concurrency: {concurrency}")
    
    latencies = []
    successes = 0
    errors = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(endpoint_func) for _ in range(num_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result.get("success"):
                successes += 1
                latencies.append(result["latency"])
            else:
                errors.append(result)
    
    if latencies:
        print(f"   ✅ Success Rate: {successes}/{num_requests} ({100*successes/num_requests:.1f}%)")
        print(f"   ⏱️  Mean Latency: {statistics.mean(latencies):.3f}s")
        print(f"   📊 Median Latency: {statistics.median(latencies):.3f}s")
        print(f"   📈 P95 Latency: {sorted(latencies)[int(len(latencies)*0.95)]:.3f}s")
        print(f"   📉 Min Latency: {min(latencies):.3f}s")
        print(f"   📈 Max Latency: {max(latencies):.3f}s")
    else:
        print(f"   ❌ All requests failed!")
    
    if errors:
        print(f"   ⚠️  Errors: {len(errors)}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"      - {error.get('error', 'Unknown error')}")
    
    return {
        "success_rate": successes / num_requests if num_requests > 0 else 0,
        "mean_latency": statistics.mean(latencies) if latencies else 0,
        "p95_latency": sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
        "errors": len(errors)
    }

def test_health_endpoint():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Marketing Intelligence API - Load Testing")
    print("=" * 60)
    
    # Check if API is available
    if not test_health_endpoint():
        print("❌ API is not available. Please start the server first.")
        exit(1)
    
    print("✅ API is available. Starting load tests...\n")
    
    # Test 1: Prediction endpoint (lightweight)
    results1 = run_load_test(test_predict_discount, num_requests=50, concurrency=5)
    
    # Test 2: RAG endpoint (heavier)
    results2 = run_load_test(test_answer_question, num_requests=20, concurrency=3)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Load Test Summary")
    print("=" * 60)
    print(f"Prediction Endpoint:")
    print(f"  Success Rate: {results1['success_rate']*100:.1f}%")
    print(f"  Mean Latency: {results1['mean_latency']:.3f}s")
    print(f"  P95 Latency: {results1['p95_latency']:.3f}s")
    
    print(f"\nRAG Endpoint:")
    print(f"  Success Rate: {results2['success_rate']*100:.1f}%")
    print(f"  Mean Latency: {results2['mean_latency']:.3f}s")
    print(f"  P95 Latency: {results2['p95_latency']:.3f}s")
    
    # Safety checks
    print("\n🔒 Safety Validation:")
    if results1['success_rate'] >= 0.95 and results2['success_rate'] >= 0.80:
        print("  ✅ System meets performance requirements")
    else:
        print("  ⚠️  System performance below recommended thresholds")
    
    if results1['p95_latency'] < 5.0 and results2['p95_latency'] < 30.0:
        print("  ✅ Latency within acceptable ranges")
    else:
        print("  ⚠️  High latency detected - consider optimization")
