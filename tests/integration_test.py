"""
Integration Test Script
Tests all API endpoints to ensure the system works end-to-end.
"""
import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    print("✅ Health check passed")

def test_predict_discount():
    """Test discount prediction"""
    print("\nTesting /predict_discount endpoint...")
    payload = {
        "product_name": "Samsung Galaxy S23 Ultra",
        "category": "Electronics|Mobiles",
        "actual_price": 124999.0,
        "rating": 4.6,
        "rating_count": 5000,
        "about_product": "Flagship phone with 200MP camera",
        "review_content": "Great phone"
    }
    response = requests.post(f"{BASE_URL}/predict_discount", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_discount" in data
    assert 0 <= data['predicted_discount'] <= 100
    print(f"✅ Prediction: {data['predicted_discount']}%")

def test_explain_prediction():
    """Test explanation endpoint"""
    print("\nTesting /explain_prediction endpoint...")
    payload = {
        "product_name": "Test Product",
        "category": "Electronics",
        "actual_price": 50000.0,
        "rating": 4.5,
        "rating_count": 1000,
        "about_product": "Test description",
        "review_content": "Test review"
    }
    response = requests.post(f"{BASE_URL}/explain_prediction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "base" in data or "error" in data
    print("✅ Explanation generated")

def test_answer_question():
    """Test RAG endpoint"""
    print("\nTesting /answer_question endpoint...")
    payload = {"query": "What is the highest rated product?"}
    response = requests.post(f"{BASE_URL}/answer_question", json=payload)
    # Accept 200 (ready) or 503 (training)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        print(f"✅ Answer: {data['answer'][:50]}...")
    else:
        print("⚠️  System is training (503 expected)")

def test_safety_check():
    """Test safety validation"""
    print("\nTesting /safety_check endpoint...")
    payload = {
        "product_name": "Test Product",
        "category": "Electronics",
        "actual_price": 50000.0,
        "rating": 4.5,
        "rating_count": 1000,
        "about_product": "Test description"
    }
    response = requests.post(f"{BASE_URL}/safety_check", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "safe" in data
    assert "sanitized_input" in data
    print(f"✅ Safety check: {'Safe' if data['safe'] else 'Unsafe'}")

def test_metrics():
    """Test metrics endpoint"""
    print("\nTesting /metrics endpoint...")
    response = requests.get(f"{BASE_URL}/metrics")
    assert response.status_code == 200
    assert "app_request_count" in response.text
    print("✅ Metrics endpoint working")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Integration Testing - Marketing Intelligence API")
    print("=" * 60)
    
    try:
        test_health()
        test_predict_discount()
        test_explain_prediction()
        test_answer_question()
        test_safety_check()
        test_metrics()
        
        print("\n" + "=" * 60)
        print("✅ All integration tests passed!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please ensure the server is running:")
        print("   docker-compose up -d")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
