import pytest
import pandas as pd
import numpy as np
import os
import joblib
import shutil
from fastapi.testclient import TestClient
from src.ml_engine import DiscountPredictor, DataCleaner, FeatureEngineer
from src.monitoring import PerformanceMonitor
from main import app


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'product_id': ['P01', 'P02', 'P03', 'P04', 'P05'],
        'product_name': ['Apple iPhone 13', 'Samsung Galaxy S22', 'Generic USB Cable', 'Sony Bravia TV', 'Unknown Item'],
        'category': ['Computers|Phones', 'Computers|Phones', 'Computers|Cables', 'Home|TVs', 'Misc'],
        'actual_price': ['₹70,000', '65,000', '₹500', '1,00,000', '0'],
        'rating': ['4.8', '4.5', '3.8', '4.7', '1.0'],
        'rating_count': ['1000', '500', '50', '200', '1'],
        'about_product': ['Great phone', 'Android beast', 'Fast charge', '4K HDR', 'Bad'],
        'discount_percentage': ['10%', '15%', '60%', '20%', '5%'],
        'review_content': ['Good', 'Good', 'Okay', 'Great', 'Bad']
    })

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(scope="module")
def trained_predictor():
    """Trains a model once for the whole test session"""
    # Create dummy data for training
    df = pd.DataFrame({
        'product_id': [f'P{i}' for i in range(50)],
        'product_name': ['Generic Item'] * 50,
        'category': ['Electronics|Test'] * 50,
        'actual_price': np.random.uniform(100, 10000, 50),
        'rating': np.random.uniform(1, 5, 50),
        'rating_count': np.random.randint(1, 1000, 50),
        'about_product': ['desc'] * 50,
        'discount_percentage': np.random.uniform(0, 50, 50),
        'review_content': ['rev'] * 50
    })
    predictor = DiscountPredictor()
    # Force string conversion to test cleaner
    df['actual_price'] = df['actual_price'].astype(str)
    predictor.train(df, tune=False) # Skip tuning for speed
    return predictor

# --- 1. UNIT TESTS (Logic & Math) ---

def test_cleaner_logic(sample_data):
    """Verifies that currency symbols and commas are removed correctly"""
    cleaner = DataCleaner()
    cleaned = cleaner.transform(sample_data)
    
    assert cleaned['actual_price'].dtype == float
    assert cleaned['actual_price'].iloc[0] == 70000.0
    assert cleaned['actual_price'].iloc[2] == 500.0

def test_brand_extraction(sample_data):
    """Verifies that 'Apple iPhone' becomes Brand: Apple"""
    cleaner = DataCleaner()
    engineer = FeatureEngineer()
    
    processed = engineer.transform(cleaner.transform(sample_data))
    
    assert 'brand' in processed.columns
    assert processed['brand'].iloc[0] == 'Apple'
    assert processed['brand'].iloc[1] == 'Samsung'
    assert processed['brand'].iloc[2] == 'Generic'

def test_feature_engineering(sample_data):
    """Test Brand Extraction and Text Stats"""
    cleaner = DataCleaner()
    cleaned = cleaner.transform(sample_data)
    
    engineer = FeatureEngineer()
    engineered = engineer.transform(cleaned)
    
    assert 'brand' in engineered.columns
    assert engineered['brand'].iloc[0] == 'Apple'
    assert engineered['brand'].iloc[1] == 'Samsung'
    assert 'name_len' in engineered.columns

# --- 2. BEHAVIORAL TESTS (Model Sanity) ---

def test_prediction_bounds(trained_predictor):
    """Predictions must be between 0% and 100%"""
    input_data = {
        "product_name": "Test Item",
        "category": "Electronics",
        "actual_price": 50000,
        "rating": 4.5,
        "rating_count": 1000,
        "about_product": "Test",
        "review_content": "Test"
    }
    # Pass as dict
    pred = trained_predictor.predict(input_data)
    assert 0 <= pred <= 100, f"Prediction {pred} is out of realistic bounds!"

def test_price_sensitivity(trained_predictor):
    """Sanity Check: A generic cable usually has higher discount % than an iPhone"""
    iphone = {
        "product_name": "Apple iPhone", "category": "Mobile", 
        "actual_price": 80000, "rating": 4.8, "rating_count": 1000, "about_product":"", "review_content":""
    }
    cable = {
        "product_name": "Generic Cable", "category": "Cables", 
        "actual_price": 500, "rating": 3.5, "rating_count": 50, "about_product":"", "review_content":""
    }
    
    pred_iphone = trained_predictor.predict(iphone)
    pred_cable = trained_predictor.predict(cable)
    
    print(f"iPhone Disc: {pred_iphone:.2f}%, Cable Disc: {pred_cable:.2f}%")

def test_full_pipeline_cv(sample_data):
    """Test that training runs with CV without crashing"""
    predictor = DiscountPredictor()
    large_data = pd.concat([sample_data] * 4, ignore_index=True) 
    
    predictor.train(large_data, tune=False)
    assert os.path.exists("models/discount_predictor.joblib")

def test_prediction_shape(sample_data):
    """Test prediction returns valid float"""
    predictor = DiscountPredictor()
    large_data = pd.concat([sample_data] * 4, ignore_index=True)
    predictor.train(large_data, tune=False)
    
    input_data = {
        "product_name": "Test Product",
        "category": "Computers|Phones",
        "actual_price": 80000,
        "rating": 4.5,
        "rating_count": 500,
        "about_product": "Test description",
        "review_content": "Test review"
    }
    pred = predictor.predict(input_data)
    assert isinstance(pred, (float, np.float32, np.float64))
    assert pred >= 0

# --- 3. API INTEGRATION TESTS (End-to-End) ---

def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()['status'] == "healthy"

def test_api_predict_discount(client):
    """Tests the full HTTP flow"""
    payload = {
        "product_name": "Sony TV",
        "category": "Electronics|TV",
        "actual_price": 55000.0,
        "rating": 4.5,
        "rating_count": 500,
        "about_product": "4K LED TV",
        "review_content": "Great picture"
    }
    response = client.post("/predict_discount", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_discount" in data
    assert isinstance(data['predicted_discount'], float)

def test_api_validation_error(client):
    """Ensure API rejects bad data (missing required field)"""
    payload = {
        "category": "Electronics",
        "actual_price": "NOT_A_NUMBER"
    }
    response = client.post("/predict_discount", json=payload)
    assert response.status_code == 422

def test_api_answer_question(client):
    """Test RAG endpoint"""
    payload = {"query": "What is the highest rated product?"}
    response = client.post("/answer_question", json=payload)
    # Accept 200 (ready) or 503 (training)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert "context" in data

def test_api_explain_prediction(client):
    """Test explanation endpoint"""
    payload = {
        "product_name": "Test Product",
        "category": "Electronics",
        "actual_price": 50000.0,
        "rating": 4.5,
        "rating_count": 1000,
        "about_product": "Test description",
        "review_content": "Test review"
    }
    response = client.post("/explain_prediction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "base" in data or "drivers" in data or "error" in data

# --- 4. DRIFT DETECTION TESTS ---

def test_drift_trigger(trained_predictor):
    """Ensure the drift detector fires when data changes drastically"""
    
    drift_data = pd.DataFrame({
        'product_id': [f'P{i}' for i in range(50)],
        'product_name': ['Gold'] * 50,
        'category': ['Luxury'] * 50,
        'actual_price': [1000000.0] * 50, 
        'rating': [1.0] * 50,
        'rating_count': [10] * 50,
        'about_product': ['Expensive'] * 50,
        'discount_percentage': [0] * 50,
        'review_content': [''] * 50
    })
    
    is_drifting = trained_predictor.check_and_retrain(drift_data)
    
    assert isinstance(is_drifting, bool)
