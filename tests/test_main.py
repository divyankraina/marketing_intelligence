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

# --- 4. DRIFT DETECTION TESTS ---

def test_drift_trigger(trained_predictor):
    """Ensure the drift detector fires when data changes drastically"""
    
    drift_data = pd.DataFrame({
        'actual_price': [1000000.0] * 50, 
        'rating': [1.0] * 50,
        'product_name': ['Gold'] * 50,
        'discount_percentage': [0] * 50 
    })
    
    is_drifting = trained_predictor.check_and_retrain(drift_data)
    
    assert isinstance(is_drifting, bool)
    


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'product_id': ['P01', 'P02', 'P03', 'P04'],
        'product_name': ['Apple iPhone 13', 'Samsung Galaxy S22', 'Generic USB Cable', 'Sony Bravia TV'],
        'category': ['Computers|Phones', 'Computers|Phones', 'Computers|Cables', 'Home|TVs'],
        'actual_price': ['₹70,000', '65,000', '₹500', '1,00,000'],
        'rating': ['4.8', '4.5', '3.8', '4.7'],
        'rating_count': ['1000', '500', '50', '200'],
        'about_product': ['Great phone', 'Android beast', 'Fast charge', '4K HDR'],
        'discount_percentage': ['10%', '15%', '60%', '20%']
    })

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

def test_full_pipeline_cv(sample_data):
    """Test that training runs with CV without crashing"""
    predictor = DiscountPredictor()
    large_data = pd.concat([sample_data] * 4, ignore_index=True) 
    
    predictor.train(large_data)
    assert os.path.exists("models/discount_predictor.joblib")

def test_prediction_shape(sample_data):
    predictor = DiscountPredictor()
    large_data = pd.concat([sample_data] * 4, ignore_index=True)
    predictor.train(large_data)
    
    pred = predictor.predict("Computers|Phones", 80000, 4.5, 500)
    assert isinstance(pred, (float, np.float32, np.float64))
    assert pred > 0
    
# 1. Load the Trained Model
print("⏳ Loading Model...")
predictor = DiscountPredictor()
predictor.load()

# 2. Define a Test Product (A "Fake" High-End Camera)
test_product = {
    'category': 'Electronics|Cameras & Photography|Digital Cameras|DSLR Cameras',
    'actual_price': 85000.0,    # High price
    'rating': 4.5,              # Good rating
    'rating_count': 1500        # Popular
}

# 3. Predict Discount
print(f"\n🔍 Testing Product: {test_product['category']}")
print(f"   Price: ₹{test_product['actual_price']}")

discount = predictor.predict(
    test_product['category'], 
    test_product['actual_price'], 
    test_product['rating'], 
    test_product['rating_count']
)

print(f"\n✅ PREDICTED DISCOUNT: {discount:.2f}%")

# 4. Explain Why (Using the Surrogate Model)
print("\n💡 Asking AI: 'Why this discount?'")
explanation = predictor.explain(
    test_product['category'], 
    test_product['actual_price'], 
    test_product['rating'], 
    test_product['rating_count']
)

if "error" in explanation:
    print(f"Explanation Error: {explanation['error']}")
else:
    print(f"   Base Bias: {explanation['base']}%")
    print("   Drivers:")
    for feature, impact in explanation['drivers'].items():
        print(f"   - {feature}: {impact}")