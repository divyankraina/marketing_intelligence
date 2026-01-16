from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.data_loader import load_or_generate_data
from src.ml_engine import DiscountPredictor
from src.rag_engine import MarketingAssistant
from src.monitoring import PerformanceMonitor
from src.fine_tune import train_qlora_model
import os
import torch
import threading
import logging
import gc
import pandas as pd
from typing import List, Dict, Optional

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Production Marketing Intel System")

# Global State Management
class SystemState:
    READY = "READY"
    TRAINING = "TRAINING"
    INITIALIZING = "INITIALIZING"

CURRENT_STATE = SystemState.INITIALIZING

# Initialize engines
predictor = DiscountPredictor()
assistant = MarketingAssistant()

# Data Models
class PredictionInput(BaseModel):
    product_name: str
    category: str
    actual_price: float
    rating: float
    rating_count: int
    about_product: str = "" 
    review_content: str = ""

class QueryInput(BaseModel):
    query: str

class EvaluationInput(BaseModel):
    test_data: List[Dict]  # List of product records with discount_percentage

class RetrainInput(BaseModel):
    new_data: List[Dict]  # List of new product records
    force: bool = False  # Force retraining even without drift

class RAGEvaluationInput(BaseModel):
    test_queries: List[Dict]  # List of dicts with 'query' and 'expected_answer'

class SafetyCheckInput(BaseModel):
    product_name: str
    category: str
    actual_price: float
    rating: float
    rating_count: int
    about_product: str = ""
    review_content: str = ""

def run_initial_training():
    """Runs training, then loads the pipeline"""
    global CURRENT_STATE
    logger.info("⚡ STATE: TRAINING STARTED. Inference is paused.")
    
    try:
        train_qlora_model()
        logger.info("✅ Training Complete. Releasing Memory...")
        
        gc.collect()
        torch.cuda.empty_cache()        
        logger.info("🤖 Loading Inference Pipeline...")
        assistant.load_pipeline()
        
        CURRENT_STATE = SystemState.READY
        logger.info("⚡ STATE: SYSTEM READY.")
    except Exception as e:
        logger.error(f"❌ Training Failed: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        assistant.load_pipeline()
        CURRENT_STATE = SystemState.READY

@app.on_event("startup")
def startup_event():
    global CURRENT_STATE
    
    logger.info("SYSTEM STARTUP CHECK:")
    if torch.cuda.is_available():
        logger.info(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    
    # 0. Load Data
    df = load_or_generate_data()
    
    # 1. Train/Load ML Predictor (Lightweight, CPU/GPU hybrid)
    if not os.path.exists("models/discount_predictor.joblib"):
        logger.info("📊 Training Discount Predictor...")
        predictor.train(df, tune=True)
    else:
        logger.info("📊 Loading Discount Predictor...")
        predictor.load()
        
    # 2. Ingest RAG Vector Data (CPU/GPU hybrid)
    if not os.path.exists("models/faiss_index_gpu"):
        logger.info("📚 Ingesting RAG Data...")
        assistant.ingest_data(df)
    
    # 3. Smart LLM Loading (The Crash Fix)
    if not os.path.exists("models/final_adapter"):
        logger.warning("⚠️ No Adapter Found. Entering TRAINING Mode.")
        CURRENT_STATE = SystemState.TRAINING
        # Start training in background thread to unblock API startup
        t = threading.Thread(target=run_initial_training)
        t.start()
    else:
        logger.info("✅ Adapter Found. Loading Inference Pipeline...")
        assistant.load_pipeline()
        CURRENT_STATE = SystemState.READY

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "state": CURRENT_STATE,
        "gpu_available": torch.cuda.is_available()
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict_discount")
def predict_discount(data: PredictionInput):
    with PerformanceMonitor.track_latency("/predict_discount"):
        try:
            input_dict = data.dict()
            discount = predictor.predict(input_dict)
            PerformanceMonitor.log_prediction(discount)
            return {"predicted_discount": round(float(discount), 2)}
        except Exception as e:
            PerformanceMonitor.log_error("/predict_discount", type(e).__name__)
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain_prediction")
def explain_prediction(data: PredictionInput):
    input_dict = data.dict()
    return predictor.explain(input_dict)

@app.post("/answer_question")
def answer_question(data: QueryInput):
    if CURRENT_STATE == SystemState.TRAINING:
        return JSONResponse(
            status_code=503, 
            content={"error": "System is currently fine-tuning on new data. Please try again in 5 minutes."}
        )
        
    with PerformanceMonitor.track_latency("/answer_question"):
        try:
            response = assistant.answer(data.query)
            return response
        except Exception as e:
            PerformanceMonitor.log_error("/answer_question", type(e).__name__)
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate_model")
def evaluate_model(data: EvaluationInput):
    """Evaluate model performance metrics (RMSE, MAE, R²)"""
    if not predictor.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        test_df = pd.DataFrame(data.test_data)
        if 'discount_percentage' not in test_df.columns:
            raise HTTPException(status_code=400, detail="Test data must include 'discount_percentage' column")
        
        metrics = predictor.evaluate(test_df)
        return metrics
    except Exception as e:
        PerformanceMonitor.log_error("/evaluate_model", type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain_model")
def retrain_model(data: RetrainInput, background_tasks: BackgroundTasks):
    """Automated retraining endpoint with drift detection"""
    global CURRENT_STATE
    
    if CURRENT_STATE == SystemState.TRAINING:
        return JSONResponse(
            status_code=503,
            content={"error": "System is already training. Please wait."}
        )
    
    try:
        new_df = pd.DataFrame(data.new_data)
        
        # Check for drift
        drift_detected = predictor.check_and_retrain(new_df) if not data.force else True
        
        if drift_detected or data.force:
            CURRENT_STATE = SystemState.TRAINING
            
            def retrain_task():
                global CURRENT_STATE
                try:
                    logger.info("🔄 Starting automated retraining...")
                    predictor.train(new_df, tune=False)  # Faster retraining
                    logger.info("✅ Retraining complete")
                    CURRENT_STATE = SystemState.READY
                except Exception as e:
                    logger.error(f"❌ Retraining failed: {e}")
                    CURRENT_STATE = SystemState.READY
            
            background_tasks.add_task(retrain_task)
            PerformanceMonitor.log_retraining()
            
            return {
                "status": "retraining_started",
                "drift_detected": drift_detected,
                "message": "Model retraining initiated in background"
            }
        else:
            return {
                "status": "no_retraining_needed",
                "drift_detected": False,
                "message": "No significant drift detected. Model is up to date."
            }
    except Exception as e:
        PerformanceMonitor.log_error("/retrain_model", type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate_rag")
def evaluate_rag(data: RAGEvaluationInput):
    """Evaluate RAG grounding accuracy and factuality"""
    if CURRENT_STATE == SystemState.TRAINING:
        return JSONResponse(
            status_code=503,
            content={"error": "System is currently fine-tuning. Please try again later."}
        )
    
    try:
        if not data.test_queries:
            raise HTTPException(status_code=400, detail="test_queries cannot be empty")
        
        results = assistant.evaluate_grounding_accuracy(data.test_queries)
        PerformanceMonitor.update_rag_metrics(
            results['grounding_accuracy'],
            results['average_factuality_score']
        )
        return results
    except Exception as e:
        PerformanceMonitor.log_error("/evaluate_rag", type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/safety_check")
def safety_check(data: SafetyCheckInput):
    """Validate input data for safety and compliance"""
    try:
        input_dict = data.dict()
        sanitized = predictor._validate_and_sanitize_input(input_dict)
        
        # Additional safety checks
        issues = []
        warnings = []
        
        # Check price reasonableness
        if sanitized['actual_price'] > 10000000:  # 10M
            warnings.append("Price exceeds typical e-commerce range")
        
        # Check rating bounds
        if sanitized['rating'] < 0 or sanitized['rating'] > 5:
            issues.append(f"Rating {sanitized['rating']} out of valid range [0, 5]")
        
        # Check for potentially harmful content
        harmful_keywords = ['hack', 'exploit', 'malware', 'virus']
        text_fields = [sanitized.get('product_name', ''), sanitized.get('about_product', '')]
        for field in text_fields:
            if any(keyword in field.lower() for keyword in harmful_keywords):
                warnings.append("Potentially suspicious content detected")
        
        is_safe = len(issues) == 0
        PerformanceMonitor.log_safety_check(is_safe)
        
        return {
            "safe": is_safe,
            "sanitized_input": sanitized,
            "issues": issues,
            "warnings": warnings
        }
    except Exception as e:
        PerformanceMonitor.log_error("/safety_check", type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))