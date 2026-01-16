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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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
    """Predict optimal discount percentage for a product"""
    with PerformanceMonitor.track_latency("/predict_discount"):
        try:
            if not predictor.model:
                raise HTTPException(status_code=503, detail="Model not loaded. Please wait for initialization.")
            
            input_dict = data.dict()
            discount = predictor.predict(input_dict)
            PerformanceMonitor.log_prediction(discount)
            return {"predicted_discount": round(float(discount), 2)}
        except HTTPException:
            raise
        except ValueError as e:
            PerformanceMonitor.log_error("/predict_discount", "ValueError")
            raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
        except Exception as e:
            PerformanceMonitor.log_error("/predict_discount", type(e).__name__)
            logger.error(f"Prediction error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error during prediction")

@app.post("/explain_prediction")
def explain_prediction(data: PredictionInput):
    """Explain prediction factors using SHAP"""
    try:
        if not predictor.model:
            raise HTTPException(status_code=503, detail="Model not loaded. Please wait for initialization.")
        
        input_dict = data.dict()
        explanation = predictor.explain(input_dict)
        return explanation
    except HTTPException:
        raise
    except Exception as e:
        PerformanceMonitor.log_error("/explain_prediction", type(e).__name__)
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during explanation")

@app.post("/answer_question")
def answer_question(data: QueryInput):
    """Answer questions using RAG-powered LLM assistant"""
    if CURRENT_STATE == SystemState.TRAINING:
        return JSONResponse(
            status_code=503, 
            content={"error": "System is currently fine-tuning on new data. Please try again in 5 minutes."}
        )
    
    if not data.query or not data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    with PerformanceMonitor.track_latency("/answer_question"):
        try:
            if not hasattr(assistant, 'pipe') or assistant.pipe is None:
                raise HTTPException(status_code=503, detail="LLM pipeline not loaded. Please wait for initialization.")
            
            response = assistant.answer(data.query)
            return response
        except HTTPException:
            raise
        except Exception as e:
            PerformanceMonitor.log_error("/answer_question", type(e).__name__)
            logger.error(f"RAG error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error during question answering")

@app.post("/evaluate_model")
def evaluate_model(data: EvaluationInput):
    """Evaluate model performance metrics (RMSE, MAE, R²)"""
    if not predictor.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        if not data.test_data or len(data.test_data) == 0:
            raise HTTPException(status_code=400, detail="Test data cannot be empty")
        
        test_df = pd.DataFrame(data.test_data)
        if 'discount_percentage' not in test_df.columns:
            raise HTTPException(status_code=400, detail="Test data must include 'discount_percentage' column")
        
        if len(test_df) < 1:
            raise HTTPException(status_code=400, detail="Test data must contain at least one sample")
        
        metrics = predictor.evaluate(test_df)
        return metrics
    except HTTPException:
        raise
    except ValueError as e:
        PerformanceMonitor.log_error("/evaluate_model", "ValueError")
        raise HTTPException(status_code=400, detail=f"Invalid test data: {str(e)}")
    except Exception as e:
        PerformanceMonitor.log_error("/evaluate_model", type(e).__name__)
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during model evaluation")

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
        if not data.new_data or len(data.new_data) == 0:
            raise HTTPException(status_code=400, detail="new_data cannot be empty")
        
        # Validate data structure
        try:
            new_df = pd.DataFrame(data.new_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
        
        # Check required columns
        required_cols = ['product_name', 'category', 'actual_price', 'rating', 'rating_count']
        missing_cols = [col for col in required_cols if col not in new_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_cols)}"
            )
        
        if len(new_df) < 10:
            raise HTTPException(
                status_code=400,
                detail="At least 10 samples required for retraining"
            )
        
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
                    logger.error(f"❌ Retraining failed: {e}", exc_info=True)
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
    except HTTPException:
        raise
    except Exception as e:
        PerformanceMonitor.log_error("/retrain_model", type(e).__name__)
        logger.error(f"Retrain error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during retraining")

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
        
        # Additional safety checks before sanitization
        issues = []
        warnings = []
        
        # Check for SQL injection patterns
        sql_patterns = [';', '--', '/*', '*/', 'xp_', 'sp_', 'exec', 'union', 'select']
        text_fields = [input_dict.get('product_name', ''), input_dict.get('about_product', ''), input_dict.get('category', '')]
        for field in text_fields:
            field_lower = str(field).lower()
            if any(pattern in field_lower for pattern in sql_patterns):
                warnings.append("Potential SQL injection pattern detected in text fields")
        
        # Check for XSS patterns
        xss_patterns = ['<script', '</script>', 'javascript:', 'onerror=', 'onclick=']
        for field in text_fields:
            field_lower = str(field).lower()
            if any(pattern in field_lower for pattern in xss_patterns):
                warnings.append("Potential XSS pattern detected in text fields")
        
        # Check for potentially harmful content
        harmful_keywords = ['hack', 'exploit', 'malware', 'virus', 'trojan', 'phishing']
        for field in text_fields:
            if any(keyword in str(field).lower() for keyword in harmful_keywords):
                warnings.append("Potentially suspicious content detected")
        
        # Validate and sanitize
        try:
            sanitized = predictor._validate_and_sanitize_input(input_dict)
        except ValueError as e:
            issues.append(f"Validation error: {str(e)}")
            is_safe = False
            PerformanceMonitor.log_safety_check(is_safe)
            return {
                "safe": is_safe,
                "sanitized_input": input_dict,
                "issues": issues,
                "warnings": warnings
            }
        
        # Check price reasonableness
        if sanitized['actual_price'] > 10000000:  # 10M
            warnings.append("Price exceeds typical e-commerce range (>10M)")
        elif sanitized['actual_price'] < 0:
            issues.append("Price cannot be negative")
        
        # Check rating bounds (should be handled by validation, but double-check)
        if sanitized['rating'] < 0 or sanitized['rating'] > 5:
            issues.append(f"Rating {sanitized['rating']} out of valid range [0, 5]")
        
        # Check rating_count reasonableness
        if sanitized['rating_count'] > 100000000:  # 100M
            warnings.append("Rating count seems unusually high")
        
        # Check text length
        for field_name in ['product_name', 'about_product', 'category']:
            if len(str(sanitized.get(field_name, ''))) > 1000:
                warnings.append(f"{field_name} exceeds recommended length (1000 chars)")
        
        is_safe = len(issues) == 0
        PerformanceMonitor.log_safety_check(is_safe)
        
        return {
            "safe": is_safe,
            "sanitized_input": sanitized,
            "issues": issues,
            "warnings": warnings
        }
    except HTTPException:
        raise
    except Exception as e:
        PerformanceMonitor.log_error("/safety_check", type(e).__name__)
        logger.error(f"Safety check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during safety check")