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

# --- Optimized Startup Logic ---

def run_initial_training():
    """Runs training, then loads the pipeline"""
    global CURRENT_STATE
    logger.info("⚡ STATE: TRAINING STARTED. Inference is paused.")
    
    try:
        train_qlora_model()
        logger.info("✅ Training Complete. Releasing Memory...")
        
        # --- DOUBLE TAP MEMORY CLEANUP ---
        gc.collect()
        torch.cuda.empty_cache()
        # ---------------------------------
        
        # Load the pipeline only AFTER training frees the GPU
        logger.info("🤖 Loading Inference Pipeline...")
        assistant.load_pipeline()
        
        CURRENT_STATE = SystemState.READY
        logger.info("⚡ STATE: SYSTEM READY.")
    except Exception as e:
        logger.error(f"❌ Training Failed: {e}")
        # Even if training fails, try to load base model
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

# --- Endpoints ---

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
            # Pass the entire Pydantic object as a dict
            input_dict = data.dict()
            discount = predictor.predict(input_dict)
            return {"predicted_discount": round(float(discount), 2)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain_prediction")
def explain_prediction(data: PredictionInput):
    # Pass the entire Pydantic object as a dict
    input_dict = data.dict()
    return predictor.explain(input_dict)

@app.post("/answer_question")
def answer_question(data: QueryInput):
    # RAG depends on the LLM, which might be training
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
            raise HTTPException(status_code=500, detail=str(e))