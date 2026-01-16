# 🚀 Production Marketing Intelligence System

A production-grade Machine Learning & RAG system designed for **E-commerce Marketing Intelligence**. This system combines advanced ML models for predicting business outcomes (discounts, conversions) with an LLM-powered assistant for answering customer queries using RAG (Retrieval-Augmented Generation).

## 📋 Table of Contents

- [Features](#-key-features)
- [Architecture](#-system-architecture)
- [Installation](#-installation--setup)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Testing](#-testing)
- [Performance Evaluation](#-performance-evaluation)
- [Technical Details](#-technical-deep-dive)
- [Deployment](#-production-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 🌟 Key Features

### 🧠 1. Advanced ML Engine (Discount Prediction)

* **Architecture:** Stacking Ensemble (Level 0: XGBoost, LightGBM, Random Forest → Level 1: ElasticNetCV)
* **Techniques:**
  * **Log-Space Target Transformation:** Handles price skew (predicts percentage differences accurately)
  * **Strict Feature Selection:** Uses `ElasticNet` to select top 100 features from BERT embeddings to prevent overfitting
  * **Drift Detection:** KS-Tests (`Kolmogorov-Smirnov`) on incoming batches to detect data drift
  * **SHAP Explainer:** Surrogate model providing "White Box" explanations for predictions
  * **Safety Validation:** Input sanitization, bounds checking, and prediction clamping
* **Evaluation Metrics:** RMSE, MAE, R² score tracking
* **Automated Retraining:** Background retraining triggered by drift detection

### 🤖 2. Hybrid RAG Engine (Market Analysis)

* **Smart Router:** Automatically detects intent
  * *Deterministic Queries* (e.g., "Highest rated product") → **Pandas Structured Sort** (100% Accuracy)
  * *Semantic Queries* (e.g., "Best gift for kids") → **Vector Search** (FAISS + mxbai-embed-large-v1 Embeddings)
* **LLM:** 4-bit Quantized **Mistral-7B-Instruct-v0.3** with LoRA adapters
* **Ingestion:** Parquet-based structured metadata storage with strict type enforcement
* **Grounding Accuracy:** Evaluation metrics for RAG factuality and grounding
* **Fine-Tuning:** QLoRA-based domain adaptation on marketing data

### ⚙️ 3. MLOps & Production Features

* **State-Based Startup:** "Maintenance Mode" prevents API crashes during model fine-tuning
* **Background Fine-Tuning:** Automatically fine-tunes the LLM on new data using QLoRA without blocking the main thread
* **Memory Management:** Explicit Garbage Collection (`gc.collect`) to prevent GPU OOM errors
* **Monitoring:** Native Prometheus `/metrics` endpoint for latency, drift, and accuracy tracking
* **Explainability:** SHAP-based explanations for all predictions
* **Observability:** Comprehensive metrics for model performance, RAG accuracy, and system health
* **Safety:** Input validation, sanitization, and bounds checking

### 🧪 4. Testing & Validation

* **Unit Tests:** Comprehensive test coverage for data processing, feature engineering, and model logic
* **Integration Tests:** End-to-end API testing for all endpoints
* **Load Testing:** Concurrent request testing to validate performance under load
* **Safety Validation:** Input sanitization and bounds checking

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   ML Engine  │  │  RAG Engine  │  │  Monitoring  │      │
│  │              │  │              │  │              │      │
│  │ - Stacking   │  │ - Hybrid     │  │ - Prometheus │      │
│  │   Ensemble   │  │   Router     │  │ - Metrics    │      │
│  │ - Drift      │  │ - FAISS      │  │ - Logging    │      │
│  │   Detection  │  │ - LLM        │  │              │      │
│  │ - SHAP       │  │ - Fine-tune  │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Models/    │    │  Vector DB   │    │   Metrics    │
│  Artifacts   │    │   (FAISS)    │    │  Endpoint    │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## 📂 Project Structure

```bash
.
├── Dockerfile.gpu              # GPU-optimized container definition
├── docker-compose.yml          # Docker Compose configuration
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata
├── main.py                     # FastAPI Entrypoint & State Management
├── data/
│   └── amazon.csv              # Input Dataset (Amazon Sales Dataset)
├── models/                      # Persisted Artifacts (GitIgnored)
│   ├── discount_predictor.joblib
│   ├── drift_stats.joblib
│   ├── surrogate_explainer.joblib
│   ├── faiss_index_gpu/
│   ├── metadata_store.parquet
│   └── final_adapter/
├── tests/
│   ├── test_main.py            # Unit & Integration Tests
│   ├── load_test.py            # Load Testing Script
│   └── integration_test.py     # Quick Integration Test Script
├── src/
│   ├── __init__.py
│   ├── config.py               # Global Constants (Paths, Model IDs)
│   ├── data_loader.py          # Data ingestion & cleaning
│   ├── ml_engine.py            # Stacking Regressor, Drift, & SHAP Logic
│   ├── rag_engine.py           # Hybrid RAG (Router + FAISS + LLM)
│   ├── fine_tune.py            # QLoRA Training Script
│   └── monitoring.py           # Prometheus Instrumentation
├── README.md                   # This file
└── CHANGELOG.md                # Change log
```

---

## 🛠️ Installation & Setup

### Prerequisites

* **Docker** with NVIDIA Container Toolkit installed
* **GPU:** Minimum 12GB VRAM recommended (RTX 3060/4070 or better)
* **Python 3.10+** (for local development)
* **CUDA 12.1+** (for GPU support)

### Dataset Requirements

Place your Amazon Sales Dataset (or similar e-commerce dataset) in `data/amazon.csv`. The dataset should contain the following columns:

**Required Columns:**
- `product_id` - Unique product identifier
- `product_name` - Product name/title
- `category` - Product category (pipe-separated, e.g., "Electronics|Mobiles")
- `actual_price` - Original product price
- `discounted_price` - Discounted price
- `discount_percentage` - Discount percentage (target variable)
- `rating` - Product rating (0-5)
- `rating_count` - Number of ratings
- `about_product` - Product description
- `review_content` - Review text (optional)

**Example Dataset:**
```csv
product_id,product_name,category,actual_price,discounted_price,discount_percentage,rating,rating_count,about_product
P001,iPhone 13,Electronics|Mobiles,70000,63000,10%,4.8,5000,Latest iPhone with A15 chip
P002,Samsung TV,Electronics|TVs,50000,40000,20%,4.5,2000,4K Smart TV
```

### 1. Build and Run with Docker

```bash
# Clone or navigate to the project directory
cd marketing_intelligence

# Build and start the container
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

**Note:** On the first run, the system will enter **Training Mode** to fine-tune the adapter. The API will return 503 until training completes (~10-15 mins).

### 2. Verify Installation

```bash
# Check health
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "state": "READY",
#   "gpu_available": true
# }
```

### 3. Local Development (Optional)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 🚀 Quick Start

### 1. Start the System

```bash
docker-compose up -d
```

### 2. Wait for Training (First Run Only)

```bash
# Check status
curl http://localhost:8000/health

# Monitor logs
docker-compose logs -f
```

### 3. Make Your First Prediction

```bash
curl -X POST "http://localhost:8000/predict_discount" \
     -H "Content-Type: application/json" \
     -d '{
           "product_name": "Samsung Galaxy S23 Ultra",
           "category": "Electronics|Mobiles",
           "actual_price": 124999,
           "rating": 4.6,
           "rating_count": 5000,
           "about_product": "Flagship phone with 200MP camera"
         }'
```

### 4. Ask a Question

```bash
curl -X POST "http://localhost:8000/answer_question" \
     -H "Content-Type: application/json" \
     -d '{ "query": "Which product has the highest rating?" }'
```

---

## 📡 API Documentation

### Core Endpoints

#### 1. Health Check

**GET** `/health`

Check system health and status.

**Response:**
```json
{
  "status": "healthy",
  "state": "READY",
  "gpu_available": true
}
```

**States:**
- `READY` - System is operational
- `TRAINING` - Model is being fine-tuned (API returns 503)
- `INITIALIZING` - System is starting up

---

#### 2. Predict Discount

**POST** `/predict_discount`

Predicts the optimal discount percentage for a product.

**Request Body:**
```json
{
  "product_name": "Samsung Galaxy S23 Ultra",
  "category": "Electronics|Mobiles",
  "actual_price": 124999.0,
  "rating": 4.6,
  "rating_count": 5000,
  "about_product": "Flagship phone with 200MP camera",
  "review_content": "Great phone with excellent camera"
}
```

**Response:**
```json
{
  "predicted_discount": 18.45
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/predict_discount" \
     -H "Content-Type: application/json" \
     -d '{
           "product_name": "Samsung Galaxy S23 Ultra",
           "category": "Electronics|Mobiles",
           "actual_price": 124999,
           "rating": 4.6,
           "rating_count": 5000,
           "about_product": "Flagship phone with 200MP camera"
         }'
```

---

#### 3. Explain Prediction

**POST** `/explain_prediction`

Returns the top factors driving a specific prediction using SHAP.

**Request Body:** (Same as `/predict_discount`)

**Response:**
```json
{
  "base": "34.00",
  "drivers": {
    "Category: Electronics": "-5.20 impact",
    "actual_price": "+2.10 impact",
    "rating": "+1.50 impact",
    "rating_count": "+0.80 impact"
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/explain_prediction" \
     -H "Content-Type: application/json" \
     -d '{
           "product_name": "Samsung Galaxy S23 Ultra",
           "category": "Electronics|Mobiles",
           "actual_price": 124999,
           "rating": 4.6,
           "rating_count": 5000,
           "about_product": "Flagship phone with 200MP camera"
         }'
```

---

#### 4. Answer Question (RAG)

**POST** `/answer_question`

Ask questions about your product catalog using RAG.

**Request Body:**
```json
{
  "query": "Which product has the highest rating count?"
}
```

**Response:**
```json
{
  "answer": "The product with the highest rating count is Camel Oil Pastels (9427 ratings).",
  "context": [
    "Name: Camel Oil Pastels | Rating Count: 9427.0 | Rating: 4.5 ..."
  ]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/answer_question" \
     -H "Content-Type: application/json" \
     -d '{ "query": "What is the cheapest product in Electronics category?" }'
```

**Note:** Returns 503 if system is training.

---

### Advanced Endpoints

#### 5. Evaluate Model Performance

**POST** `/evaluate_model`

Get regression metrics (RMSE, MAE, R²) on test data.

**Request Body:**
```json
{
  "test_data": [
    {
      "product_name": "Test Product",
      "category": "Electronics",
      "actual_price": 50000.0,
      "rating": 4.5,
      "rating_count": 1000,
      "about_product": "Test description",
      "review_content": "Test review",
      "discount_percentage": 15.5
    }
  ]
}
```

**Response:**
```json
{
  "rmse": 3.45,
  "mae": 2.12,
  "r2": 0.87,
  "n_samples": 100
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/evaluate_model" \
     -H "Content-Type: application/json" \
     -d '{
           "test_data": [
             {
               "product_name": "Test Product",
               "category": "Electronics",
               "actual_price": 50000.0,
               "rating": 4.5,
               "rating_count": 1000,
               "about_product": "Test",
               "review_content": "Test",
               "discount_percentage": 15.5
             }
           ]
         }'
```

---

#### 6. Automated Retraining

**POST** `/retrain_model`

Trigger model retraining with drift detection.

**Request Body:**
```json
{
  "new_data": [
    {
      "product_name": "New Product",
      "category": "Electronics",
      "actual_price": 60000.0,
      "rating": 4.7,
      "rating_count": 1500,
      "about_product": "New product description",
      "discount_percentage": 12.0
    }
  ],
  "force": false
}
```

**Parameters:**
- `new_data` - Array of new product records
- `force` - If `true`, retrain even without drift detection

**Response:**
```json
{
  "status": "retraining_started",
  "drift_detected": true,
  "message": "Model retraining initiated in background"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/retrain_model" \
     -H "Content-Type: application/json" \
     -d '{
           "new_data": [...],
           "force": false
         }'
```

---

#### 7. Evaluate RAG Grounding Accuracy

**POST** `/evaluate_rag`

Evaluate RAG factuality and grounding accuracy.

**Request Body:**
```json
{
  "test_queries": [
    {
      "query": "What is the highest rated product?",
      "expected_answer": "Product X with rating 4.8"
    }
  ]
}
```

**Response:**
```json
{
  "grounding_accuracy": 0.92,
  "average_factuality_score": 0.88,
  "total_queries": 10,
  "correct_answers": 9
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/evaluate_rag" \
     -H "Content-Type: application/json" \
     -d '{
           "test_queries": [
             {
               "query": "What is the highest rated product?",
               "expected_answer": "Product X with rating 4.8"
             }
           ]
         }'
```

---

#### 8. Safety Check

**POST** `/safety_check`

Validate input data for safety and compliance.

**Request Body:**
```json
{
  "product_name": "Test Product",
  "category": "Electronics",
  "actual_price": 50000.0,
  "rating": 4.5,
  "rating_count": 1000,
  "about_product": "Test description"
}
```

**Response:**
```json
{
  "safe": true,
  "sanitized_input": {
    "product_name": "Test Product",
    "category": "Electronics",
    "actual_price": 50000.0,
    "rating": 4.5,
    "rating_count": 1000,
    "about_product": "Test description"
  },
  "issues": [],
  "warnings": []
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/safety_check" \
     -H "Content-Type: application/json" \
     -d '{
           "product_name": "Test Product",
           "category": "Electronics",
           "actual_price": 50000.0,
           "rating": 4.5,
           "rating_count": 1000
         }'
```

---

#### 9. Monitoring Metrics

**GET** `/metrics`

Prometheus-compatible metrics endpoint.

**Response:** (Prometheus format)
```
# HELP app_request_count Total request count
# TYPE app_request_count counter
app_request_count{endpoint="/predict_discount",method="POST"} 150.0

# HELP app_request_latency_seconds Request latency
# TYPE app_request_latency_seconds histogram
app_request_latency_seconds_bucket{endpoint="/predict_discount",le="0.5"} 120.0
...
```

**Example:**
```bash
curl http://localhost:8000/metrics
```

---

## 🧪 Testing

### Unit & Integration Tests

```bash
# Run all tests
pytest tests/test_main.py -v

# Run with coverage
pytest tests/test_main.py --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_main.py::test_cleaner_logic -v
pytest tests/test_main.py::test_api_predict_discount -v
pytest tests/test_main.py::test_drift_trigger -v
```

### Integration Testing

```bash
# Ensure API is running
docker-compose up -d

# Run integration tests
python tests/integration_test.py
```

### Load Testing

```bash
# Ensure API is running
docker-compose up -d

# Run load tests
python tests/load_test.py
```

The load test will:
- Test prediction endpoint with 50 concurrent requests
- Test RAG endpoint with 20 concurrent requests
- Report success rates, latency metrics (mean, median, P95)
- Validate performance thresholds

**Expected Results:**
- Prediction endpoint: >95% success rate, <5s P95 latency
- RAG endpoint: >80% success rate, <30s P95 latency

---

## 📊 Performance Evaluation

### Model Metrics

The system tracks and reports:
- **RMSE (Root Mean Squared Error):** Average prediction error
- **MAE (Mean Absolute Error):** Average absolute prediction error
- **R² Score:** Coefficient of determination (model fit quality)

**Typical Performance:**
- RMSE: 3-5% (on discount percentage)
- MAE: 2-3% (on discount percentage)
- R²: 0.85-0.92 (depending on dataset quality)

### RAG Metrics

- **Grounding Accuracy:** Percentage of answers correctly grounded in context (target: >90%)
- **Factuality Score:** Average factuality score (0-1) based on context alignment (target: >0.85)

### Monitoring

All metrics are exposed via Prometheus and can be visualized in Grafana:
- Request latency (histogram)
- Model accuracy (gauge)
- Drift events (counter)
- RAG accuracy (gauge)
- Error counts (counter)
- Safety check results (counter)

**Example Grafana Dashboard:**
```yaml
# Import these metrics into Grafana:
- app_request_latency_seconds
- model_last_r2_score
- rag_grounding_accuracy
- model_drift_events
- app_error_count
```

---

## 🧠 Technical Deep Dive

### The "Hybrid Router" Logic

To solve the common RAG failure of "hallucinating numbers," the `rag_engine.py` uses a **Deterministic Router**:

1. **Input:** *"Show me the cheapest phone"*
2. **Router:** Detects keyword `cheapest`
3. **Action:** Bypasses Vector DB. Executes `df.sort_values('discounted_price', ascending=True)`
4. **Result:** Fetches the actual mathematical minimum, not a "semantically similar" text chunk

**Supported Deterministic Queries:**
- "highest price", "most expensive" → Sort by price (descending)
- "lowest price", "cheapest" → Sort by price (ascending)
- "highest rating", "best rated" → Sort by rating (descending)
- "most reviews", "highest rating count" → Sort by rating_count (descending)

### The Stacking Pipeline

We utilize a **3-Layer Stacking Architecture** to maximize generalization:

1. **Feature Selection:** `SelectFromModel` drops noise from 700+ embedding dimensions
2. **Base Learners:**
   * `XGBoost`: Captures complex interactions
   * `LightGBM`: Handles categorical features swiftly (Leaf-wise growth)
   * `RandomForest`: Reduces variance (Bagging)
3. **Meta Learner:** `ElasticNetCV` learns the optimal weighted average of the base learners

**Feature Engineering:**
- Text embeddings (BERT-based)
- Category encoding (Target encoding)
- Polynomial features
- K-means clustering features
- TF-IDF features

### Drift Detection

The system uses Kolmogorov-Smirnov (KS) tests to detect distribution shifts in:
- `actual_price` distribution
- `rating` distribution

When drift is detected (p-value < 0.01), automatic retraining is triggered.

**Drift Detection Process:**
1. Collect reference statistics during initial training
2. Compare new data distributions using KS-test
3. If drift detected → Trigger background retraining
4. Update reference statistics after retraining

### Safety Validation

All inputs are validated and sanitized:
- Numeric bounds checking (prices ≥ 0, ratings 0-5)
- Text length limits (max 1000 chars per field)
- Harmful content detection
- Type coercion and cleaning
- Prediction clamping (0-100% for discounts)

### LLM Fine-Tuning

The system uses **QLoRA (Quantized LoRA)** for efficient fine-tuning:
- 4-bit quantization (NF4)
- LoRA adapters (rank=16, alpha=32)
- Gradient checkpointing for memory efficiency
- Domain-specific training on marketing data

**Fine-Tuning Process:**
1. Generate instruction-following dataset from product data
2. Train LoRA adapters on Mistral-7B base model
3. Save adapters separately (smaller than full model)
4. Load adapters during inference

---

## 🚀 Production Deployment

### Recommended Setup

1. **Use GPU-enabled instances** (AWS p3, GCP n1-standard with GPU, Azure NC-series)
2. **Set up monitoring** with Prometheus + Grafana
3. **Enable logging** to external service (CloudWatch, Stackdriver, etc.)
4. **Add rate limiting** using nginx or API Gateway
5. **Set up health checks** for container orchestration (Kubernetes, ECS, etc.)

### Environment Variables

```bash
# Optional: Override default paths
export HF_HOME="/app/hf_cache"
export SENTENCE_TRANSFORMERS_HOME="/app/hf_cache"
export BNB_CUDA_VERSION=121
```

### Docker Compose Production

```yaml
# Example production docker-compose.yml
version: '3.8'
services:
  marketing-api:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: marketing-intelligence
spec:
  replicas: 2
  selector:
    matchLabels:
      app: marketing-intelligence
  template:
    metadata:
      labels:
        app: marketing-intelligence
    spec:
      containers:
      - name: api
        image: marketing-prod-v1
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
```

---

## ⚠️ Troubleshooting

### 1. `CUDA out of memory`

**Cause:** Training and Inference running simultaneously

**Fix:** 
- The system uses `gc.collect()` and `torch.cuda.empty_cache()` aggressively
- If it persists, reduce batch size in `fine_tune.py`
- Consider using CPU offloading: `llm_int8_enable_fp32_cpu_offload=True` in `rag_engine.py`

### 2. `ModuleNotFoundError: faiss.swigfaiss_avx2`

**Cause:** Broken pip install on Windows/Linux

**Fix:** Inside Docker, `faiss-cpu` is used (CPU version covers index operations, GPU is used for Embeddings/LLM)

### 3. "System is currently fine-tuning" (503 Error)

**Cause:** Initial startup detects missing adapters

**Fix:** 
- Wait ~10 minutes for the background thread to finish QLoRA training
- Check logs with `docker logs <container_id>`
- Monitor status: `curl http://localhost:8000/health`

### 4. Model not loading

**Cause:** Missing model files

**Fix:** 
- Ensure `data/amazon.csv` exists
- Run training manually:
  ```python
  from src.data_loader import load_or_generate_data
  from src.ml_engine import DiscountPredictor
  p = DiscountPredictor()
  p.train(load_or_generate_data())
  ```

### 5. High latency on RAG endpoint

**Cause:** LLM inference is computationally expensive

**Fix:** 
- Ensure GPU is available and properly configured
- Consider reducing `max_new_tokens` in `rag_engine.py`
- Use deterministic routing for simple queries
- Enable model quantization

### 6. Drift detection not working

**Cause:** Insufficient data or incorrect data format

**Fix:**
- Ensure new data has same structure as training data
- Check that `actual_price` and `rating` columns exist
- Verify data types are correct

### 7. Poor prediction accuracy

**Cause:** Insufficient or low-quality training data

**Fix:**
- Ensure dataset has at least 1000 samples
- Check data quality (no missing values in key columns)
- Verify `discount_percentage` is correctly calculated
- Retrain with more data: `POST /retrain_model`

---

## 🔒 Security & Safety

- **Input Validation:** All inputs are sanitized and validated
- **Bounds Checking:** Predictions are clamped to reasonable ranges (0-100%)
- **Error Handling:** Comprehensive error handling prevents information leakage
- **Rate Limiting:** Consider adding rate limiting in production (not included by default)
- **Authentication:** Add API keys or OAuth for production use

---

## 📝 License

This project is provided as-is for educational and commercial use.

---

## 🤝 Contributing

Contributions are welcome! Please ensure:
- All tests pass (`pytest tests/`)
- Code follows PEP 8 style guidelines
- New features include appropriate tests
- Documentation is updated

---

## 📧 Support

For issues and questions:
1. Check the Troubleshooting section
2. Review logs: `docker-compose logs`
3. Check metrics: `curl http://localhost:8000/metrics`
4. Review CHANGELOG.md for recent changes

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Metrics](https://prometheus.io/docs/concepts/metric_types/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

**Built with ❤️ for Production ML Systems**
