# 🚀 Production Marketing Intel System

This is a production-grade Machine Learning & RAG system designed for **E-commerce Pricing Intelligence**. It combines a Grandmaster-level Stacking Regressor for price prediction with a Hybrid RAG (Retrieval Augmented Generation) engine for grounded market analysis.

## 🌟 Key Features

### 🧠 1. Advanced ML Engine (Price Prediction)

* **Architecture:** Stacking Ensemble (Level 0: XGBoost, LightGBM, Random Forest → Level 1: ElasticNetCV).
* **Techniques:**
* **Log-Space Target Transformation:** Handles price skew (predicts percentage differences accurately).
* **Strict Feature Selection:** Uses `ElasticNet` to select the top 100 features from BERT embeddings to prevent overfitting.
* **Drift Detection:** KS-Tests (`Kolmogorov-Smirnov`) on incoming batches to detect data drift.
* **SHAP Explainer:** Surrogate model providing "White Box" explanations for predictions.



### 🤖 2. Hybrid RAG Engine (Market Analysis)

* **Smart Router:** Automatically detects intent.
* *Deterministic Queries* (e.g., "Highest rated product") → **Pandas Structured Sort** (100% Accuracy).
* *Semantic Queries* (e.g., "Best gift for kids") → **Vector Search** (FAISS + BGE-Large Embeddings).


* **LLM:** 4-bit Quantized **Mistral-7B-Instruct-v0.3** with LoRA adapters.
* **Ingestion:** Parquet-based structured metadata storage with strict type enforcement.

### ⚙️ 3. MLOps & Architecture

* **State-Based Startup:** "Maintenance Mode" prevents API crashes during model fine-tuning.
* **Background Fine-Tuning:** Automatically fine-tunes the LLM on new data using QLoRA without blocking the main thread.
* **Memory Management:** Explicit Garbage Collection (`gc.collect`) to prevent GPU OOM errors.
* **Monitoring:** Native Prometheus `/metrics` endpoint for latency and drift tracking.

---

## 📂 Project Structure

```bash
.
├── Dockerfile                  # GPU-optimized container definition
├── requirements.txt            # Python dependencies (Torch, Transformers, FAISS)
├── main.py                     # FastAPI Entrypoint & State Management
├── amazon.csv                  # Input Dataset
├── models/                     # Persisted Artifacts (GitIgnored)
│   ├── discount_predictor.joblib
│   ├── faiss_index_gpu/
│   ├── metadata_store.parquet
│   └── final_adapter/
└── src/
    ├── __init__.py
    ├── config.py               # Global Constants (Paths, Model IDs)
    ├── data_loader.py          # Data ingestion & cleaning
    ├── ml_engine.py            # Stacking Regressor, Drift, & SHAP Logic
    ├── rag_engine.py           # Hybrid RAG (Router + FAISS + LLM)
    ├── fine_tune.py            # QLoRA Training Script
    └── monitoring.py           # Prometheus Instrumentation

```

---

## 🛠️ Installation & Setup

### Prerequisites

* **Docker** with NVIDIA Container Toolkit installed.
* **GPU:** Minimum 12GB VRAM recommended (RTX 3060/4070 or better).

### 1. Build the Container

```bash
docker-compose up --build

```

* (Note: On the first run, the system will enter **Training Mode** to fine-tune the adapter. The API will return 503 until training completes (~10-15 mins).*

---

## 📡 API Usage

### 1. Predict Discount (ML Engine)

Predicts the optimal discount percentage for a product.

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

**Response:** `{"predicted_discount": 18.45}`

### 2. Explain Prediction (SHAP)

Returns the top factors driving that specific prediction.

```bash
curl -X POST "http://localhost:8000/explain_prediction" \
     -H "Content-Type: application/json" \
     -d '{ ... same body as above ... }'

```

**Response:**

```json
{
  "base": "34.00",
  "drivers": {
    "Category: Electronics": "-5.20 impact",
    "actual_price": "+2.10 impact"
  }
}

```

### 3. Market Analysis (Hybrid RAG)

Ask questions about your catalog.

```bash
curl -X POST "http://localhost:8000/answer_question" \
     -H "Content-Type: application/json" \
     -d '{ "query": "Which product has the highest rating count?" }'

```

**Response:**

```json
{
  "answer": "The product with the highest rating count is Camel Oil Pastels (9427 ratings).",
  "context": ["Name: Camel Oil Pastels | Rating Count: 9427.0 ..."]
}

```

### 4. Monitoring

```bash
curl http://localhost:8000/metrics

```

Visualizes latency and drift stats (Import to Grafana).

---

## 🧠 Technical Deep Dive

### The "Hybrid Router" Logic

To solve the common RAG failure of "hallucinating numbers," the `rag_engine.py` uses a **Deterministic Router**:

1. **Input:** *"Show me the cheapest phone"*
2. **Router:** Detects keyword `cheapest`.
3. **Action:** Bypasses Vector DB. Executes `df.sort_values('discounted_price', ascending=True)`.
4. **Result:** Fetches the actual mathematical minimum, not a "semantically similar" text chunk.

### The Stacking Pipeline

We utilize a **3-Layer Stacking Architecture** to maximize generalization:

1. **Feature Selection:** `SelectFromModel` drops noise from 700+ embedding dimensions.
2. **Base Learners:**
* `XGBoost`: Captures complex interactions.
* `LightGBM`: Handles categorical features swiftly (Leaf-wise growth).
* `RandomForest`: Reduces variance (Bagging).


3. **Meta Learner:** `ElasticNetCV` learns the optimal weighted average of the base learners.

---

## ⚠️ Troubleshooting

**1. `CUDA out of memory**`

* **Cause:** Training and Inference running simultaneously.
* **Fix:** The system uses `gc.collect()` and `torch.cuda.empty_cache()` aggressively. If it persists, enable `llm_int8_enable_fp32_cpu_offload=True` in `rag_engine.py`.

**2. `ModuleNotFoundError: faiss.swigfaiss_avx2**`

* **Cause:** Broken pip install on Windows/Linux.
* **Fix:** Inside Docker, use `pip install faiss-cpu` (CPU version covers index operations, GPU is used for Embeddings/LLM).

**3. "System is currently fine-tuning" (503 Error)**

* **Cause:** Initial startup detects missing adapters.
* **Fix:** Wait ~10 minutes for the background thread to finish QLoRA training. Check logs with `docker logs <container_id>`.