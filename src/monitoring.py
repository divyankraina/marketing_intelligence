from prometheus_client import Counter, Histogram, Gauge, Summary
import time

# Metrics Definitions
REQUEST_COUNT = Counter("app_request_count", "Total request count", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency", ["endpoint"])
DRIFT_EVENTS = Counter("model_drift_events", "Number of times data drift was detected")
MODEL_ACCURACY = Gauge("model_last_r2_score", "R2 Score of the latest model training")
PREDICTION_VALUE = Histogram("predicted_discount_values", "Distribution of predicted discounts")
RAG_ACCURACY = Gauge("rag_grounding_accuracy", "RAG grounding accuracy score")
RAG_FACTUALITY = Gauge("rag_factuality_score", "RAG factuality score")
RETRAINING_EVENTS = Counter("model_retraining_events", "Number of model retraining events")
ERROR_COUNT = Counter("app_error_count", "Total error count", ["endpoint", "error_type"])
SAFETY_CHECKS = Counter("safety_check_count", "Number of safety validation checks", ["status"])

class PerformanceMonitor:
    @staticmethod
    def track_latency(endpoint: str):
        """Context manager to track latency"""
        class LatencyTracker:
            def __init__(self, endpoint):
                self.endpoint = endpoint
                self.start_time = 0
            
            def __enter__(self):
                self.start_time = time.time()
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                REQUEST_LATENCY.labels(endpoint=self.endpoint).observe(duration)
                REQUEST_COUNT.labels(method="POST", endpoint=self.endpoint).inc()
                
        return LatencyTracker(endpoint)

    @staticmethod
    def log_drift():
        DRIFT_EVENTS.inc()

    @staticmethod
    def update_accuracy(r2_score):
        MODEL_ACCURACY.set(r2_score)
    
    @staticmethod
    def update_rag_metrics(accuracy, factuality):
        RAG_ACCURACY.set(accuracy)
        RAG_FACTUALITY.set(factuality)
    
    @staticmethod
    def log_retraining():
        RETRAINING_EVENTS.inc()
    
    @staticmethod
    def log_error(endpoint: str, error_type: str):
        ERROR_COUNT.labels(endpoint=endpoint, error_type=error_type).inc()
    
    @staticmethod
    def log_safety_check(safe: bool):
        status = "safe" if safe else "unsafe"
        SAFETY_CHECKS.labels(status=status).inc()
    
    @staticmethod
    def log_prediction(value: float):
        PREDICTION_VALUE.observe(value)