from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics Definitions
REQUEST_COUNT = Counter("app_request_count", "Total request count", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency", ["endpoint"])
DRIFT_EVENTS = Counter("model_drift_events", "Number of times data drift was detected")
MODEL_ACCURACY = Gauge("model_last_r2_score", "R2 Score of the latest model training")
PREDICTION_VALUE = Histogram("predicted_discount_values", "Distribution of predicted discounts")

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