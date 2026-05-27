import time
import queue
import threading
import numpy as np
import joblib
import os
from typing import Callable, Dict, Any, List

from src.config import (
    STREAM_TOPIC, 
    MODEL_DIR, 
    ALL_FEATURES, 
    SEQUENCE_LENGTH, 
    DEFAULT_ALPHA
)
from src.stream.broker import SimulatedKafkaBroker
from src.feature_store import StatefulFeatureStore
from src.models.isolation_forest import AnomalyIsolationForest
from src.models.lstm_autoencoder import AnomalyLSTMAutoencoder

class RealTimeInferenceConsumer(threading.Thread):
    """
    Consumer thread that subscribes to the simulated Kafka broker, processes records, 
    calculates real-time stateful features, performs parallel multi-model inference, 
    calculates consensus scores, and yields the final enriched payloads to web clients.
    """
    def __init__(self, callback: Callable[[Dict[str, Any]], None], alpha: float = DEFAULT_ALPHA):
        super(RealTimeInferenceConsumer, self).__init__()
        self.broker = SimulatedKafkaBroker()
        self.callback = callback
        self.running = False
        self.daemon = True
        self.queue = None
        
        # ML Blending weight (can be altered dynamically from FastAPI dashboard)
        self.alpha = alpha
        
        # Initialize Feature Store
        self.store = StatefulFeatureStore()
        
        # Load pre-trained assets
        self.scaler = None
        self.iforest = None
        self.lstm_ae = None
        self.ensemble_threshold = 0.5
        self.is_ready = False
        
        # We will attempt to load the models
        self.load_ml_assets()

    def load_ml_assets(self) -> bool:
        """Load fitted standard scaler and anomaly detection models."""
        try:
            scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
            iforest_path = os.path.join(MODEL_DIR, "isolation_forest.joblib")
            lstm_path = os.path.join(MODEL_DIR, "lstm_autoencoder.pth")
            ensemble_path = os.path.join(MODEL_DIR, "ensemble_meta.joblib")
            
            if not (os.path.exists(scaler_path) and os.path.exists(iforest_path) and os.path.exists(lstm_path)):
                print("[Consumer WARNING] Pre-trained models not found in models/ directory. Please run the training pipeline first!")
                return False
                
            self.scaler = joblib.load(scaler_path)
            self.iforest = joblib.load(iforest_path)
            
            self.lstm_ae = AnomalyLSTMAutoencoder()
            self.lstm_ae.load(lstm_path)
            
            if os.path.exists(ensemble_path):
                meta = joblib.load(ensemble_path)
                self.ensemble_threshold = meta.get("threshold", 0.5)
                
            self.is_ready = True
            print("[Consumer] Successfully loaded Scaler, Isolation Forest, and LSTM Autoencoder models.")
            return True
        except Exception as e:
            print(f"[Consumer ERROR] Error loading ML assets: {e}")
            return False

    def run(self):
        # Subscribe to topic
        self.queue = self.broker.subscribe(STREAM_TOPIC)
        self.running = True
        
        print("[Consumer] Consumer thread is active and listening for transaction messages...")
        
        while self.running:
            try:
                # Wait for next transaction message from queue (timeout to allow checking self.running flag)
                raw_tx = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            start_latency_time = time.time()
            
            # 1. State Store Enrichment (Calculates velocity, balance discrepancies, coords)
            enriched_tx = self.store.enrich(raw_tx)
            
            # If models are not loaded yet, forward the enriched transaction with zero scores
            if not self.is_ready:
                # Retry loading
                if not self.load_ml_assets():
                    enriched_tx["iforest_score"] = 0.0
                    enriched_tx["lstm_score"] = 0.0
                    enriched_tx["consensus_score"] = 0.0
                    enriched_tx["is_anomaly"] = 0
                    enriched_tx["latency_ms"] = 0.0
                    self.callback(enriched_tx)
                    self.queue.task_done()
                    continue
            
            # 2. Extract scaled raw transaction vector for Isolation Forest
            # Config feature list order: ALL_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
            # Read vector from the end of history for nameOrig
            sender = enriched_tx["nameOrig"]
            feature_dim = len(ALL_FEATURES)
            
            # Get latest stateful feature vector
            latest_tx_list = self.store.history[sender]
            if not latest_tx_list:
                # Fallback, should not happen as enrich() records it
                self.queue.task_done()
                continue
                
            raw_vector = np.array([latest_tx_list[-1]["feature_vector"]])
            scaled_vector = self.scaler.transform(raw_vector)
            
            # 3. Model A: Isolation Forest Inference
            iforest_score_raw = float(self.iforest.predict_score(scaled_vector)[0])
            
            # 4. Model B: PyTorch LSTM Autoencoder Inference
            # Retrieve recent sliding history sequence of length 5
            seq_raw = self.store.get_recent_sequence(sender, SEQUENCE_LENGTH, feature_dim)
            # Scale each item in sequence
            seq_scaled = self.scaler.transform(np.array(seq_raw))
            # Shape: (1, seq_length, feature_dim)
            seq_scaled_batch = np.expand_dims(seq_scaled, axis=0)
            
            lstm_score_raw = float(self.lstm_ae.predict_score(seq_scaled_batch)[0])
            
            # Apply Real-Time Domain Heuristics & Rules
            tx_type = enriched_tx.get("type")
            is_eligible = 1.0 if tx_type in ["TRANSFER", "CASH_OUT"] else 0.0
            
            rule_boost = 0.0
            if tx_type in ["TRANSFER", "CASH_OUT"]:
                # Empty account signature
                if float(enriched_tx.get("newbalanceOrig", 0.0)) == 0.0 and float(enriched_tx.get("oldbalanceOrg", 0.0)) > 0.0 and abs(float(enriched_tx.get("oldbalanceOrg", 0.0)) - float(enriched_tx.get("amount", 0.0))) < 1.0:
                    rule_boost = 0.95
                # Destination discrepancy
                elif tx_type == "TRANSFER" and float(enriched_tx.get("amount", 0.0)) > 1000.0 and abs(float(enriched_tx.get("oldbalanceDest", 0.0)) + float(enriched_tx.get("amount", 0.0)) - float(enriched_tx.get("newbalanceDest", 0.0))) > 10.0:
                    rule_boost = 0.90
                    
            # Overdraft Penalty
            penalty = 1.0
            if float(enriched_tx.get("balance_error_orig", 0.0)) < -1.0:
                penalty = 0.0
                
            iforest_score = max(iforest_score_raw * is_eligible, rule_boost) * penalty
            lstm_score = max(lstm_score_raw * is_eligible, rule_boost) * penalty
            
            # 5. Consensus Ensemble Blending
            consensus_score = self.alpha * iforest_score + (1.0 - self.alpha) * lstm_score
            
            # Anomaly Classification Flag
            is_anomaly = int(consensus_score > self.ensemble_threshold)
            
            # Calculate System Inference Latency
            latency_ms = (time.time() - start_latency_time) * 1000.0
            
            # 6. Package enriched payload
            enriched_tx["iforest_score"] = iforest_score
            enriched_tx["lstm_score"] = lstm_score
            enriched_tx["consensus_score"] = consensus_score
            enriched_tx["is_anomaly"] = is_anomaly
            enriched_tx["latency_ms"] = latency_ms
            
            # Set thresholds
            enriched_tx["iforest_threshold"] = self.iforest.threshold
            enriched_tx["lstm_threshold"] = self.lstm_ae.threshold
            enriched_tx["ensemble_threshold"] = self.ensemble_threshold
            
            # Send payload to callback (FastAPI dashboard broadcaster)
            self.callback(enriched_tx)
            
            self.queue.task_done()

    def update_alpha(self, new_alpha: float):
        """Allow on-the-fly adjustment of ensemble blending parameter alpha."""
        self.alpha = float(np.clip(new_alpha, 0.0, 1.0))
        print(f"[Consumer] Dynamically updated ensemble alpha blending weight to: {self.alpha}")

    def update_threshold(self, new_threshold: float):
        """Allow on-the-fly threshold adjustments from dashboard server."""
        self.ensemble_threshold = float(new_threshold)
        print(f"[Consumer] Dynamically updated consensus anomaly threshold to: {self.ensemble_threshold}")

    def stop(self):
        """Gracefully shutdown consumer, unsubscribing from broker."""
        self.running = False
        if self.queue:
            self.broker.unsubscribe(STREAM_TOPIC, self.queue)
        print("[Consumer] Consumer thread signaled to stop.")
