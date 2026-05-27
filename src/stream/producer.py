import os
import time
import threading
import csv
import random
from typing import Optional

from src.config import (
    STREAM_TOPIC, 
    STREAM_DELAY, 
    REAL_DATA_PATH, 
    SYNTHETIC_DATA_PATH
)
from src.stream.broker import SimulatedKafkaBroker

class SimulatedKafkaProducer(threading.Thread):
    """
    Simulation producer thread that reads from our offline dataset records 
    and publishes them to the Kafka broker at a configured time frequency.
    Uses standard built-in csv library (fully pandas-free).
    """
    def __init__(self, data_path: Optional[str] = None, delay: float = STREAM_DELAY):
        super(SimulatedKafkaProducer, self).__init__()
        self.broker = SimulatedKafkaBroker()
        self.delay = delay
        self.running = False
        self.daemon = True
        
        # Load dataset
        if data_path and os.path.exists(data_path):
            self.data_path = data_path
        elif os.path.exists(REAL_DATA_PATH):
            self.data_path = REAL_DATA_PATH
        elif os.path.exists(SYNTHETIC_DATA_PATH):
            self.data_path = SYNTHETIC_DATA_PATH
        else:
            raise FileNotFoundError("No transaction dataset found. Please run training pipeline first!")
            
        print(f"[Producer] Initializing transaction stream using data source: {self.data_path}")

    def run(self):
        print(f"[Producer] Loading dataset rows to memory...")
        records = []
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Cast fields to float/int
                records.append({
                    "step": int(row["step"]),
                    "type": row["type"],
                    "amount": float(row["amount"]),
                    "nameOrig": row["nameOrig"],
                    "oldbalanceOrg": float(row["oldbalanceOrg"]),
                    "newbalanceOrig": float(row["newbalanceOrig"]),
                    "nameDest": row["nameDest"],
                    "oldbalanceDest": float(row["oldbalanceDest"]),
                    "newbalanceDest": float(row["newbalanceDest"]),
                    "isFraud": int(row["isFraud"]),
                    "isFlaggedFraud": int(row["isFlaggedFraud"]),
                    "latitude": float(row["latitude"]) if "latitude" in row else random.uniform(40.5, 40.9),
                    "longitude": float(row["longitude"]) if "longitude" in row else random.uniform(-74.3, -73.7),
                    "timestamp": float(row["timestamp"]) if "timestamp" in row else time.time()
                })
                if len(records) >= 150000:
                    break
        
        # Sort chronologically
        records.sort(key=lambda x: x["timestamp"])
        n_records = len(records)
        print(f"[Producer] Loaded {n_records} records. Streaming started...")
        
        self.running = True
        idx = 0
        
        while self.running:
            record = records[idx % n_records]
            
            # Enrich with real-time timestamps so the Feature Store works in real time
            record["timestamp"] = time.time()
            
            # Publish to topic
            self.broker.publish(STREAM_TOPIC, record)
            
            idx += 1
            if idx % 100 == 0:
                print(f"[Producer] Stream status: published {idx} transactions total.")
                
            time.sleep(self.delay)

    def stop(self):
        """Signal thread to stop execution."""
        self.running = False
        print("[Producer] Streaming thread flagged to stop.")
