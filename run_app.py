import os
import sys

from src.config import MODEL_DIR

def check_and_train_models():
    """
    Check if the pre-trained standard scaler and model files exist.
    If missing, automatically execute src/train.py to train, calibrate, 
    and serialize the Isolation Forest and LSTM Autoencoder.
    """
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    iforest_path = os.path.join(MODEL_DIR, "isolation_forest.joblib")
    lstm_path = os.path.join(MODEL_DIR, "lstm_autoencoder.pth")
    
    if not (os.path.exists(scaler_path) and os.path.exists(iforest_path) and os.path.exists(lstm_path)):
        print("\n========================================================")
        
        # Checking if real dataset is present. If not, synthesis happens
        real_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "PS_20174392719_1491204439457_log.csv")
        if os.path.exists(real_data_path):
            print(" [Bootstrap] Found real PaySim dataset. Bootstrapping training...")
        else:
            print(" [Bootstrap] No model checkpoint files or dataset detected.")
            print(" [Bootstrap] Launching automatic synthetic transaction synthesis and training...")
            
        print("========================================================\n")
        
        # Run training loop in-process
        from src.train import main as run_train
        run_train()
    else:
        print("\n[Bootstrap] Pre-trained models verified. Ready to launch dashboard directly.\n")

if __name__ == "__main__":
    # Ensure working directory is added to Python path for absolute imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        
    # 1. Self-healing model bootstrap check
    check_and_train_models()
    
    # 2. Boot up the FastAPI Web Server
    print("========================================================")
    print(" LAUNCHING SENTRY-AI REAL-TIME SECURE STREAM SERVER")
    print("========================================================")
    print(" Web Dashboard: http://localhost:8000")
    print(" WebSocket Tunnel: ws://localhost:8000/ws")
    print(" Press Ctrl+C to terminate all worker threads safely.")
    print("========================================================\n")
    
    import uvicorn
    uvicorn.run("src.dashboard.server:app", host="0.0.0.0", port=8000, log_level="info")
