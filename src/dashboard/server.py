import os
import json
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from typing import Set

from src.config import BASE_DIR, LOG_DIR, DEFAULT_ALPHA
from src.stream.producer import SimulatedKafkaProducer
from src.stream.consumer import RealTimeInferenceConsumer

# Global running loop pointer to safely schedule tasks from other OS threads
main_loop = None

app = FastAPI(title="Real-Time Fraud Detection Dashboard")

# Active WebSocket connections
active_connections: Set[WebSocket] = set()

# Thread handles
producer_thread = None
consumer_thread = None

# Active Learning Log path
FEEDBACK_LOG_PATH = os.path.join(LOG_DIR, "active_learning_feedback.jsonl")

# Schema for runtime tuning
class ModelConfig(BaseModel):
    alpha: float
    threshold: float

# Schema for active learning feedback
class FraudFeedback(BaseModel):
    timestamp: float
    nameOrig: str
    amount: float
    consensus_score: float
    system_flag: int
    user_label: str  # "CONFIRM_FRAUD" or "FALSE_ALARM"

# Schema for streaming control
class StreamControl(BaseModel):
    action: str  # "START", "PAUSE", "RESUME"

def broadcast_callback(enriched_tx: dict):
    """
    Callback executed by the Consumer thread whenever a new transaction is processed.
    Enqueues the message into the main asyncio loop to be broadcasted to all connected WebSockets.
    """
    global main_loop
    if not active_connections or main_loop is None:
        return
        
    # Schedule the coroutine on the running main FastAPI loop safely across threads
    asyncio.run_coroutine_threadsafe(broadcast_to_clients(enriched_tx), main_loop)

async def broadcast_to_clients(enriched_tx: dict):
    """Broadcast enriched transaction JSON payload to all active WebSocket clients."""
    if not active_connections:
        return
        
    # Prepare serializable values (convert numpy values to native Python floats/ints)
    clean_tx = {}
    for k, v in enriched_tx.items():
        if isinstance(v, (np.float32, np.float64)):
            clean_tx[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            clean_tx[k] = int(v)
        else:
            clean_tx[k] = v
            
    disconnected = set()
    for ws in active_connections:
        try:
            await ws.send_json(clean_tx)
        except Exception:
            disconnected.add(ws)
            
    for ws in disconnected:
        active_connections.remove(ws)

@app.on_event("startup")
def startup_event():
    """Startup worker threads on application boot."""
    global producer_thread, consumer_thread, main_loop
    
    # Store the actual running FastAPI event loop of the main thread
    main_loop = asyncio.get_running_loop()
    
    print("[Server] Starting system consumer thread...")
    consumer_thread = RealTimeInferenceConsumer(callback=broadcast_callback, alpha=DEFAULT_ALPHA)
    consumer_thread.start()
    
    print("[Server] Starting transaction stream producer thread...")
    producer_thread = SimulatedKafkaProducer(delay=0.15) # ~6 transactions per second
    producer_thread.start()
    
    print("[Server] FastAPI application fully started and background workers are active.")

@app.on_event("shutdown")
def shutdown_event():
    """Graceful termination of producer and consumer threads."""
    global producer_thread, consumer_thread
    
    if producer_thread:
        producer_thread.stop()
    if consumer_thread:
        consumer_thread.stop()
        
    print("[Server] Shutdown complete. All threads closed.")

@app.get("/")
async def get_dashboard():
    """Serve the premium HTML5 Dashboard page directly."""
    html_path = os.path.join(BASE_DIR, "src", "dashboard", "templates", "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse(content="<h1>Dashboard Template Not Found!</h1>", status_code=404)
        
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection endpoint for streaming live transactions."""
    await websocket.accept()
    active_connections.add(websocket)
    print(f"[Server] WebSocket client connected. Active connections: {len(active_connections)}")
    try:
        # Keep connection open
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"[Server] WebSocket client disconnected. Active connections: {len(active_connections)}")

@app.post("/api/config")
async def update_config(config: ModelConfig):
    """Adjust ensemble alpha blending weight and consensus threshold on-the-fly."""
    global consumer_thread
    if consumer_thread and consumer_thread.is_alive():
        consumer_thread.update_alpha(config.alpha)
        consumer_thread.update_threshold(config.threshold)
        return JSONResponse(content={
            "status": "success", 
            "alpha": config.alpha, 
            "threshold": config.threshold
        })
    return JSONResponse(content={"status": "error", "message": "Consumer thread inactive"}, status_code=500)

@app.post("/api/feedback")
async def active_learning_feedback(feedback: FraudFeedback):
    """
    Active Learning Endpoint. Receives confirmed fraud labeling feedback from 
    dashboard and logs it chronologically to append into training retraining loops.
    """
    try:
        record = feedback.dict()
        record["logged_at"] = time.time()
        
        # Write to JSONL active learning log file
        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
        print(f"[Active Learning] Feedback recorded for sender {feedback.nameOrig}: label = {feedback.user_label}")
        return JSONResponse(content={"status": "success", "message": "Feedback captured"})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/control")
async def control_stream(control: StreamControl):
    """Control (Start/Pause) the transaction stream producer thread."""
    global producer_thread
    action = control.action.upper()
    
    if not producer_thread:
        return JSONResponse(content={"status": "error", "message": "Producer not initialized"}, status_code=500)
        
    if action == "PAUSE":
        producer_thread.stop()
        return JSONResponse(content={"status": "success", "message": "Stream paused"})
    elif action in ["START", "RESUME"]:
        if not producer_thread.running:
            # Recreate thread as Python threads cannot be restarted once stopped
            producer_thread = SimulatedKafkaProducer(delay=0.15)
            producer_thread.start()
            return JSONResponse(content={"status": "success", "message": "Stream resumed"})
        return JSONResponse(content={"status": "success", "message": "Stream already running"})
        
    return JSONResponse(content={"status": "error", "message": "Invalid action"}, status_code=400)
