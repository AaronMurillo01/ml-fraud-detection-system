#!/usr/bin/env python3
"""
Simple test server to demonstrate the enhanced UI
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import random
import time
import asyncio
from datetime import datetime

app = FastAPI(title="FraudGuard AI - Enhanced UI Demo")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the enhanced web UI."""
    return FileResponse('static/index_enhanced.html')

@app.get("/legacy")
async def legacy_ui():
    """Serve the original web UI."""
    return FileResponse('static/index.html')

@app.post("/api/v1/fraud/predict")
async def predict_fraud(transaction_data: dict):
    """Mock fraud prediction endpoint for testing the enhanced UI."""
    
    # Simulate processing time
    processing_start = time.time()
    await asyncio.sleep(random.uniform(0.5, 2.0))  # Simulate ML processing
    processing_time = (time.time() - processing_start) * 1000
    
    # Generate mock prediction results
    fraud_probability = random.uniform(0.05, 0.95)
    confidence = random.uniform(0.7, 0.99)
    
    # Determine risk level and action
    if fraud_probability < 0.3:
        recommended_action = "APPROVE"
        risk_level = "LOW"
    elif fraud_probability < 0.7:
        recommended_action = "REVIEW"
        risk_level = "MEDIUM"
    else:
        recommended_action = "DECLINE"
        risk_level = "HIGH"
    
    # Generate mock risk factors
    risk_factors = {
        "Amount Risk": random.uniform(0.1, 0.9),
        "Location Risk": random.uniform(0.1, 0.8),
        "Time Risk": random.uniform(0.1, 0.7),
        "Merchant Risk": random.uniform(0.1, 0.6),
        "User Behavior": random.uniform(0.1, 0.8),
        "Payment Method": random.uniform(0.1, 0.5)
    }
    
    return JSONResponse({
        "transaction_id": transaction_data.get("transactionId", "unknown"),
        "fraud_probability": fraud_probability,
        "confidence": confidence,
        "recommended_action": recommended_action,
        "risk_level": risk_level,
        "processing_time_ms": round(processing_time, 2),
        "model_version": "enhanced_demo_v2.0",
        "risk_factors": risk_factors,
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    })

@app.get("/api/v1/stats")
async def get_system_stats():
    """Mock system statistics endpoint."""
    return JSONResponse({
        "total_predictions": random.randint(15000, 20000),
        "fraud_detected": random.randint(300, 500),
        "accuracy": random.uniform(0.94, 0.98),
        "avg_response_time_ms": random.uniform(20, 50),
        "uptime_hours": random.randint(100, 1000),
        "last_updated": datetime.now().isoformat()
    })

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "enhanced_demo_v2.0"
    })

@app.get("/docs")
async def api_docs():
    """Redirect to FastAPI docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import asyncio
    print("🚀 Starting FraudGuard AI Enhanced UI Demo Server...")
    print("📱 Enhanced UI: http://localhost:8000")
    print("🔗 Legacy UI: http://localhost:8000/legacy")
    print("📚 API Docs: http://localhost:8000/docs")
    print("❤️  Health Check: http://localhost:8000/health")
    print("\n✨ Features in Enhanced UI:")
    print("   • Modern responsive design")
    print("   • Dark/Light mode toggle")
    print("   • Real-time form validation")
    print("   • Interactive data visualization")
    print("   • Enhanced loading states")
    print("   • Toast notifications")
    print("   • Transaction history")
    print("   • Improved accessibility")
    print("\n🎯 Try adding '?demo=true' to auto-fill the form!")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
