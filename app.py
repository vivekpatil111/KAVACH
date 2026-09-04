import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import xgboost as xgb
import pandas as pd
import uvicorn
from contextlib import asynccontextmanager

# Global variables to hold the loaded model
MODEL_PATH = "models/chargeback_xgb.json"
model = None
OPERATING_THRESHOLD = 0.70

# Define the Pydantic schema for the 36 input features
class TransactionFeatures(BaseModel):
    has_billing_addr: int = Field(..., description="Address & Mismatch")
    ProductCD_train_fraud_rate: float = Field(..., description="Product Category Risk")
    amt_is_round_dollar: int = Field(..., description="Amount Anomaly")
    card_address_count_C1: float = Field(..., description="Historical Entity Hopping")
    has_identity_data: int = Field(..., description="Device/Identity Presence")
    email_count_C13: float = Field(..., description="Account Velocity")
    has_recipient_email: int = Field(..., description="Dropshipping / Gift Card Risk")
    addr_card_country_mismatch: int = Field(..., description="Cross-Border Discrepancy")
    is_mobile_device: int = Field(..., description="Device Fingerprint")
    time_delta_prev_txn_D2: float = Field(..., description="Transaction Timing Velocity")
    amt_cents: float = Field(..., description="Amount Granularity")
    transaction_count_C2: float = Field(..., description="Historical Volume")
    browser_is_known: int = Field(..., description="Identity Verification")
    avs_shipping_match: int = Field(..., description="AVS Fulfillment Match")
    amt_log: float = Field(..., description="Monetary Scale")
    is_desktop_device: int = Field(..., description="Device Form Factor")
    os_is_known: int = Field(..., description="Operating System Verification")
    time_delta_card_creation_D1: float = Field(..., description="Account Age")
    email_domain_mismatch: int = Field(..., description="P/R Email Discrepancy")
    ip_proxy_risk_flag: int = Field(..., description="Proxy / Anonymizer Flag")
    card_txn_count_7d: float = Field(..., description="Weekly Card Velocity")
    avs_address_match: int = Field(..., description="AVS Address Match")
    amt_to_card_mean_ratio: float = Field(..., description="Relative Spike Ratio")
    email_domain_txn_count_24h: float = Field(..., description="Email Burst Velocity")
    shipping_billing_dist: float = Field(..., description="Physical Distance")
    total_verification_matches: float = Field(..., description="Multi-Factor Match Depth")
    card_txn_count_24h: float = Field(..., description="Daily Card Velocity")
    has_device_info: int = Field(..., description="Hardware Telemetry")
    card_txn_amt_sum_24h: float = Field(..., description="Trailing Dollar Exposure")
    card_time_since_last_txn_hours: float = Field(..., description="Inter-Transaction Velocity")
    hour_of_day: float = Field(..., description="Circadian Context")
    card_amt_zscore: float = Field(..., description="Relative Card Deviation")
    avs_billing_match: int = Field(..., description="AVS Billing Match")
    day_of_week: float = Field(..., description="Day Context")
    is_night_transaction: int = Field(..., description="Off-Hours Flag")
    has_shipping_dist: int = Field(..., description="Distance Availability")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"Loading XGBoost model from {MODEL_PATH}...")
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    yield
    print("Shutting down model inference service.")

app = FastAPI(
    title="Chargeback Evidence Responder Inference API",
    description="Real-time XGBoost inference endpoint for fraud detection.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/v1/health")
async def health_check():
    """Basic liveness check."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}

@app.post("/v1/score")
async def score_transaction(features: TransactionFeatures):
    """
    Score a transaction for fraud risk using the loaded XGBoost model.
    Accepts 36 explicitly defined hand-engineered features.
    """
    start_time = time.perf_counter()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
        
    try:
        # Expected feature order from XGBoost model
        expected_features = [
            'card_txn_count_24h', 'card_txn_count_7d', 'card_txn_amt_sum_24h', 
            'card_time_since_last_txn_hours', 'email_domain_txn_count_24h', 
            'card_amt_zscore', 'amt_to_card_mean_ratio', 'amt_log', 
            'amt_is_round_dollar', 'amt_cents', 'ProductCD_train_fraud_rate', 
            'addr_card_country_mismatch', 'has_billing_addr', 'has_shipping_dist', 
            'shipping_billing_dist', 'email_domain_mismatch', 'has_recipient_email', 
            'has_identity_data', 'is_mobile_device', 'is_desktop_device', 
            'has_device_info', 'browser_is_known', 'os_is_known', 'ip_proxy_risk_flag', 
            'avs_address_match', 'avs_billing_match', 'avs_shipping_match', 
            'total_verification_matches', 'card_address_count_C1', 'transaction_count_C2', 
            'email_count_C13', 'time_delta_card_creation_D1', 'time_delta_prev_txn_D2', 
            'hour_of_day', 'is_night_transaction', 'day_of_week'
        ]
        
        # Convert Pydantic model to DataFrame and reorder columns
        df = pd.DataFrame([features.dict()])
        df = df[expected_features]
        
        # Predict probability for class 1 (Fraud)
        proba = model.predict_proba(df)[0, 1]
        
        predicted_class = "HIGH_RISK" if proba >= OPERATING_THRESHOLD else "LOW_RISK"
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "risk_score": float(proba),
            "predicted_class": predicted_class,
            "threshold_used": OPERATING_THRESHOLD,
            "latency_ms": latency_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
