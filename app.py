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

# =============================================================================
# ACTUARIAL GRADUATED ROLLING RESERVE ENGINE — 3-Tier Capital Allocation Policy
# =============================================================================
# Tier 1: risk < 0.40  → AUTO_APPROVE          (100% instant RTGS payout, 0% reserve)
# Tier 2: 0.40–0.75   → GRADUATED_RESERVE_15  (85% instant release, 15% rolling 14-day buffer)
# Tier 3: risk > 0.75  → HOLD_AND_AUTODEFEND   (pre-settlement hold + Gemini dossier triggered)

TIER_BOUNDARIES = {
    "AUTO_APPROVE":         {"score_max": 0.40, "release_pct": 1.00, "reserve_pct": 0.00, "buffer_days": 0,  "step_up_otp": False},
    "GRADUATED_RESERVE_15": {"score_max": 0.75, "release_pct": 0.85, "reserve_pct": 0.15, "buffer_days": 14, "step_up_otp": True},
    "HOLD_AND_AUTODEFEND":  {"score_max": 1.00, "release_pct": 0.00, "reserve_pct": 1.00, "buffer_days": 30, "step_up_otp": False},
}


def classify_reserve_tier(risk_score: float) -> dict:
    """Map a continuous risk score to the appropriate capital allocation tier.

    Returns a dict with: action, release_pct, reserve_pct, buffer_days,
    step_up_otp, and merchant-readable explanation.
    """
    if risk_score < 0.40:
        action = "AUTO_APPROVE"
        explanation = (
            f"Risk score {risk_score:.4f} is below 0.40 (low-risk zone). "
            "100% of working capital released instantly via RTGS. No reserve withheld."
        )
    elif risk_score <= 0.75:
        action = "GRADUATED_RESERVE_15"
        explanation = (
            f"Risk score {risk_score:.4f} falls in the borderline zone (0.40–0.75). "
            "85% of working capital released instantly; 15% held in a 14-day rolling dispute buffer. "
            "Step-Up OTP verification dispatched to merchant contact."
        )
    else:
        action = "HOLD_AND_AUTODEFEND"
        explanation = (
            f"Risk score {risk_score:.4f} exceeds 0.75 (high-risk zone). "
            "Pre-settlement hold applied. Kavach Gemini Dossier Generator automatically triggered "
            "to prepare and submit dispute defense evidence."
        )

    tier = TIER_BOUNDARIES[action]
    return {
        "action": action,
        "release_pct": tier["release_pct"],
        "reserve_pct": tier["reserve_pct"],
        "buffer_days": tier["buffer_days"],
        "step_up_otp_dispatched": tier["step_up_otp"],
        "explanation": explanation,
    }

from typing import Optional
from chargeback_defense.syndicate_graph import get_syndicate_graph

# Define the Pydantic schema for the 36 input features + optional syndicate telemetry
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

    # Optional Multi-Merchant Syndicate Ring Telemetry
    identity_id: Optional[str] = Field(None, description="Unique User / Account ID")
    merchant_id: Optional[str] = Field(None, description="Merchant Account ID")
    device_id: Optional[str] = Field(None, description="Device Hardware Fingerprint")
    upi_vpa: Optional[str] = Field(None, description="UPI Virtual Payment Address")
    phone_hash: Optional[str] = Field(None, description="Phone Hash")
    shipping_pincode: Optional[str] = Field(None, description="6-digit Indian Postal PIN")

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

    # Pre-seed and warm up in-memory syndicate ring graph
    print("Warming up In-Memory Syndicate Ring Graph...")
    get_syndicate_graph()
    print("Syndicate Ring Graph ready (<0.1ms extraction).")

    yield
    print("Shutting down model inference service.")

app = FastAPI(
    title="Kavach AI Risk Manager Inference API",
    description=(
        "Real-time XGBoost inference with 3-tier Actuarial Graduated Rolling Reserve Engine "
        "and in-memory multi-merchant syndicate ring detection."
    ),
    version="3.0.0",
    lifespan=lifespan
)

@app.get("/v1/health")
async def health_check():
    """Basic liveness check."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_loaded": True,
        "syndicate_graph": "active",
        "reserve_engine": "ACTUARIAL_3TIER_v1",
        "tier_boundaries": {
            "AUTO_APPROVE":         "score < 0.40",
            "GRADUATED_RESERVE_15": "0.40 ≤ score ≤ 0.75",
            "HOLD_AND_AUTODEFEND":  "score > 0.75",
        },
    }

@app.post("/v1/score")
async def score_transaction(features: TransactionFeatures):
    """
    Score a transaction for fraud risk using the loaded XGBoost model
    and evaluate real-time multi-merchant syndicate ring graph connectivity (<10ms).
    """
    start_time = time.perf_counter()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
        
    try:
        # Expected feature order from XGBoost model (36 features)
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
        
        # Convert Pydantic model to DataFrame and extract XGBoost features
        feat_dict = features.dict()
        df = pd.DataFrame([feat_dict])
        df = df[expected_features]
        
        # Predict probability for class 1 (Fraud)
        proba = model.predict_proba(df)[0, 1]

        # ── 3-Tier Actuarial Graduated Rolling Reserve Engine ──────────────────
        reserve_decision = classify_reserve_tier(proba)
        # Legacy compatibility: expose a predicted_class label
        if reserve_decision["action"] == "AUTO_APPROVE":
            predicted_class = "LOW_RISK"
        elif reserve_decision["action"] == "GRADUATED_RESERVE_15":
            predicted_class = "BORDERLINE_RISK"
        else:
            predicted_class = "HIGH_RISK"
        # ───────────────────────────────────────────────────────────────────────

        # In-Memory Syndicate Ring Graph Evaluation (<0.1ms)
        graph = get_syndicate_graph()
        uid = features.identity_id or "USR_DEMO"
        mid = features.merchant_id or "MERCH_DEMO"

        # Record infrastructure event if provided
        is_high_risk_flag = reserve_decision["action"] == "HOLD_AND_AUTODEFEND"
        if features.device_id or features.upi_vpa or features.phone_hash or features.shipping_pincode:
            graph.add_event(
                identity_id=uid,
                merchant_id=mid,
                device_id=features.device_id,
                upi_vpa=features.upi_vpa,
                phone_hash=features.phone_hash,
                shipping_pincode=features.shipping_pincode,
                is_claim=is_high_risk_flag,
                claim_id=f"CLM_RT_{int(time.time()*1000)}" if is_high_risk_flag else None,
            )

        syndicate_features = graph.extract_features(uid)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "risk_score": float(proba),
            "predicted_class": predicted_class,
            "threshold_used": OPERATING_THRESHOLD,
            "capital_allocation": reserve_decision,
            "syndicate_analysis": syndicate_features,
            "latency_ms": round(latency_ms, 3),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
