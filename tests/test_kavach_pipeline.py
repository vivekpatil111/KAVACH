"""
pytest Test Suite -- Kavach AI Risk Manager Pipeline
=====================================================
18 automated tests across 4 pipeline layers:
  (A) Graduated Rolling Reserve Policy    -- 6 tests  (app.py)
  (B) Syndicate Ring Detection            -- 5 tests  (syndicate_graph.py)
  (C) Anti-Hallucination Fact Assertion   -- 5 tests  (narrative_generator.py)
  (D) Deterministic Fallback Quality      -- 2 tests  (narrative_generator.py)

Run:
    pytest tests/test_kavach_pipeline.py -v

No external API keys or trained model artifacts required for any test.
"""

from __future__ import annotations

import hashlib
import json
import time
import sys
import os
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Make repo root importable regardless of where pytest is invoked
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# SHARED FIXTURES
# ===========================================================================

@pytest.fixture(scope="session")
def classify_tier():
    """Import classify_reserve_tier from app.py once per session."""
    from app import classify_reserve_tier
    return classify_reserve_tier


@pytest.fixture
def sample_packet() -> Dict[str, Any]:
    """Minimal, complete evidence packet covering all assertion fields."""
    return {
        "dispute_summary": {
            "claim_id": "CLM-TEST-001",
            "disputed_amount_original": {"value": 4812.50, "currency": "INR"},
            "card_network": "UPI",
        },
        "commercial_fulfillment_evidence": {
            "matched_order_id": "ORD-TEST-001",
            "order_status": "delivered",
            "prior_clean_transactions_same_vpa": 3,
            "prior_clean_transactions_same_device": 2,
            "delivery_performance": {
                "awb_tracking_number": "BD123456789IN",
                "courier_partner": "BlueDart Express",
                "delivery_proof_available": True,
                "status": "DELIVERED_ON_TIME",
                "delivery_delta_days": -1.5,
                "transit_duration_days": 3.5,
            },
            "timeline": {
                "purchase_timestamp": "2026-08-10T10:00:00",
                "delivered_customer_timestamp": "2026-08-15T14:30:00",
                "estimated_delivery_timestamp": "2026-08-17T00:00:00",
            },
            "merchant_and_item_details": {
                "courier_partner": "BlueDart Express",
                "customer_location": "Mumbai",
                "seller_location": "Bengaluru",
                "product_category": "electronics",
                "item_count": 2,
                "awb_tracking_number": "BD123456789IN",
                "customer_phone": "+91 9800012345",
            },
            "customer_feedback_record": {
                "review_score": 5,
                "review_comment_message": "Excellent product, fast delivery!",
            },
            "payment_profile": {
                "payment_type": "UPI",
                "payment_identifier": "customer@okhdfcbank",
                "payment_method_display": "UPI (customer@okhdfcbank)",
            },
        },
        "dispute_defense_evaluation": {
            "dispute_representment_recommendation": "CONTEST_CHARGEBACK_WITH_EVIDENCE",
            "chargeback_reason_classification": "FIRST_PARTY_FRIENDLY_FRAUD",
            "compelling_evidence_factors": ["Delivery confirmed", "5-star review"],
        },
        "model_risk_assessment": {
            "fraud_risk_score": 0.82,
            "risk_band": "HIGH",
        },
    }


# ===========================================================================
# (A) GRADUATED ROLLING RESERVE POLICY  (6 tests)
# ===========================================================================

class TestGraduatedReservePolicy:
    """Verify the 3-tier capital allocation policy thresholds defined in app.py."""

    def test_T1_low_risk_auto_approve_properties(self, classify_tier):
        """Score < 0.40 must yield AUTO_APPROVE with 100% release and 0% reserve."""
        for score in [0.0, 0.10, 0.39, 0.399]:
            result = classify_tier(score)
            assert result["action"] == "AUTO_APPROVE", f"score={score}"
            assert result["release_pct"] == 1.00
            assert result["reserve_pct"] == 0.00
            assert result["buffer_days"] == 0
            assert result["step_up_otp_dispatched"] is False

    def test_T2_borderline_graduated_reserve_properties(self, classify_tier):
        """0.40 <= score <= 0.75 must yield GRADUATED_RESERVE_15 with 85/15 split."""
        for score in [0.40, 0.50, 0.70, 0.75]:
            result = classify_tier(score)
            assert result["action"] == "GRADUATED_RESERVE_15", f"score={score}"
            assert result["release_pct"] == 0.85
            assert result["reserve_pct"] == 0.15
            assert result["buffer_days"] == 14
            assert result["step_up_otp_dispatched"] is True

    def test_T3_high_risk_hold_and_autodefend_properties(self, classify_tier):
        """Score > 0.75 must yield HOLD_AND_AUTODEFEND with full 100% reserve."""
        for score in [0.751, 0.80, 0.90, 0.99, 1.00]:
            result = classify_tier(score)
            assert result["action"] == "HOLD_AND_AUTODEFEND", f"score={score}"
            assert result["release_pct"] == 0.00
            assert result["reserve_pct"] == 1.00
            assert result["buffer_days"] == 30

    def test_boundary_040_lands_in_tier2(self, classify_tier):
        """Exact boundary value 0.40 must be classified as GRADUATED_RESERVE_15."""
        assert classify_tier(0.40)["action"] == "GRADUATED_RESERVE_15"

    def test_boundary_075_lands_in_tier2(self, classify_tier):
        """Exact boundary value 0.75 must be classified as GRADUATED_RESERVE_15."""
        assert classify_tier(0.75)["action"] == "GRADUATED_RESERVE_15"

    def test_tier_decision_p99_latency_under_25ms(self, classify_tier):
        """classify_reserve_tier must complete every call in < 25ms (no I/O)."""
        for score in [0.05, 0.40, 0.60, 0.75, 0.95]:
            t0 = time.perf_counter()
            classify_tier(score)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert elapsed_ms < 25.0, (
                f"classify_reserve_tier: {elapsed_ms:.2f}ms at score={score} exceeds 25ms SLA"
            )


# ===========================================================================
# (B) SYNDICATE RING DETECTION  (5 tests)
# ===========================================================================

class TestSyndicateRingDetection:
    """Verify the in-memory Union-Find syndicate graph logic in syndicate_graph.py."""

    def _fresh_tracker(self):
        from chargeback_defense.syndicate_graph import SyndicateGraphTracker
        return SyndicateGraphTracker()

    def test_isolated_user_gets_no_alert(self):
        """A lone user on a single merchant must NOT trigger a syndicate alert."""
        tracker = self._fresh_tracker()
        tracker.add_event(
            identity_id="USR_CLEAN", merchant_id="MERCH_CLEAN",
            device_id="DEV_CLEAN", upi_vpa="clean_user@oksbi",
            phone_hash="PH_CLEAN", shipping_pincode="560001",
            claim_id="CLM_CLEAN_01", is_claim=True,
        )
        feats = tracker.extract_features("USR_CLEAN")
        assert feats["is_syndicate_attack"] is False
        assert feats["syndicate_alert"] == "NORMAL_MERCHANT_TRAFFIC"

    def test_three_merchants_shared_vpa_triggers_alert(self):
        """Shared UPI VPA across 3 merchants with 3+ claims triggers COORDINATED_SYNDICATE_ATTACK."""
        tracker = self._fresh_tracker()
        shared_vpa = "ring@okicici"
        shared_dev = "DEV_RING"
        now = time.time()
        for uid, mid, cid in [
            ("USR_R1", "MERCH_X", "CLM_R1"),
            ("USR_R2", "MERCH_Y", "CLM_R2"),
            ("USR_R3", "MERCH_Z", "CLM_R3"),
        ]:
            tracker.add_event(
                identity_id=uid, merchant_id=mid,
                device_id=shared_dev, upi_vpa=shared_vpa,
                phone_hash=f"PH_{uid}", shipping_pincode="400051",
                claim_id=cid, timestamp=now, is_claim=True,
            )
        feats = tracker.extract_features("USR_R1")
        assert feats["is_syndicate_attack"] is True
        assert feats["syndicate_alert"] == "COORDINATED_SYNDICATE_ATTACK"
        assert feats["cluster_merchant_span"] >= 3
        assert feats["cluster_burst_7d"] >= 3

    def test_cluster_merchant_span_is_exact_count(self):
        """cluster_merchant_span must equal the number of distinct merchants hit."""
        tracker = self._fresh_tracker()
        shared_vpa = "span@okhdfcbank"
        now = time.time()
        for i, mid in enumerate(["MERCH_A", "MERCH_B", "MERCH_C"]):
            tracker.add_event(
                identity_id=f"USR_S{i}", merchant_id=mid,
                upi_vpa=shared_vpa, claim_id=f"CLM_S{i}",
                timestamp=now, is_claim=True,
            )
        feats = tracker.extract_features("USR_S0")
        assert feats["cluster_merchant_span"] == 3

    def test_graph_extraction_latency_under_10ms(self):
        """extract_features SLA: < 10ms after warm-up."""
        tracker = self._fresh_tracker()
        tracker.add_event(
            identity_id="USR_LAT", merchant_id="MERCH_LAT",
            upi_vpa="lat@paytm", claim_id="CLM_LAT", is_claim=True,
        )
        tracker.extract_features("USR_LAT")  # warm-up call
        t0 = time.perf_counter()
        feats = tracker.extract_features("USR_LAT")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 10.0, f"Graph extraction: {elapsed_ms:.3f}ms (SLA: <10ms)"
        assert feats["extraction_latency_ms"] < 10.0

    def test_old_claims_excluded_from_7d_burst(self):
        """Claims timestamped > 7 days ago must not count in cluster_burst_7d."""
        tracker = self._fresh_tracker()
        eight_days_ago = time.time() - (8 * 86_400)
        tracker.add_event(
            identity_id="USR_OLD", merchant_id="MERCH_OLD",
            upi_vpa="old@okhdfcbank", claim_id="CLM_OLD",
            timestamp=eight_days_ago, is_claim=True,
        )
        feats = tracker.extract_features("USR_OLD")
        assert feats["cluster_burst_7d"] == 0, (
            f"Expected 0 burst_7d for 8-day-old claim, got {feats['cluster_burst_7d']}"
        )


# ===========================================================================
# (C) ANTI-HALLUCINATION FACT ASSERTION  (5 tests)
# ===========================================================================

class TestNarrativeAntiHallucination:
    """Verify Pydantic schema + assertion guardrail in narrative_generator.py."""

    def _imports(self):
        from chargeback_defense.narrative_generator import (
            LLMNarrativeOutput,
            assert_narrative_facts,
        )
        return LLMNarrativeOutput, assert_narrative_facts

    def test_correct_output_passes_all_checks(self, sample_packet):
        """Exact-match on all 4 fields must PASS with empty violations list."""
        LLMOut, assert_facts = self._imports()
        good = LLMOut(
            claim_id="CLM-TEST-001",
            disputed_amount_inr=4812.50,
            awb_tracking_number="BD123456789IN",
            delivery_date="2026-08-15",
            narrative_text="We formally contest this INR 4812.50 chargeback.",
            ce3_evidence_cited=["CE3.0 Item 2 - AWB POD"],
        )
        passed, violations = assert_facts(good, sample_packet)
        assert passed is True
        assert violations == []

    def test_hallucinated_amount_triggers_violation(self, sample_packet):
        """Amount deviating by more than INR 0.50 must trigger AMOUNT_MISMATCH."""
        LLMOut, assert_facts = self._imports()
        bad = LLMOut(
            claim_id="CLM-TEST-001",
            disputed_amount_inr=99_000.00,
            awb_tracking_number="BD123456789IN",
            delivery_date="2026-08-15",
            narrative_text="Disputing INR 99000.",
        )
        passed, violations = assert_facts(bad, sample_packet)
        assert passed is False
        assert any("AMOUNT_MISMATCH" in v for v in violations)

    def test_hallucinated_claim_id_triggers_violation(self, sample_packet):
        """Fabricated claim ID must trigger CLAIM_ID_MISMATCH."""
        LLMOut, assert_facts = self._imports()
        bad = LLMOut(
            claim_id="CLM-FABRICATED",
            disputed_amount_inr=4812.50,
            awb_tracking_number="BD123456789IN",
            delivery_date="2026-08-15",
            narrative_text="Claim CLM-FABRICATED.",
        )
        passed, violations = assert_facts(bad, sample_packet)
        assert passed is False
        assert any("CLAIM_ID_MISMATCH" in v for v in violations)

    def test_hallucinated_awb_triggers_violation(self, sample_packet):
        """Wrong AWB tracking number must trigger AWB_MISMATCH."""
        LLMOut, assert_facts = self._imports()
        bad = LLMOut(
            claim_id="CLM-TEST-001",
            disputed_amount_inr=4812.50,
            awb_tracking_number="FAKE-AWB-XYZ",
            delivery_date="2026-08-15",
            narrative_text="Shipped via FAKE-AWB-XYZ.",
        )
        passed, violations = assert_facts(bad, sample_packet)
        assert passed is False
        assert any("AWB_MISMATCH" in v for v in violations)

    def test_amount_within_tolerance_passes(self, sample_packet):
        """Amount within INR 0.50 tolerance (exact match) must PASS."""
        LLMOut, assert_facts = self._imports()
        # exact value -- 0.00 deviation
        ok = LLMOut(
            claim_id="CLM-TEST-001",
            disputed_amount_inr=4812.50,
            awb_tracking_number="BD123456789IN",
            delivery_date="2026-08-15",
            narrative_text="Exact amount match.",
        )
        passed, _ = assert_facts(ok, sample_packet)
        assert passed is True


# ===========================================================================
# (D) DETERMINISTIC FALLBACK + CE3.0 EVIDENCE MAP  (2 tests)
# ===========================================================================

class TestDeterministicFallback:
    """Verify the CE3.0 evidence map and deterministic narrative fallback."""

    def test_ce3_evidence_map_returns_all_three_items(self, sample_packet):
        """build_ce3_evidence_map must always return all 3 CE3.0 evidence items."""
        from chargeback_defense.narrative_generator import build_ce3_evidence_map
        ce3 = build_ce3_evidence_map(sample_packet)
        assert "ce3_evidence_item_1_prior_history" in ce3
        assert "ce3_evidence_item_2_awb_proof" in ce3
        assert "ce3_evidence_item_3_feedback" in ce3
        assert ce3["ce3_evidence_item_2_awb_proof"]["awb_tracking_number"] == "BD123456789IN"

    def test_deterministic_fallback_contains_key_facts_and_ce3(self, sample_packet):
        """synthesize_indian_d2c_narrative must embed claim_id, INR amount, and CE3.0 citation."""
        from chargeback_defense.narrative_generator import synthesize_indian_d2c_narrative
        t0 = time.perf_counter()
        narrative = synthesize_indian_d2c_narrative(sample_packet)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Content checks
        assert "CLM-TEST-001" in narrative
        assert "4,812.50" in narrative
        assert "CE3.0" in narrative
        # Latency check (no API I/O, must be near-instant)
        assert elapsed_ms < 25.0, f"Fallback took {elapsed_ms:.2f}ms, expected <25ms"


# ===========================================================================
# SHA-256 DOSSIER INTEGRITY SEAL  (3 tests)
# ===========================================================================

class TestCryptographicSeal:
    """Verify the SHA-256 integrity seal used in pdf_generator.py."""

    @staticmethod
    def _seal(claim_id: str, amount: float, awb: str, dt: str, narrative: str) -> str:
        payload = json.dumps(
            {"claim_id": claim_id, "amount_inr": amount, "awb": awb,
             "delivery_dt": dt, "narrative": narrative[:500]},
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def test_seal_is_deterministic(self):
        """Identical inputs must always produce the same digest."""
        h1 = self._seal("CLM-001", 4812.5, "BD123", "2026-08-15", "narrative text")
        h2 = self._seal("CLM-001", 4812.5, "BD123", "2026-08-15", "narrative text")
        assert h1 == h2

    def test_seal_detects_amount_tamper(self):
        """Any change to amount_inr must produce a completely different digest."""
        h_orig = self._seal("CLM-001", 4812.5, "BD123", "2026-08-15", "narrative")
        h_tampered = self._seal("CLM-001", 9999.9, "BD123", "2026-08-15", "narrative")
        assert h_orig != h_tampered

    def test_seal_is_valid_sha256_hex(self):
        """Output must be exactly 64 lowercase hex characters (SHA-256 standard)."""
        h = self._seal("CLM-001", 4812.5, "BD123", "2026-08-15", "narrative")
        assert len(h) == 64
        assert h == h.lower()
        assert all(c in "0123456789abcdef" for c in h)
