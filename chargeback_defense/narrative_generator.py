"""LLM Narrative Generation Layer: Plain-English Dispute Defense Summaries.

Transforms structured evidence packets into concise, dispute-ready defense
narratives tailored for card networks (Visa/Mastercard) and payment verifiers (NPCI).

Supports:
- Google Gemini (gemini-2.5-flash / gemini-1.5-flash) via google.genai SDK
- Anthropic Claude (claude-3-5-sonnet / claude-3-5-haiku) via anthropic SDK

This module operates strictly downstream from real-time scoring, ensuring zero
impact on core transaction evaluation latency.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import dotenv

dotenv.load_dotenv()

# Optional SDK imports
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore
    genai_types = None  # type: ignore

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


# =============================================================================
# PRICING CONFIGURATION (per million tokens)
# =============================================================================
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # Google Gemini
    "gemini-2.5-flash": {"input_per_m": 0.075, "output_per_m": 0.30},
    "gemini-1.5-flash": {"input_per_m": 0.075, "output_per_m": 0.30},
    "gemini-2.5-pro": {"input_per_m": 1.25, "output_per_m": 5.00},
    # Anthropic Claude
    "claude-3-5-sonnet-20241022": {"input_per_m": 3.00, "output_per_m": 15.00},
    "claude-3-5-haiku-20241022": {"input_per_m": 0.80, "output_per_m": 4.00},
    "claude-3-haiku-20240307": {"input_per_m": 0.25, "output_per_m": 1.25},
}


def get_active_provider_and_model() -> Tuple[str, str, Optional[str]]:
    """Determine LLM provider, model name, and API key based on environment."""
    explicit_provider = os.getenv("LLM_PROVIDER", "").lower().strip()

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if explicit_provider == "anthropic" or (not explicit_provider and anthropic_key and not gemini_key):
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        return "anthropic", model, anthropic_key

    # Default to Gemini if GEMINI_API_KEY / GOOGLE_API_KEY present or requested
    if explicit_provider == "gemini" or gemini_key:
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return "gemini", model, gemini_key

    # If neither is set, check if anthropic key is configured
    if anthropic_key:
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        return "anthropic", model, anthropic_key

    # Fallback default
    return "gemini", "gemini-2.5-flash", None


# =============================================================================
# FEATURE TRANSLATION HELPER (No Jargon in Dossiers)
# =============================================================================
FEATURE_BUSINESS_TRANSLATIONS: Dict[str, str] = {
    "card_address_count_C1": "rapid card velocity across multiple billing addresses in trailing windows",
    "transaction_count_C2": "abnormal transaction velocity spike on this card profile",
    "email_count_C13": "elevated email domain velocity associated with identity changes",
    "amt_log": "unusually high transaction dollar amount relative to merchant baseline",
    "time_delta_prev_txn_D2": "compressed time interval since preceding transaction",
    "time_delta_card_creation_D1": "newly active card with limited historical tenure",
    "email_domain_txn_count_24h": "high 24-hour transaction frequency from this email domain",
    "ProductCD_train_fraud_rate": "elevated historical fraud velocity in this merchandise category",
    "has_billing_addr": "absence of complete cardholder billing address verification",
    "card_amt_zscore": "transaction amount significantly exceeds historical cardholder average",
    "addr_switch_velocity": "rapid switching between distinct billing zip codes",
    "avs_address_match": "mismatch between entered billing address and issuing bank records",
}


def translate_signal_to_business(feature_name: str, risk_direction: str) -> str:
    """Translate technical feature names into plain payment-risk terminology."""
    desc = FEATURE_BUSINESS_TRANSLATIONS.get(
        feature_name,
        feature_name.replace("_", " ").title(),
    )
    if risk_direction == "INCREASES_RISK":
        return f"Risk Indicator: {desc}"
    return f"Mitigating Factor: {desc}"


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================
SYSTEM_PROMPT = """You are a senior payment-risk analyst drafting a professional, authoritative chargeback defense summary for submission to card networks (Visa, Mastercard) and dispute verifiers (NPCI).

Your objective is to produce a concise, plain-English dispute defense narrative (under 200 words) based strictly on the provided structured evidence packet.

CRITICAL INSTRUCTIONS:
1. Business Language: Never output technical machine-learning jargon or SHAP values. Do NOT mention fraud scores, risk bands, risk signals, model outputs, or any ML-system terminology. The narrative is an external document submitted to the card network on behalf of the merchant — it must present only commercial fulfillment evidence.
2. Concrete Evidence: You MUST cite specific fulfillment facts from the evidence packet (actual delivery date, whether delivery was on-time or delayed by how many days, review star rating, customer feedback text, payment method, and item category). Never write generic boilerplate.
3. Strict Factual Fidelity: Do NOT fabricate or assume any details not present in the packet. If a field is null or missing (e.g. missing customer comment), omit it gracefully.
4. Risk Signal Embargo: Do NOT reference the fraud_risk_score, key_risk_signals, risk_band, contributing features, or any internal model assessment in the external narrative — even if these fields appear in the packet. These are internal analyst data ONLY and must NEVER appear in the dispute submission text, as revealing that the merchant's own system flagged the transaction as suspicious would directly undermine the contest.
5. Tone Alignment:
   - If recommendation is 'CONTEST_CHARGEBACK_WITH_EVIDENCE': Adopt an assertive, evidence-led tone demonstrating that the customer received the merchandise as ordered and confirmed receipt with positive feedback (Friendly Fraud / First-Party Fraud).
   - If recommendation is 'ACCEPT_CHARGEBACK_OR_ISSUE_REFUND': Adopt an objective, transparent tone noting the merchant fulfillment defect (e.g. unfulfilled order, canceled status, or wrong item) and advising against contestation.
   - If recommendation is 'REVIEW_SHIPPING_SLA_BEFORE_REPRESENTMENT': Adopt a balanced, prudent tone acknowledging that carrier delivery occurred but exceeded the promised delivery SLA, advising merchant SLA review prior to representment. Do NOT mention any fraud or risk characteristics.
6. Length: Keep the entire narrative strictly under 200 words. Output ONLY the narrative text with no conversational preamble or meta-commentary.
"""


def build_user_prompt(packet: Dict[str, Any]) -> str:
    """Format an evidence packet into a clear, structured prompt for the LLM."""
    summary = packet.get("dispute_summary", {})
    risk = packet.get("model_risk_assessment", {})
    fulfillment = packet.get("commercial_fulfillment_evidence", {})
    timeline = fulfillment.get("timeline", {})
    deliv = fulfillment.get("delivery_performance", {})
    feedback = fulfillment.get("customer_feedback_record", {})
    merchant = fulfillment.get("merchant_and_item_details", {})
    payment = fulfillment.get("payment_profile", {})
    defense = packet.get("dispute_defense_evaluation", {})

    orig_amt = summary.get("disputed_amount_original", {})
    conv_amt = summary.get("disputed_amount_converted", {})

    signals = risk.get("top_contributing_signals", [])
    translated_signals = [
        translate_signal_to_business(s.get("feature", ""), s.get("risk_direction", ""))
        for s in signals
    ]

    # NOTE: fraud_risk_score and key_risk_signals are intentionally excluded from the
    # LLM context. These are internal analyst data only. Including them in the external
    # dispute narrative would reveal that the merchant's own system flagged the transaction
    # as suspicious, which would directly undermine the contest position.
    context = {
        "claim_id": summary.get("claim_id"),
        "disputed_amount": f"${orig_amt.get('value', 0):.2f} {orig_amt.get('currency', 'USD')} (Converted: R$ {conv_amt.get('value', 0):.2f} {conv_amt.get('currency', 'BRL')})",
        "card_profile": f"{summary.get('card_network', 'unknown').upper()} {summary.get('card_type', 'card')} (Product code: {summary.get('product_category_code', 'standard')})",
        "matched_order_id": fulfillment.get("matched_order_id"),
        "commercial_order_value": f"R$ {fulfillment.get('matched_order_value_brl', 0):.2f} BRL",
        "order_status": fulfillment.get("order_status"),
        "item_details": f"{merchant.get('item_count', 1)} item(s) in category '{merchant.get('product_category', 'merchandise')}' from seller in {merchant.get('seller_location', 'unknown')}",
        "customer_location": merchant.get("customer_location", "unknown"),
        "payment_method": f"{payment.get('payment_type', 'card')} with {payment.get('installments', 1)} installment(s)",
        "delivery_proof_available": deliv.get("delivery_proof_available", False),
        "delivery_status": deliv.get("status"),
        "delivery_timeline": {
            "purchased": timeline.get("purchase_timestamp"),
            "carrier_dispatched": timeline.get("carrier_dispatched_timestamp"),
            "delivered_to_customer": timeline.get("delivered_customer_timestamp"),
            "estimated_deadline": timeline.get("estimated_delivery_timestamp"),
        },
        "delivery_delta_days": deliv.get("delivery_delta_days"),
        "transit_duration_days": deliv.get("transit_duration_days"),
        "customer_review_score": feedback.get("review_score"),
        "customer_review_comment": feedback.get("review_comment_message"),
        "compelling_defense_factors": defense.get("compelling_evidence_factors", []),
        "dispute_recommendation": defense.get("dispute_representment_recommendation"),
        "chargeback_classification": defense.get("chargeback_reason_classification"),
    }

    return f"""Please draft the official dispute defense narrative for the following evidence dossier:

{json.dumps(context, indent=2)}

Draft the narrative under 200 words now:"""


# =============================================================================
# NARRATIVE GENERATOR FUNCTION
# =============================================================================
def generate_narrative(
    evidence_packet: Dict[str, Any],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = 2,
) -> Tuple[str, Dict[str, int]]:
    """Generate a plain-English dispute defense narrative using Gemini or Claude API.

    Args:
        evidence_packet: Full structured evidence packet dict.
        provider: Optional 'gemini' or 'anthropic' override.
        model: Optional model name override.
        api_key: Optional API key override.
        max_retries: Maximum number of retry attempts on transient API errors.

    Returns:
        Tuple of (narrative_text, usage_dict) where usage_dict has input_tokens and output_tokens.
    """
    detected_provider, detected_model, detected_key = get_active_provider_and_model()
    active_provider = provider or detected_provider
    active_model = model or detected_model
    active_key = api_key or detected_key

    if not active_key:
        error_msg = (
            f"[API KEY MISSING: Please set GEMINI_API_KEY or ANTHROPIC_API_KEY in your environment "
            f"or .env file to generate live LLM dispute narratives with {active_provider}.]"
        )
        print(f"Warning: {error_msg}", file=sys.stderr)
        return error_msg, {"input_tokens": 0, "output_tokens": 0}

    user_prompt = build_user_prompt(evidence_packet)
    backoff = 2.0

    # -------------------------------------------------------------------------
    # 1. GOOGLE GEMINI EXECUTION
    # -------------------------------------------------------------------------
    if active_provider == "gemini":
        if genai is None:
            return (
                "[DEPENDENCY MISSING: google-genai package is not installed. Run 'pip install google-genai'.]",
                {"input_tokens": 0, "output_tokens": 0},
            )
        client = genai.Client(api_key=active_key)
        config = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=2048,
        )

        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=active_model,
                    contents=user_prompt,
                    config=config,
                )
                narrative = (response.text or "").strip()
                in_tok = 0
                out_tok = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    in_tok = response.usage_metadata.prompt_token_count or 0
                    out_tok = response.usage_metadata.candidates_token_count or 0

                return narrative, {"input_tokens": in_tok, "output_tokens": out_tok}

            except Exception as err:
                if attempt < max_retries and ("429" in str(err) or "timeout" in str(err).lower() or "503" in str(err)):
                    print(f"Transient Gemini error ({err}). Retrying in {backoff:.1f}s (attempt {attempt+1}/{max_retries})...", file=sys.stderr)
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    return f"[GEMINI API ERROR: {err}]", {"input_tokens": 0, "output_tokens": 0}

    # -------------------------------------------------------------------------
    # 2. ANTHROPIC CLAUDE EXECUTION
    # -------------------------------------------------------------------------
    elif active_provider == "anthropic":
        if anthropic is None:
            return (
                "[DEPENDENCY MISSING: anthropic package is not installed. Run 'pip install anthropic'.]",
                {"input_tokens": 0, "output_tokens": 0},
            )
        client = anthropic.Anthropic(api_key=active_key)

        for attempt in range(max_retries + 1):
            try:
                response = client.messages.create(
                    model=active_model,
                    max_tokens=400,
                    temperature=0.2,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                narrative = response.content[0].text.strip()
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                return narrative, usage

            except (
                getattr(anthropic, "RateLimitError", Exception),
                getattr(anthropic, "APITimeoutError", Exception),
                getattr(anthropic, "APIConnectionError", Exception),
                getattr(anthropic, "InternalServerError", Exception),
            ) as transient_err:
                if attempt < max_retries:
                    print(f"Transient Claude error ({transient_err}). Retrying in {backoff:.1f}s (attempt {attempt+1}/{max_retries})...", file=sys.stderr)
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    return f"[ANTHROPIC API RETRY EXHAUSTED: {transient_err}]", {"input_tokens": 0, "output_tokens": 0}
            except Exception as err:
                return f"[ANTHROPIC API ERROR: {err}]", {"input_tokens": 0, "output_tokens": 0}

    return f"[UNKNOWN PROVIDER: {active_provider}]", {"input_tokens": 0, "output_tokens": 0}


# =============================================================================
# BATCH RUNNER & REPORT GENERATOR
# =============================================================================
def run_batch_narrative_generation(
    input_path: str = "reports/sample_evidence_packets.json",
    output_path: str = "reports/sample_evidence_packets_with_narratives.json",
    md_output_path: str = "reports/narrative_examples.md",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load sample evidence packets, generate narratives for all cases, and export results."""
    print("\n" + "=" * 72)
    print("TASK: LLM NARRATIVE GENERATION LAYER")
    print("=" * 72)
    t0 = time.time()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input evidence packets not found at: {input_path}")

    with open(input_path, "r", encoding="utf-8") as fh:
        packets: List[Dict[str, Any]] = json.load(fh)

    active_provider, active_model, active_key = get_active_provider_and_model()
    if provider: active_provider = provider
    if model: active_model = model
    if api_key: active_key = api_key

    print(f"Loaded {len(packets)} sample evidence packets from: {input_path}")
    print(f"Active LLM Provider: {active_provider.upper()} ({active_model})")
    print(f"API Key Configured: {'Yes (valid key found)' if active_key else 'No (missing)'}")

    total_input_tokens = 0
    total_output_tokens = 0
    pricing = MODEL_PRICING.get(active_model, {"input_per_m": 0.075, "output_per_m": 0.30})

    enriched_packets: List[Dict[str, Any]] = []

    for idx, pkt in enumerate(packets, start=1):
        claim_id = pkt.get("dispute_summary", {}).get("claim_id", f"Case {idx}")
        rec = pkt.get("dispute_defense_evaluation", {}).get("dispute_representment_recommendation", "UNKNOWN")
        print(f"\nProcessing Case {idx} ({claim_id}) | Recommendation: {rec}...")

        narrative, usage = generate_narrative(
            pkt,
            provider=active_provider,
            model=active_model,
            api_key=active_key,
        )
        pkt["narrative"] = narrative
        enriched_packets.append(pkt)

        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        print(f"  Generated narrative ({len(narrative.split())} words, {in_tok} in / {out_tok} out tokens).")

    # Calculate token costs
    batch_cost = (
        (total_input_tokens / 1_000_000.0) * pricing["input_per_m"]
        + (total_output_tokens / 1_000_000.0) * pricing["output_per_m"]
    )

    # Scale estimation for all 1,613 high-risk claims
    scale_claims = 1613
    avg_in = (total_input_tokens / len(packets)) if (packets and total_input_tokens > 0) else 550
    avg_out = (total_output_tokens / len(packets)) if (packets and total_output_tokens > 0) else 175
    scale_in_tok = avg_in * scale_claims
    scale_out_tok = avg_out * scale_claims
    scale_cost = (
        (scale_in_tok / 1_000_000.0) * pricing["input_per_m"]
        + (scale_out_tok / 1_000_000.0) * pricing["output_per_m"]
    )

    # Save enriched JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(enriched_packets, fh, indent=2)
    print(f"\nSaved enriched evidence packets with narratives to: {output_path}")

    # Generate Markdown Summary
    write_narrative_markdown_report(
        enriched_packets=enriched_packets,
        md_output_path=md_output_path,
        provider=active_provider,
        model=active_model,
        batch_input_tokens=total_input_tokens,
        batch_output_tokens=total_output_tokens,
        batch_cost=batch_cost,
        scale_claims=scale_claims,
        scale_cost=scale_cost,
    )
    print(f"Saved human-readable narrative report to: {md_output_path}")

    print("\n" + "-" * 72)
    print("BATCH TOKEN & COST SUMMARY")
    print("-" * 72)
    print(f"Provider & Model:           {active_provider.upper()} ({active_model})")
    print(f"Total Batch Input Tokens:   {total_input_tokens:,}")
    print(f"Total Batch Output Tokens:  {total_output_tokens:,}")
    print(f"Total Batch Cost (4 Cases): ${batch_cost:.4f} USD")
    print(f"Projected Cost for All {scale_claims:,} Flagged Claims: ${scale_cost:.2f} USD")
    print(f"Completed in {time.time()-t0:.2f}s.")

    return enriched_packets


def write_narrative_markdown_report(
    enriched_packets: List[Dict[str, Any]],
    md_output_path: str,
    provider: str,
    model: str,
    batch_input_tokens: int,
    batch_output_tokens: int,
    batch_cost: float,
    scale_claims: int,
    scale_cost: float,
) -> None:
    """Generate reports/narrative_examples.md with side-by-side case reviews."""
    lines: List[str] = [
        "# Chargeback Evidence Responder: LLM Dispute Defense Narratives",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Provider & Model:** `{provider.upper()}` (`{model}`)  ",
        "**Status:** Downstream Async LLM Layer · Defense-Ready Dossiers",
        "",
        "---",
        "",
        "## 1. Executive Summary & Cost Analytics",
        "",
        "The **Evidence-Narrative Generation Layer** translates structured multi-modal evidence packets (incorporating IEEE-CIS fraud signals, TreeSHAP feature importance, Olist carrier fulfillment timestamps, and customer reviews) into plain-English dispute representment summaries.",
        "",
        "| Metric | Batch Run (4 Cases) | Full Rollout (1,613 Flagged Claims) |",
        "| :--- | :--- | :--- |",
        f"| **Input Tokens** | {batch_input_tokens:,} | ~{int(scale_claims * (batch_input_tokens/max(1, len(enriched_packets)) if batch_input_tokens > 0 else 550)):,} |",
        f"| **Output Tokens** | {batch_output_tokens:,} | ~{int(scale_claims * (batch_output_tokens/max(1, len(enriched_packets)) if batch_output_tokens > 0 else 175)):,} |",
        f"| **Estimated Cost** | **${batch_cost:.4f} USD** | **~${scale_cost:.2f} USD** |",
        "",
        "> [!TIP]",
        f"> At **~${scale_cost:.2f} USD** to generate professional dispute dossiers for all 1,613 high-risk claims, automated narrative generation costs under $0.005 per dispute—representing an overwhelming ROI compared to human analyst drafting costs ($15–$30/case).",
        "",
        "---",
        "",
        "## 2. Generated Case Narratives (Side-by-Side Review)",
        "",
    ]

    for idx, pkt in enumerate(enriched_packets, start=1):
        summary = pkt.get("dispute_summary", {})
        defense = pkt.get("dispute_defense_evaluation", {})
        fulfillment = pkt.get("commercial_fulfillment_evidence", {})
        deliv = fulfillment.get("delivery_performance", {})
        feedback = fulfillment.get("customer_feedback_record", {})
        risk = pkt.get("model_risk_assessment", {})

        orig_amt = summary.get("disputed_amount_original", {})
        conv_amt = summary.get("disputed_amount_converted", {})
        rec = defense.get("dispute_representment_recommendation", "UNKNOWN")
        cls = defense.get("chargeback_reason_classification", "UNKNOWN")

        lines.extend([
            f"### Case {idx}: {summary.get('claim_id')} — `{rec}`",
            "",
            f"- **Chargeback Classification:** `{cls}`",
            f"- **Disputed Amount:** ${orig_amt.get('value', 0):.2f} USD (Normalized: R$ {conv_amt.get('value', 0):.2f} BRL @ {conv_amt.get('fx_rate_used', 3.5)} FX)",
            f"- **Matched Commercial Order:** `{fulfillment.get('matched_order_id')}` (R$ {fulfillment.get('matched_order_value_brl', 0):.2f} BRL, {fulfillment.get('amount_match_delta_pct', 0):.2f}% delta)",
            f"- **Model Fraud Risk Score:** `{risk.get('fraud_risk_score', 0):.4f}` ({risk.get('risk_band', 'UNKNOWN')})",
            f"- **Fulfillment Outcome:** {deliv.get('status')} ({abs(deliv.get('delivery_delta_days') or 0):.1f} days {'early' if (deliv.get('delivery_delta_days') or 0) <= 0 else 'late'})",
            f"- **Customer Feedback:** {feedback.get('review_score') or 'N/A'} Stars ({'Written comment present' if feedback.get('has_written_feedback') else 'Rating only'})",
            "",
            "#### Generated Dispute Defense Narrative:",
            "> " + pkt.get("narrative", "").replace("\n", "\n> "),
            "",
            "---",
            "",
        ])

    os.makedirs(os.path.dirname(md_output_path), exist_ok=True)
    with open(md_output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    run_batch_narrative_generation()
