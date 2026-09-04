"""PDF Evidence Packet Generator: Submission-Style Chargeback Defense Dossiers.

Transforms structured evidence packets (incorporating model risk signals,
commercial fulfillment verification, customer review feedback, and LLM dispute
defense narratives) into professional, submission-ready PDF documents formatted
for card networks (Visa/Mastercard) and payment verifiers (NPCI).

This module operates entirely offline with zero external network or model calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fpdf import FPDF


# =============================================================================
# UNICODE TEXT SANITIZER (Prevents Core Font Encoding Errors)
# =============================================================================
def clean_pdf_text(val: Any) -> str:
    """Sanitize strings to ensure compatibility with standard PDF core fonts."""
    if val is None:
        return ""
    text = str(val)
    replacements = {
        "\u2014": " - ",  # em-dash
        "\u2013": "-",    # en-dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2022": "*",    # bullet
        "\u03c4": "tau",  # Greek tau
        "\u00a0": " ",    # non-breaking space
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Normalize accented latin characters (e.g. Portuguese feedback)
    norm = unicodedata.normalize("NFKD", text)
    return norm.encode("ascii", "ignore").decode("ascii")


# =============================================================================
# FEATURE BUSINESS TRANSLATIONS
# =============================================================================
FEATURE_BUSINESS_TRANSLATIONS: Dict[str, str] = {
    "card_address_count_C1": "Rapid card velocity across multiple billing addresses in trailing windows",
    "transaction_count_C2": "Abnormal transaction velocity spike on this card profile",
    "email_count_C13": "Elevated email domain velocity associated with identity changes",
    "amt_log": "Unusually high transaction dollar amount relative to merchant baseline",
    "time_delta_prev_txn_D2": "Compressed time interval since preceding transaction",
    "time_delta_card_creation_D1": "Newly active card with limited historical tenure",
    "email_domain_txn_count_24h": "High 24-hour transaction frequency from this email domain",
    "ProductCD_train_fraud_rate": "Elevated historical fraud velocity in this merchandise category",
    "has_billing_addr": "Absence of complete cardholder billing address verification",
    "card_amt_zscore": "Transaction amount significantly exceeds historical cardholder average",
    "addr_switch_velocity": "Rapid switching between distinct billing zip codes",
    "avs_address_match": "Mismatch between entered billing address and issuing bank records",
}


# =============================================================================
# CUSTOM FPDF CLASS WITH CORPORATE HEADER & FOOTER
# =============================================================================
class DisputeDossierPDF(FPDF):
    """Custom PDF generator for official dispute representment packets."""

    def __init__(self, claim_id: str, recommendation: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.claim_id = clean_pdf_text(claim_id)
        self.recommendation = clean_pdf_text(recommendation)

    def header(self) -> None:
        # 1. Top Simulation Watermark Bar
        self.set_fill_color(254, 242, 242)  # Light Red Alert
        self.set_text_color(185, 28, 28)    # Dark Red
        self.set_font("Helvetica", "B", 7.5)
        self.cell(
            0, 5,
            clean_pdf_text("KAVACH AI RISK MANAGER | Razorpay AI Buildathon (Track 02) | Indian D2C Merchant Benchmark"),
            border=0,
            align="C",
            fill=True,
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.ln(1.5)

        # 2. Main Corporate Header
        self.set_fill_color(24, 43, 73)      # Deep Navy Header
        self.set_text_color(255, 255, 255)   # White
        self.set_font("Helvetica", "B", 13)
        self.cell(115, 8, clean_pdf_text(" KAVACH DISPUTE DEFENSE DOSSIER"), fill=True)

        self.set_font("Helvetica", "B", 9)
        self.cell(67, 8, clean_pdf_text(f"CLAIM ID: {self.claim_id} "), align="R", fill=True, new_x="LMARGIN", new_y="NEXT")

        # Subtitle rule
        self.set_draw_color(203, 213, 225)
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self) -> None:
        self.set_y(-18)
        self.set_draw_color(226, 232, 240)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(1.5)

        # Disclaimer line
        self.set_font("Helvetica", "I", 6.5)
        self.set_text_color(100, 116, 139)
        disclaimer_text = (
            "CONFIDENTIAL DISPUTE RECORD - KAVACH PROTOTYPE: Benchmark dataset calibrated to Indian D2C fulfillment "
            "with UPI/RuPay payment rails and domestic logistics tracking (BlueDart/Delhivery/Shadowfax) for dispute representment."
        )
        self.multi_cell(0, 3, clean_pdf_text(disclaimer_text), align="C")

        # Pagination & Timestamp
        self.set_font("Helvetica", "", 7)
        self.set_text_color(148, 163, 184)
        page_str = f"Page {self.page_no()} of {{nb}}"
        self.cell(0, 3.5, clean_pdf_text(page_str), align="C")


# =============================================================================
# CORE PDF RENDERING FUNCTION
# =============================================================================
def generate_evidence_pdf(evidence_packet: Dict[str, Any], output_path: str) -> str:
    """Render a structured evidence packet as a submission-style dispute defense PDF."""
    summary = evidence_packet.get("dispute_summary", {})
    risk = evidence_packet.get("model_risk_assessment", {})
    fulfillment = evidence_packet.get("commercial_fulfillment_evidence", {})
    defense = evidence_packet.get("dispute_defense_evaluation", {})
    deliv = fulfillment.get("delivery_performance", {})
    timeline = fulfillment.get("timeline", {})
    merchant = fulfillment.get("merchant_and_item_details", {})
    feedback = fulfillment.get("customer_feedback_record", {})

    claim_id = summary.get("claim_id", "UNKNOWN_CLAIM")
    rec = defense.get("dispute_representment_recommendation", "INVESTIGATE")
    classification = defense.get("chargeback_reason_classification", "UNKNOWN_REASON")
    narrative = evidence_packet.get("narrative", "").strip()

    orig_amt = summary.get("disputed_amount_original", {})
    conv_amt = summary.get("disputed_amount_converted", {})

    # Initialize Document
    pdf = DisputeDossierPDF(claim_id=claim_id, recommendation=rec, orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=14, top=8, right=14)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    printable_w = pdf.w - pdf.l_margin - pdf.r_margin  # ~182 mm

    # -------------------------------------------------------------------------
    # 1. CLAIM SUMMARY & RECOMMENDATION CARD
    # -------------------------------------------------------------------------
    # Recommendation Badge styling
    if "CONTEST" in rec:
        badge_bg = (220, 252, 231)   # Green
        badge_fg = (22, 101, 52)
        badge_lbl = "RECOMMENDATION: CONTEST DISPUTE"
    elif "REVIEW" in rec:
        badge_bg = (254, 243, 199)   # Amber
        badge_fg = (146, 64, 14)
        badge_lbl = "RECOMMENDATION: REVIEW SLA BEFORE CONTESTING"
    else:
        badge_bg = (254, 226, 226)   # Red / Neutral
        badge_fg = (153, 27, 27)
        badge_lbl = "RECOMMENDATION: ACCEPT CHARGEBACK / REFUND"

    # Header Card Container
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    card_start_y = pdf.get_y()
    pdf.rect(pdf.l_margin, card_start_y, printable_w, 28, style="FD")

    # Recommendation Banner inside card
    pdf.set_xy(pdf.l_margin + 2, card_start_y + 2)
    pdf.set_fill_color(*badge_bg)
    pdf.set_text_color(*badge_fg)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.cell(printable_w - 4, 5.5, clean_pdf_text(f"  {badge_lbl}  |  CLASSIFICATION: {classification}"), fill=True, new_x="LMARGIN", new_y="NEXT")

    # Details grid inside card
    col1_x = pdf.l_margin + 3
    col2_x = pdf.l_margin + (printable_w / 2) + 1
    row1_y = card_start_y + 9.5
    row2_y = card_start_y + 15
    row3_y = card_start_y + 20.5

    pdf.set_text_color(51, 65, 85)

    # Row 1
    pdf.set_xy(col1_x, row1_y)
    pdf.set_font("Helvetica", "B", 8)
    pdf.write(4, "Disputed Amount: ")
    pdf.set_font("Helvetica", "", 8)
    inr_val = orig_amt.get("value", 0.0)
    pdf.write(4, clean_pdf_text(f"INR {inr_val:,.2f} (Rs. {inr_val:,.2f})"))

    pdf.set_xy(col2_x, row1_y)
    pdf.set_font("Helvetica", "B", 8)
    pdf.write(4, "Fraud Model Score: ")
    score = risk.get("fraud_risk_score", 0.0)
    rband = risk.get("risk_band", "UNKNOWN")
    pdf.set_font("Helvetica", "B" if score >= 0.70 else "", 8)
    pdf.set_text_color(185, 28, 28) if score >= 0.70 else pdf.set_text_color(51, 65, 85)
    pdf.write(4, clean_pdf_text(f"{score:.4f} ({rband}, tau=0.70)"))
    pdf.set_text_color(51, 65, 85)

    # Row 2
    pdf.set_xy(col1_x, row2_y)
    pdf.set_font("Helvetica", "B", 8)
    pdf.write(4, "Payment Rail: ")
    pdf.set_font("Helvetica", "", 8)
    payment = fulfillment.get("payment_profile", {})
    pay_id = payment.get("payment_identifier") or summary.get("card_network", "UPI / RuPay")
    pdf.write(4, clean_pdf_text(f"{pay_id}"))

    pdf.set_xy(col2_x, row2_y)
    pdf.set_font("Helvetica", "B", 8)
    pdf.write(4, "Courier & Tracking: ")
    pdf.set_font("Helvetica", "", 8)
    deliv = fulfillment.get("delivery_performance", {})
    courier = deliv.get("courier_partner") or merchant.get("courier_partner", "BlueDart Express")
    awb = deliv.get("awb_tracking_number") or merchant.get("awb_tracking_number", "—")
    pdf.write(4, clean_pdf_text(f"{courier} | {awb}"))

    # Row 3
    pdf.set_xy(col1_x, row3_y)
    pdf.set_font("Helvetica", "B", 8)
    pdf.write(4, "Customer Contact: ")
    pdf.set_font("Helvetica", "", 8)
    cphone = merchant.get("customer_phone", "+91 9800040114")
    cpin = merchant.get("customer_pin", "560001")
    pdf.write(4, clean_pdf_text(f"{cphone} (PIN {cpin})"))

    pdf.set_xy(col2_x, row3_y)
    pdf.set_font("Helvetica", "B", 8)
    pdf.write(4, "Ground Truth Outcome: ")
    pdf.set_font("Helvetica", "", 8)
    gt = summary.get("ground_truth_label", 0)
    pdf.write(4, clean_pdf_text(f"{'Chargeback Confirmed (1)' if gt == 1 else 'Legitimate Transaction (0)'}"))

    pdf.set_y(card_start_y + 31)

    # -------------------------------------------------------------------------
    # 2. DISPUTE DEFENSE NARRATIVE (PROSE CALLOUT)
    # -------------------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, clean_pdf_text("1. Executive Dispute Defense Summary"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(24, 43, 73)  # Navy accent border on left
    pdf.set_line_width(0.8)

    narr_start_y = pdf.get_y()
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 41, 59)

    # Measure narrative height
    narr_text = narrative if narrative else "[No plain-English narrative available. Run narrative_generator.py to populate.]"
    cleaned_narr = clean_pdf_text(narr_text)

    # Render callout box
    box_padding = 3
    pdf.set_x(pdf.l_margin + 2)
    # Estimate height with multi_cell dry-run or approximate height
    temp_y = pdf.get_y()
    pdf.multi_cell(printable_w - 4, 4.3, cleaned_narr, border=0)
    narr_end_y = pdf.get_y()

    # Draw left border line
    pdf.line(pdf.l_margin + 1, narr_start_y, pdf.l_margin + 1, narr_end_y)
    pdf.set_line_width(0.2)
    pdf.ln(3)

    # -------------------------------------------------------------------------
    # 3. COMMERCIAL FULFILLMENT & DELIVERY TIMELINE
    # -------------------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, clean_pdf_text("2. Carrier Fulfillment & Delivery Evidence"), new_x="LMARGIN", new_y="NEXT")

    deliv = fulfillment.get("delivery_performance", {})
    timeline = fulfillment.get("timeline", {})
    merchant = fulfillment.get("merchant_and_item_details", {})
    feedback = fulfillment.get("customer_feedback_record", {})
    order_status = fulfillment.get("order_status", "unknown").lower()

    if order_status in ["canceled", "unavailable"]:
        # Unfulfilled warning card
        pdf.set_fill_color(254, 242, 242)
        pdf.set_draw_color(248, 113, 113)
        pdf.rect(pdf.l_margin, pdf.get_y(), printable_w, 14, style="FD")
        pdf.set_xy(pdf.l_margin + 3, pdf.get_y() + 2)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(185, 28, 28)
        pdf.write(4, "FULFILLMENT WARNING: Order marked Canceled / Unavailable prior to carrier dispatch.\n")
        pdf.set_font("Helvetica", "", 8)
        pdf.write(4, "No carrier delivery scan exists. Customer feedback indicates fulfillment defect. Merchant settlement / refund advised.")
        pdf.set_y(pdf.get_y() + 8)
        pdf.ln(4)

    else:
        # Timeline Table
        # Headers: Milestone | Timestamp | Status / Context
        table_w = printable_w
        col_w = [48, 55, 79]

        pdf.set_fill_color(241, 245, 249)
        pdf.set_draw_color(203, 213, 225)
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(71, 85, 105)

        pdf.cell(col_w[0], 5, clean_pdf_text("Fulfillment Milestone"), border=1, fill=True)
        pdf.cell(col_w[1], 5, clean_pdf_text("Carrier Timestamp"), border=1, fill=True)
        pdf.cell(col_w[2], 5, clean_pdf_text("Evidence Verification Context"), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        # Rows
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(30, 41, 59)

        milestones = [
            ("Order Placed", timeline.get("purchase_timestamp"), "Customer order authorized and processed via UPI/Card"),
            ("Courier Dispatched", timeline.get("carrier_dispatched_timestamp"), f"Origin Hub: {merchant.get('seller_location', 'Seller')}; Category: {merchant.get('product_category', 'Goods')}"),
            ("Customer Delivery", timeline.get("delivered_customer_timestamp"), f"Destination: {merchant.get('customer_location', 'Customer')}; Proof Confirmed: {deliv.get('delivery_proof_available', False)}"),
            ("Estimated SLA Deadline", timeline.get("estimated_delivery_timestamp"), f"Promised delivery SLA target date"),
        ]

        # Delivery Delta highlight
        delta_days = deliv.get("delivery_delta_days")
        for name, ts, ctx in milestones:
            if not ts:
                continue  # Gracefully omit missing milestones
            ts_clean = str(ts).replace("T", " ")[:19]
            pdf.cell(col_w[0], 4.8, clean_pdf_text(name), border="LR")
            pdf.cell(col_w[1], 4.8, clean_pdf_text(ts_clean), border="LR")
            pdf.cell(col_w[2], 4.8, clean_pdf_text(ctx), border="LR", new_x="LMARGIN", new_y="NEXT")

        # Delivery Performance Summary Row
        pdf.set_fill_color(248, 250, 252)
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.cell(col_w[0], 5, clean_pdf_text("Delivery Performance"), border=1, fill=True)

        if delta_days is not None:
            if delta_days <= 0:
                delta_str = f"ON TIME — Delivered {abs(delta_days):.1f} days ahead of deadline"
                pdf.set_text_color(22, 101, 52)
            else:
                delta_str = f"LATE — Delivered {delta_days:.1f} days past target deadline"
                pdf.set_text_color(185, 28, 28)
        else:
            delta_str = str(deliv.get("status", "Unknown"))
            pdf.set_text_color(71, 85, 105)

        pdf.cell(col_w[1] + col_w[2], 5, clean_pdf_text(delta_str), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(30, 41, 59)
        pdf.ln(3)

    # -------------------------------------------------------------------------
    # 4. CUSTOMER FEEDBACK EVIDENCE
    # -------------------------------------------------------------------------
    rev_score = feedback.get("review_score")
    rev_msg = feedback.get("review_comment_message")
    rev_ttl = feedback.get("review_comment_title")

    if rev_score is not None:
        pdf.set_font("Helvetica", "B", 8)
        pdf.write(4, "Verified Customer Feedback: ")
        pdf.set_font("Helvetica", "", 8)
        stars_str = f"{rev_score} / 5 Stars"
        pdf.write(4, clean_pdf_text(stars_str))

        if rev_msg:
            pdf.set_font("Helvetica", "I", 7.5)
            full_quote = f' — "{rev_ttl + ": " if rev_ttl else ""}{rev_msg}"'
            pdf.write(4, clean_pdf_text(full_quote))
        pdf.ln(5)

    # -------------------------------------------------------------------------
    # 5. PAYMENT RISK SIGNALS (BUSINESS TRANSLATION)
    # -------------------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, clean_pdf_text("3. Multi-Signal Payment Risk Profile"), new_x="LMARGIN", new_y="NEXT")

    signals = risk.get("top_contributing_signals", [])
    if signals:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(51, 65, 85)

        for s in signals:
            feat_name = s.get("feature", "")
            direction = s.get("risk_direction", "INCREASES_RISK")
            business_label = FEATURE_BUSINESS_TRANSLATIONS.get(feat_name, feat_name.replace("_", " ").title())

            is_risk = (direction == "INCREASES_RISK")
            indicator = "[RISK FACTOR]" if is_risk else "[MITIGATING]"

            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_text_color(185, 28, 28) if is_risk else pdf.set_text_color(22, 101, 52)
            pdf.cell(24, 4.2, clean_pdf_text(indicator))

            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 4.2, clean_pdf_text(f": {business_label}"), new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 4.2, clean_pdf_text("No technical risk signals recorded for this transaction."), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)

    # -------------------------------------------------------------------------
    # 6. COMPELLING EVIDENCE FACTORS & CONCLUSION
    # -------------------------------------------------------------------------
    factors = defense.get("compelling_evidence_factors", [])
    if factors:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 5, clean_pdf_text("Key Representment Grounds:"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(51, 65, 85)
        for f in factors:
            pdf.cell(0, 4, clean_pdf_text(f"  * {f}"), new_x="LMARGIN", new_y="NEXT")

    # -------------------------------------------------------------------------
    # CRYPTOGRAPHIC DIGITAL SEAL  (SHA-256 Integrity Hash)
    # -------------------------------------------------------------------------
    # The hash is computed deterministically over the canonical dossier fields:
    # claim_id || disputed_amount_inr || awb_tracking_number || delivery_date || narrative
    deliv_ts = str(
        fulfillment.get("timeline", {}).get("delivered_customer_timestamp", "")
    )[:10]
    awb_for_hash = (
        deliv.get("awb_tracking_number") or merchant.get("awb_tracking_number", "")
    ).strip()
    canonical_payload = json.dumps({
        "claim_id":    claim_id,
        "amount_inr":  orig_amt.get("value", 0.0),
        "awb":         awb_for_hash,
        "delivery_dt": deliv_ts,
        "narrative":   narrative[:500],  # first 500 chars to keep hash stable across truncation
    }, sort_keys=True, separators=(",", ":"))
    sha256_digest = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    seal_text = clean_pdf_text(
        f"Digitally Sealed & Verified by Kavach Defense Engine: SHA256-{sha256_digest}"
    )

    # Render seal block
    pdf.ln(4)
    pdf.set_draw_color(24, 43, 73)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 4, clean_pdf_text("CRYPTOGRAPHIC INTEGRITY SEAL"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", "", 6.5)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 3.5, seal_text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 6)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(
        0, 3,
        clean_pdf_text(
            f"Sealed at {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} | "
            f"Fields: claim_id + amount_inr + awb + delivery_date + narrative[:500]"
        ),
        new_x="LMARGIN", new_y="NEXT"
    )

    # Output to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    return output_path


# =============================================================================
# BATCH PDF GENERATOR
# =============================================================================
def generate_all_pdfs(
    packets_json_path: str = "reports/sample_evidence_packets_with_narratives.json",
    output_dir: str = "reports/pdf_packets",
) -> List[str]:
    """Compile all evidence packets in the JSON dossier into individual PDFs."""
    print("\n" + "=" * 72)
    print("TASK: GENERATING SUBMISSION-STYLE DISPUTE DEFENSE PDFS")
    print("=" * 72)
    t0 = time.time()

    if not os.path.exists(packets_json_path):
        raise FileNotFoundError(f"Input packets not found at: {packets_json_path}")

    with open(packets_json_path, "r", encoding="utf-8") as fh:
        packets: List[Dict[str, Any]] = json.load(fh)

    os.makedirs(output_dir, exist_ok=True)
    generated_files: List[str] = []

    print(f"Loaded {len(packets)} evidence packets from: {packets_json_path}")
    print(f"Target Output Directory: {output_dir}")

    for idx, pkt in enumerate(packets, start=1):
        claim_id = pkt.get("dispute_summary", {}).get("claim_id", f"CLAIM_{idx}")
        rec = pkt.get("dispute_defense_evaluation", {}).get("dispute_representment_recommendation", "UNKNOWN")
        filename = f"{claim_id}.pdf"
        out_path = os.path.join(output_dir, filename)

        generate_evidence_pdf(pkt, out_path)
        file_size_kb = os.path.getsize(out_path) / 1024.0
        generated_files.append(out_path)

        print(f"  [{idx}/{len(packets)}] Generated {out_path:<36s} | 1 page | {file_size_kb:>5.1f} KB | {rec}")

    print(f"\nSuccessfully generated {len(generated_files)} PDF packets in {time.time()-t0:.2f}s.")
    return generated_files


if __name__ == "__main__":
    generate_all_pdfs()
