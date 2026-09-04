# Ring Sentinel — Shared Contract

Locked at Task 0.4. Every track consumes/produces exactly these names and values.

## Claim feature schema (Track B outputs, C & D consume)

Exact 10 column names, in order:

1. `identity_order_count_so_far`
2. `identity_merchant_count_so_far`
3. `identity_claim_count_so_far`
4. `identity_claim_approval_ratio_so_far`
5. `shared_infra_neighbor_count`
6. `cluster_size`
7. `cluster_merchant_span`
8. `cluster_claim_burst_7d`
9. `reason_text_reuse_flag`
10. `amount`

## Action vocabulary (exactly three values)

- `AUTO_APPROVE`
- `STEP_UP_VERIFICATION`
- `HOLD_PAYOUT_HUMAN_REVIEW`

## Thresholds (tunable, but locked as starting values)

- `HIGH = 0.85` → `HOLD_PAYOUT_HUMAN_REVIEW`
- `MEDIUM = 0.50` → `STEP_UP_VERIFICATION`
- below `MEDIUM` → `AUTO_APPROVE`

## Key data facts (Tracks A/B/C/D must agree)

- `START_DATE = 2026-01-01`, window = 182 days.
- Populations: ~20,000 legit (~15% power shoppers), ~800 camouflage (`CAMO_` prefix on `identity_key`), ~591 ring identities across 70 rings (sizes 3–14).
- Labels: `is_ring_label` on claims; camouflage is legitimate (label 0) but tracked via the `CAMO_` prefix.
- Temporal split: train months 1–4, val month 5, test month 6 — by day, never random.
