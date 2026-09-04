# Competitive Analysis: Docket-Risk (Internal Research Report)

**Date:** September 4, 2026  
**Target Repository:** `Kavach-Team/Kavach` (internal project name: `Ring Sentinel`)  
**Hackathon Track:** Razorpay AI Buildathon 2026 — Track 02: *AI Risk Manager ("Stop the merchant losing money to fraud, returns and chargebacks")*  
**Document Purpose:** Strategic architectural analysis, gap identification, and capability verification to establish clear differentiation for a new, independent competitor project.

---

## 1. What It Actually Does (Grounded in Real Code)

Despite extensive marketing in `README.md` and `DOCKET_COMPLETE_SYSTEM_DOCUMENTATION.md`, an audit of the Python backend in [`src/`](file:///d:/Docket-Risk/src) reveals that Docket-Risk is essentially an **in-memory Disjoint Set (Union-Find) graph clusterer coupled with a monotonic XGBoost binary classifier and a SQLite case management store**.

### 1.1 Ingestion-to-Decision Pipeline Tracing

The runtime operates across two primary synchronous execution paths in [`src/score_service.py`](file:///d:/Docket-Risk/src/score_service.py):

```
                       [ Incoming HTTP Request ]
                                   │
                                   ▼
                   [ request_context_middleware ]
                     • Generates X-Request-ID (UUID4)
                     • Measures endpoint latency
                                   │
                                   ▼
                     [ body_size_middleware ]
                     • Enforces 64KB max body limit
                                   │
                                   ▼
                         [ auth_guard ]
                     • Timing-safe X-API-Key check (hmac.compare_digest)
                     • In-memory sliding-window rate limit (600 req/min)
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   POST /v1/ingest/order                     POST /v1/score
              │                                         │
              ▼                                         ▼
   [ Idempotency Dedup ]                    [ Idempotency Dedup ]
   • LRU cache (_seen_orders)               • LRU cache (_seen_claims)
              │                                         │
              ▼                                         ▼
    [ state.ingest_order ]                 [ state.compute_features ]
    Under RLock (atomic):                  Under RLock (atomic critical section):
    • Checks max_nodes ceiling (2M)        • Path compression: find root
    • _union(identity, dev_)               • Computes 10 graph & claim features
    • _union(identity, vpa_)               • Predictor runs INSIDE lock:
    • _union(identity, ph_)                    XGBoost predict_proba
    • _union(identity, adr_)               • If not shadow: records claim to
    • _union(identity, card_)                  cluster_claims & reason_users
    • Increments order & merchant          • Applies policy thresholds:
      span counters                            >= 0.85 ➔ HOLD
                                               >= 0.50 ➔ STEP_UP
                                               <  0.50 ➔ AUTO_APPROVE
                                                        │
                                                        ▼
                                             [ Audit & Side Effects ]
                                             • SQLite decisions table insert
                                             • SSE event emission (/v1/stream)
                                             • _check_ring_forming detector
                                             • Alert rules evaluation & webhook
```

#### Step 1: Order Ingestion ([`src/score_service.py:1081-1123`](file:///d:/Docket-Risk/src/score_service.py#L1081-L1123))
1. Handler `_do_ingest` extracts the payload.
2. Checks an in-memory `OrderedDict` (`_seen_orders`, capped at 50,000 items) for deduplication.
3. Acquires `state._lock` (`threading.RLock`).
4. Checks that total nodes do not exceed `SETTINGS.max_nodes` (default: 2,000,000). Raises `CapacityError` (HTTP 500) if exceeded.
5. Unifies the `identity_key` with each of the 5 provided infrastructure tokens (`device_id`, `vpa_id`, `phone_id`, `address_id`, `card_id`) using classical union-find with rank balancing.
6. Increments `identity_order_counts[identity_key]` and updates `cluster_merchants`.
7. Returns `IngestOut(status="ingested", order_id=..., known_identities=..., deduplicated=bool)`.

#### Step 2: Claim Scoring & Decisioning ([`src/score_service.py:1126-1258`](file:///d:/Docket-Risk/src/score_service.py#L1126-L1258))
1. Handler `_do_score` checks the `_seen_claims` idempotency cache.
2. Clamps timestamps: if the claim's `ts` is more than 60 seconds into the future, it is clamped to `now_utc` to prevent future-timestamp velocity evasion.
3. Invokes `state.compute_features` under a single acquisition of `state._lock`:
   - Runs `_find(identity_key)` to identify the cluster root.
   - Calculates 10 features: historical order/claim counts, approval ratio, cluster size, shared infrastructure neighbor count, merchant span, 7-day rolling claim burst (`cluster_claim_burst_7d`), reason text reuse flag (`reason_text_reuse_flag`), and monetary amount.
   - Evaluates the model prediction (`predict_score`) **inside the lock** so concurrent claims cannot observe half-recorded cluster state.
   - If not in `shadow` mode, mutates `state.cluster_claims[root]` and increments `identity_claim_stats`.
4. Evaluates the policy cutoff:
   - $\text{score} \ge 0.85 \implies \text{HOLD\_PAYOUT\_HUMAN\_REVIEW}$
   - $\text{score} \ge 0.50 \implies \text{STEP\_UP\_VERIFICATION}$
   - $\text{score} < 0.50 \implies \text{AUTO\_APPROVE}$
5. If the model is missing or fails during evaluation, it **fails open** to `AUTO_APPROVE` with `degraded=True`.
6. Side effects (synchronous):
   - Inserts record into SQLite audit table `decisions`.
   - Emits event into in-memory subscriber queues for SSE (`/v1/stream`).
   - Evaluates `alert_rules` and triggers fire-and-forget daemon thread HTTP webhooks.
   - Runs `_check_ring_forming`: checks if the cluster gained $\ge 3$ members in 30 minutes; if so, emits a high-severity alert.
7. Returns `ScoreOut`.

---

### 1.2 Comprehensive API Endpoint Catalog & Actual Schemas

Every endpoint defined in [`src/score_service.py`](file:///d:/Docket-Risk/src/score_service.py) was inspected:

| Category | Method & Route | Auth Required | Input Schema | Output Schema / Description |
| :--- | :--- | :---: | :--- | :--- |
| **System** | `GET /health`, `GET /healthz` | No | None | `{"status": "ok", "model_loaded": bool, "model_sha_verified": bool, "known_identities": int}` |
| **System** | `GET /readyz` | No | None | `{"ready": True, "model_loaded": bool, "known_identities": int}` |
| **System** | `GET /version` | No | None | `{"service": str, "version": str, "model_sha256": str, "thresholds": dict}` |
| **System** | `GET /metrics` | No | None | Standard Prometheus text metrics (`CONTENT_TYPE_LATEST`) |
| **System** | `GET /models-static/eval_report.json` | No | None | Direct file response of `models/eval_report.json` |
| **System** | `GET /models-static/gnn_report.json` | No | None | Direct file response of optional GNN report |
| **Inference** | `POST /v1/ingest/order`, `POST /ingest/order` | **Yes** | `OrderIn`: `order_id`, `identity_key`, `merchant_id`, `device_id`, `vpa_id`, `phone_id`, `address_id`, `card_id`, `ts?` | `IngestOut`: `status`, `order_id`, `known_identities`, `deduplicated` |
| **Inference** | `POST /v1/score`, `POST /score` | **Yes** | `ClaimIn`: `claim_id?`, `identity_key`, `merchant_id`, `amount`, `reason_text`, `approved?`, `ts?` | `ScoreOut`: `claim_id`, `score`, `action`, `degraded`, `thresholds`, `features`, `evidence`, `model_loaded`, `model_sha256`, `request_id` |
| **Inference** | `POST /v1/score/shadow` | **Yes** | `ClaimIn` | Same as `ScoreOut`, but does **not** update cluster state or write to audit log |
| **Console Data** | `GET /v1/claims` | **Yes** | Query: `risk`, `merchant`, `q`, `min_amount`, `max_amount`, `sort`, `order`, `page`, `page_size` | Paginated JSON list of claims read from `data/test_claims_scored.parquet` enriched with case status |
| **Console Data** | `GET /v1/claims/{claim_id}` | **Yes** | Path: `claim_id` | Complete claim dossier: features, cluster evidence, graph nodes/edges, timeline, identity history |
| **Console Data** | `GET /v1/claims/{claim_id}/counterfactuals`| **Yes** | Path: `claim_id` | Perturbation attribution against hardcoded benign baselines (`CF_BASELINES`), greedy step path |
| **Simulator** | `GET /v1/settlement/impact` | **Yes** | Query: `high`, `medium` | Aggregates test set into hypothetical bands: held value, delayed payout calendar, top affected merchants |
| **Merchant** | `GET /v1/merchants/{merchant_id}/risk` | **Yes** | Path: `merchant_id` | Aggregated merchant profile: risk level (`HIGH/MED/LOW`), claim rate, connected rings, held amount |
| **Audit/Ops** | `POST /v1/decisions` | **Yes** | `DecisionIn`: `claim_id`, `kind`, `prev_action`, `new_action`, `reason` | Records human review override in `analyst_actions` SQLite table |
| **Audit/Ops** | `GET /v1/decisions` | **Yes** | Query: `claim_id?`, `limit?` | Returns list of recorded analyst actions |
| **Cases** | `GET /v1/cases` | **Yes** | None | Returns all rows from SQLite `case_state` |
| **Cases** | `GET /v1/case/{claim_id}` | **Yes** | Path: `claim_id` | Returns single case state, notes list, watchlist flag |
| **Cases** | `PATCH /v1/case/{claim_id}` | **Yes** | `CasePatchIn`: `status?`, `assigned_to?`, `sla_hours?` | Updates case status, assignment, SLA deadline |
| **Notes** | `GET /v1/claims/{claim_id}/notes` | **Yes** | Path: `claim_id` | Retrieves notes from `case_notes` table |
| **Notes** | `POST /v1/claims/{claim_id}/notes` | **Yes** | `NoteIn`: `body` | Inserts note into `case_notes` and logs analyst audit event |
| **Bulk Ops** | `POST /v1/claims/bulk` | **Yes** | `BulkActionIn`: `claim_ids`, `action`, `assigned_to?`, `reason` | Executes batch status/assign update on multiple claims |
| **Watchlist** | `GET /v1/watchlist` | **Yes** | None | Lists entities on watchlist from SQLite `watchlist` table |
| **Watchlist** | `POST /v1/watchlist` | **Yes** | `WatchIn`: `entity`, `kind`, `reason` | Adds identity, infra, or ring ID to watchlist |
| **Watchlist** | `DELETE /v1/watchlist/{entity}` | **Yes** | Path: `entity` | Deletes entity from watchlist |
| **Alert Rules**| `GET /v1/alert-rules` | **Yes** | None | Lists configured alert rules from `alert_rules` table |
| **Alert Rules**| `POST /v1/alert-rules` | **Yes** | `AlertRuleIn`: `name`, `metric`, `threshold`, `webhook_url?`| Inserts new alert rule |
| **Alert Rules**| `PATCH /v1/alert-rules/{rule_id}` | **Yes** | Query: `enabled` | Enables or disables alert rule |
| **Alert Rules**| `DELETE /v1/alert-rules/{rule_id}` | **Yes** | Path: `rule_id` | Deletes alert rule |
| **Alerts** | `GET /v1/alerts` | **Yes** | Query: `limit?` | Lists recent triggered alerts from `alert_events` table |
| **Webhooks** | `POST /v1/webhooks/test` | **Yes** | `WebhookTestIn`: `url` | Fires SSRF-validated test ping signed with HMAC-SHA256 |
| **Streaming** | `GET /v1/stream` | Auth via Query or Header | Query: `api_key?` | Server-Sent Events (SSE) stream emitting `score`, `alert`, and `ring_forming` payloads |

---

### 1.3 Reality Matrix Audit: Live vs. Simulated vs. Fictional

The project's documentation (`README.md` Section 6 and `DOCKET_COMPLETE_SYSTEM_DOCUMENTATION.md` Section 2) presents an "Implementation Reality Matrix". Here is the rigorous audit of those claims against actual source code:

| Component Claimed | Status Claimed in Docs | Actual Verification in Source Code | Audit Verdict |
| :--- | :---: | :--- | :---: |
| **In-Memory Disjoint Union-Find** | LIVE (Production Code) | [`src/score_service.py:176-390`](file:///d:/Docket-Risk/src/score_service.py#L176-L390) implements genuine Union-Find with path compression and rank-balancing under RLock. | **VERIFIED (LIVE)** |
| **XGBoost Monotonic Scoring Engine** | LIVE (Production Model) | Model file exists (`models/ring_sentinel_xgb.json`). Monotone constraints are compiled into trees (`monotone_constraints` vector in [`src/train_eval.py:90-116`](file:///d:/Docket-Risk/src/train_eval.py#L90-L116)). | **VERIFIED (LIVE)** |
| **Gain-Based Feature Attribution / Shapley Vectors** | LIVE (Production Math) | Claims in docs: *"Shapley contribution vectors"*. In reality, `src/train_eval.py` exports standard XGBoost feature importances. In `src/score_service.py:1745-1801`, explainability is a **brute-force 1D perturbation** swapping features to hardcoded baselines (`CF_BASELINES`), not actual Shapley/TreeSHAP. | **PARTIALLY MISREPRESENTED** |
| **Audit Ledger & SHA-256 Digital Seals** | LIVE (Database) | SQLite database (`data/decisions.db`) runs in WAL mode with tables for decisions, analyst actions, cases, notes, and alerts. | **VERIFIED (LIVE)** |
| **DPDP Pseudonymization Layer** | LIVE (Tokenization Engine) | Function `anonymize_pii_token` exists in [`src/config.py:127-142`](file:///d:/Docket-Risk/src/config.py#L127-L142). **However, it is never called inside `src/score_service.py`!** Raw IDs pass directly into the in-memory graph without being tokenized. | **DORMANT / NOT INTEGRATED** |
| **Carrier EDI Track-and-Trace API (BlueDart/Delhivery)** | SIMULATED CONTRACT | Docs claim: *"Queries BlueDart/Delhivery EDI APIs, severs false-positive edges, drops risk from 94.2% to 3.8%"*. **In actual code, there is ZERO trace of this.** No EDI schema, no carrier client, no edge-severing API endpoint, no mock server. | **COMPLETELY FICTIONAL** |
| **Automated RTGS Unfreeze Webhook** | LIVE (Webhook Dispatcher) | Docs claim: *"Emits standard Razorpay Route JSON payloads with signature headers to settlement endpoints"*. In actual code (`src/score_service.py:721-756`), it is a generic HTTP POST for internal alert rules (`{"topic": "alert.triggered", ...}`). There is no Razorpay Route payload, no unfreeze event, and no RTGS settlement schema. | **COMPLETELY FICTIONAL** |
| **PostgreSQL 16 & Redis 7 Infrastructure** | Stack in `docker-compose.yml` | `docker-compose.yml` provisions Postgres and Redis, but `src/score_service.py` **connects to neither**. Graph state is Python memory + JSON snapshots; audit storage is local SQLite. The services in docker-compose are completely idle. | **DEAD INFRASTRUCTURE** |

---

## 2. Model & Data Approach

### 2.1 Features Used by the Model

Defined in [`src/graph_features.py:25-36`](file:///d:/Docket-Risk/src/graph_features.py#L25-L36):

```python
FEATURE_ORDER = [
    "identity_order_count_so_far",          # Unconstrained
    "identity_merchant_count_so_far",       # Unconstrained
    "identity_claim_count_so_far",          # Unconstrained
    "identity_claim_approval_ratio_so_far", # Unconstrained (default 0.62)
    "shared_infra_neighbor_count",          # Monotonic constraint (+1)
    "cluster_size",                         # Monotonic constraint (+1)
    "cluster_merchant_span",                # Monotonic constraint (+1)
    "cluster_claim_burst_7d",               # Monotonic constraint (+1)
    "reason_text_reuse_flag",               # Monotonic constraint (+1)
    "amount",                               # Unconstrained
]
```

#### Critical Finding: Extreme Feature Importance Imbalance
According to the model's serialized evaluation artifact ([`models/eval_report.json:654-695`](file:///d:/Docket-Risk/models/eval_report.json#L654-L695)):

```json
"feature_importance": [
  { "feature": "cluster_size", "importance": 0.5883 },
  { "feature": "shared_infra_neighbor_count", "importance": 0.3929 },
  { "feature": "cluster_merchant_span", "importance": 0.0076 },
  { "feature": "identity_claim_count_so_far", "importance": 0.0054 },
  { "feature": "identity_order_count_so_far", "importance": 0.0037 },
  { "feature": "amount", "importance": 0.0011 },
  { "feature": "identity_merchant_count_so_far", "importance": 0.0010 },
  { "feature": "identity_claim_approval_ratio_so_far", "importance": 0.0 },
  { "feature": "cluster_claim_burst_7d", "importance": 0.0 },
  { "feature": "reason_text_reuse_flag", "importance": 0.0 }
]
```

> **Key Vulnerability:** **98.12% of the model's predictive weight relies on just two features: `cluster_size` (58.8%) and `shared_infra_neighbor_count` (39.3%)**.  
> The remaining 8 features contribute less than 2% combined, and three features have literally **0.0000** importance. The model is essentially a two-feature decision tree masked as a 10-feature complex model.

---

### 2.2 Dataset Origin & Generation Logic

The entire dataset is **100% synthetic**. There is no real-world merchant or payment gateway data. All transactions and claims are generated by [`src/data_gen.py`](file:///d:/Docket-Risk/src/data_gen.py).

#### Actual Data Generation Rules:
1. **Identities ([`src/data_gen.py:20-25`](file:///d:/Docket-Risk/src/data_gen.py#L20-L25))**:
   - `N_LEGIT = 20,000`: Formatted as `USR_000000`. 15% designated as `is_power_shopper`.
   - `N_CAMOUFLAGE = 800`: Formatted as `CAMO_00000`. Legit shoppers injected with exactly one shared device or address taken from a random legit donor (simulating shared office/IP environments).
   - `N_RING_IDENTITIES = 591`: Formatted as `RNG{r:03d}_{m:02d}` across 70 rings (`N_RINGS = 70`).
2. **Ring Topologies ([`src/data_gen.py:38-47`](file:///d:/Docket-Risk/src/data_gen.py#L38-L47))**:
   - Four discrete ring types cycled deterministically: `DEVICE_ONLY`, `VPA_ONLY`, `ADDRESS_ONLY`, and `MIXED`.
3. **Hardcoded Reason Text Strings ([`src/data_gen.py:54-85`](file:///d:/Docket-Risk/src/data_gen.py#L54-L85))**:
   - Legitimate claims randomly pick from 24 plausible phrases (e.g., *"Item did not fit"*, *"Wrong color received"*).
   - Ring claims randomly pick from only **3 specific phrases**:
     - `"Item never arrived at my address"`
     - `"Package showed delivered but is missing"`
     - `"Delivery confirmed but nothing received"`
4. **Hardcoded Probabilities**:
   - `RING_CLAIM_PROB = 0.22` (vs. legit category rates 0.07–0.22).
   - `RING_APPROVAL_PROB = 0.85` (vs. `LEGIT_APPROVAL_PROB = 0.62`).
   - Claim window: claims follow orders by `(1, 10)` days.

---

### 2.3 Reported Metrics & Temporal Split Verification

In [`src/train_eval.py:43-44, 83-87`](file:///d:/Docket-Risk/src/train_eval.py#L43-L44):

```python
VAL_START = pd.Timestamp("2026-05-01")
TEST_START = pd.Timestamp("2026-06-01")

def temporal_split(df: pd.DataFrame):
    train = df[df["claim_ts"] < VAL_START]
    val = df[(df["claim_ts"] >= VAL_START) & (df["claim_ts"] < TEST_START)]
    test = df[df["claim_ts"] >= TEST_START]
    return train, val, test
```

#### Split Verification:
- **Training Set (Jan 1 – Apr 30, 2026):** $N = 12,644$ claims, 215 ring positives (**1.70% base rate**).
- **Validation Set (May 1 – May 31, 2026):** $N = 3,368$ claims, 57 ring positives (**1.69% base rate**).
- **Test Set (Jun 1 – Jun 30, 2026):** $N = 3,877$ claims, 66 ring positives (**1.70% base rate**).

#### Reported Metrics on Test Set ($N = 3,877$):
- **PR-AUC:** `0.9142` (against base rate `0.0170`).
- **ROC-AUC:** `0.9421`.
- **Calibration:** Brier score = `0.0248`, ECE (10-bin) = `0.0295`.
- **Threshold Cutoffs**:
  - `HIGH` ($\ge 0.85$): Precision = `0.9077`, Recall = `0.8939`, F1 = `0.9008`. Flagged = 65 claims (59 TP, 6 FP).
  - `MEDIUM` ($\ge 0.50$): Precision = `0.7848`, Recall = `0.9394`, F1 = `0.8552`. Flagged = 79 claims (62 TP, 17 FP).
- **Camouflage Cohort False-Flag Rate:** 5.62% at `HIGH`, 15.73% at `MEDIUM`.

#### Integrity Assessment of Metrics:
The temporal split implementation is **methodologically clean** (zero lookahead leakage in [`src/graph_features.py:189-202`](file:///d:/Docket-Risk/src/graph_features.py#L189-L202) because order replay strictly precedes claim evaluation).  
**However**, the exceptional PR-AUC (`0.9142` with a 1.7% base rate) is an artifact of the synthetic data generation: ring members are explicitly generated sharing identical device/VPA tokens in tight clusters of sizes 3–14, making graph connectivity an almost perfect deterministic predictor.

---

## 3. Genuine Strengths (Objective Engineering Review)

Do not underestimate this competitor—it has several genuinely impressive software engineering strengths that set a high bar:

1. **Dual Fail-Open Protection ([`src/score_service.py:415-460`](file:///d:/Docket-Risk/src/score_service.py#L415-L460))**:
   - At startup: if the model JSON is missing or its SHA-256 signature fails verification, the service logs a critical warning and starts anyway, defaulting to `AUTO_APPROVE` with `degraded=True`.
   - At runtime: any unexpected exception during inference is caught, metric `ring_sentinel_failopen_runtime_total` is incremented, and an `AUTO_APPROVE` decision is returned. The `/v1/score` endpoint **never returns an HTTP 500 status**. In payment gateways, checkout conversion is sacred, and this is standard tier-1 production practice.
2. **Sub-15ms Latency via In-Memory Union-Find**:
   - Rather than executing slow multi-hop relational SQL queries or graph database roundtrips (e.g. Neo4j Cypher queries), `GraphState` operates in local Python memory using disjoint sets with near $O(1)$ inverse Ackermann complexity ($O(\alpha(N))$). In benchmark runs, p99 decision latency is under 15ms.
3. **Atomic TOCTOU Prevention ([`src/score_service.py:253-328`](file:///d:/Docket-Risk/src/score_service.py#L253-L328))**:
   - In distributed refund attacks, multiple members of a syndicate fire claims simultaneously. Docket-Risk runs feature extraction, XGBoost inference, and graph claim mutation inside a single re-entrant lock acquisition (`with self._lock:`). Simultaneous claims can never observe half-updated cluster state.
4. **Strict Monotonic Tree Constraints ([`src/train_eval.py:90-116`](file:///d:/Docket-Risk/src/train_eval.py#L90-L116))**:
   - By enforcing XGBoost's `monotone_constraints` on all 5 graph density features, the system mathematically guarantees that adding more shared infrastructure or enlarging a syndicate cluster will **never** decrease a risk score.
5. **Cryptographic Model Verification**:
   - Model files are checksummed on load via SHA-256 and compared using constant-time `hmac.compare_digest` against a companion `.sha256` file to prevent tampering.
6. **Robust Operational Hardening**:
   - Idempotency caching for up to 50,000 orders/claims using LRU `OrderedDict`.
   - Sliding-window in-memory rate limiting (600 requests/min).
   - Disk snapshotting: writes memory state to disk on shutdown and every 60 seconds atomically via temporary file replacement (`os.replace`).
   - Standardized Prometheus metrics (`/metrics`) and structured JSON logs with correlation IDs.

---

## 4. Gaps & Blind Spots (Strategic Differentiation Opportunities)

These are the clear, defensible gaps where an independent competing project can decisively win:

### 4.1 Zero LLM / Generative AI / Plain-English Explainability Layer
- **Current State:** Docket-Risk has **no LLM integration**. Its explainability consists solely of raw numerical feature tables, feature importance bars, and a simple 1D brute-force perturbation step (`CLAIMS_STORE.counterfactuals`).
- **The Gap:** Payment gateway risk operations require human-readable narrative explanations. An analyst or merchant receiving a payout hold cannot interpret *"cluster_size: 6, shared_infra_neighbor_count: 5"*. There is no automated synthesis explaining *why* the merchant is flagged, *which* entities are malicious, or *what* specific action resolves it.
- **Your Differentiation:** Build a generative AI Narrative Explainability Engine (e.g. Gemini 1.5 Flash / Claude 3.5 Sonnet) that consumes the telemetry, graph evidence, and transaction metadata to produce natural language risk dossiers and merchant-facing notices.

### 4.2 Complete Absence of Chargeback Evidence Document Generation
- **Current State:** Zero capability. There is no PDF generation, no document builder, no receipt compiler, and no card-network dispute representment packager.
- **The Gap:** Track 02 explicitly highlights *"Chargeback evidence responder: compile order details, delivery proof, customer communications into a dispute response packet"*. Docket-Risk completely ignores this entire half of the track.
- **Your Differentiation:** Implement an end-to-end Dispute Evidence Compiler that parses courier proofs, order logs, terms of service agreements, and customer chat history into compliant PDF dispute defense packets formatted according to Visa/Mastercard/NPCI guidelines.

### 4.3 No Conversational / Natural Language Risk Copilot
- **Current State:** The ops UI is purely static forms and pre-defined dashboard filters.
- **The Gap:** Risk investigators cannot query their dataset with natural questions (e.g., *"Show me all high-value electronics merchants whose refund velocity spiked by >50% following the Diwali weekend"* or *"Draft an email to Merchant MRC_0012 explaining why a 15% reserve was applied"*).
- **Your Differentiation:** Integrate a conversational AI Risk Analyst Copilot allowing risk teams to investigate clusters, query risk state, and trigger policy adjustments via natural language.

### 4.4 Fictional Carrier EDI & Auto-Unfreeze Claims
- **Current State:** The documentation claims to support *"BlueDart and Delhivery mTLS tracking APIs"* and *"Carrier-Verified Auto-Unfreeze Sandbox"*. As proven in Section 1.3, **there is literally zero code for this in the entire repository**.
- **The Gap:** It is an unfulfilled marketing promise.
- **Your Differentiation:** Actually build a working Carrier Verification & Dispute Auto-Responder module—either against real sandbox carrier tracking APIs or a verified mock carrier server that inspects tracking numbers, validates delivery timestamps against claim timestamps, and programmatically resolves dispute holds.

### 4.5 Extreme Model Fragility (Over-Reliance on Hardcoded Identifiers)
- **Current State:** 98.1% of the model's feature importance is based on `cluster_size` and `shared_infra_neighbor_count`.
- **The Gap:** Real-world syndicates use anti-detect browsers, multi-accounting tools (Dolphin Anty, Multilogin), burner SIM cards, and rotated VPAs. When device and VPA tokens are rotated, Docket-Risk's graph completely breaks because it has no behavioral velocity modeling, no SKU-level risk signals, and no NLP analysis of dispute text.
- **Your Differentiation:** Build a multi-modal risk architecture that combines graph signals with transactional anomaly detection, behavioral velocity, and semantic dispute text analysis.

---

## 5. Track Brief Alignment Matrix

The Razorpay Hackathon Track 02 brief outlines 4 core sample directions:

```
Track 02: AI Risk Manager
"Stop the merchant losing money to fraud, returns and chargebacks"
│
├── 1. Abuse-Ring Sentinel
├── 2. Return-Risk Scorer
├── 3. Fraud-Spike Detector
└── 4. Chargeback Evidence Responder
```

Here is the exact coverage breakdown of Docket-Risk against these 4 directions:

| Track Direction | Competitor Coverage | Evidence in Docket-Risk Codebase | Assessment |
| :--- | :---: | :--- | :--- |
| **1. Abuse-Ring Sentinel** | **90%** | [`src/graph_features.py`](file:///d:/Docket-Risk/src/graph_features.py), [`src/score_service.py`](file:///d:/Docket-Risk/src/score_service.py). Core implementation of bipartite graph union-find, identity projection, and monotonic cluster scoring. | **HEAVILY COVERED** — This is what the project actually is. Attempting to compete solely on union-find ring detection is redundant. |
| **2. Return-Risk Scorer** | **30%** | Uses claim amount, return category, and prior approval ratio. However, it does **not** evaluate customer return frequency, product SKU return propensity, delivery verification, or return policy abuse. | **SUPERFICIALLY COVERED** — Scored only as a binary claim label, not as a nuanced return-risk policy optimizer. |
| **3. Fraud-Spike Detector** | **15%** | Minimal implementation: [`_check_ring_forming`](file:///d:/Docket-Risk/src/score_service.py#L815-L855) checks if a cluster grew by $\ge 3$ members in 30 minutes, and basic alert rules. No statistical anomaly detection (e.g. z-score, Isolation Forest), no merchant volume spike tracking, no gateway-wide burst mitigation. | **MOSTLY UNTOUCHED** — Very weak heuristic implementation. |
| **4. Chargeback Evidence Responder** | **0%** | **Completely absent.** No evidence compilation, no document generator, no carrier proof reconciliation, no automated dispute response generator. | **COMPLETELY UNTOUCHED** — Massive open opportunity. |

---

## 6. Strategic Takeaways for Your New Project

To clearly beat and differentiate from this competitor, your project should execute the following strategy:

1. **Competitor Analysis & Differentiators: Kavach vs Docket-Risk**
   - Docket-Risk is 90% focused on Direction 1 (Abuse-Ring Sentinel) and has 0% in Direction 4.
   - Building a **GenAI Chargeback Defense & Evidence Auto-Responder** directly targets the largest unaddressed mandate in the track brief.
2. **Implement Real Generative AI:**
   - Docket-Risk uses traditional ML (XGBoost) and classical algorithms (Union-Find). It uses **zero LLM technology**.
   - Your project should position itself as the *intelligent generative layer*—drafting evidence packs, summarizing dispute timelines, and communicating with merchants.
   - Actually implement courier proof verification (e.g. parsing shipping labels, AWB status validation) and an actual Dispute Resolution & Capital Unfreeze workflow.
4. **Complement, Don't Clone:**
   - Avoid spending time reimplementing an in-memory disjoint-set graph from scratch. If you touch graph analysis, keep it simple or leverage an existing library, and focus your engineering hours on **actionable dispute defense, LLM explainability, and merchant capital preservation**.

---
*Report generated strictly from inspection of `src/score_service.py`, `src/graph_features.py`, `src/train_eval.py`, `src/data_gen.py`, `src/config.py`, `models/eval_report.json`, and project documentation.*
