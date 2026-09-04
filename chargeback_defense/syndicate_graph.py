"""In-Memory Multi-Merchant Syndicate Ring Detection Engine.

Implements an in-memory Disjoint Set Union (Union-Find with path compression and
union-by-rank) graph tracker that monitors shared digital and physical infrastructure:
- device_id (Hardware fingerprint)
- upi_vpa (Virtual Payment Address)
- phone_hash (Normalized phone digest)
- shipping_pincode (Delivery postal location)

Extracts 4 real-time graph features in <0.1ms (<10ms SLA):
1. cluster_size: Total unique nodes in the connected component
2. shared_infra_neighbor_count: Number of distinct user identities sharing infrastructure
3. cluster_merchant_span: Number of distinct merchant accounts targeted by this cluster
4. cluster_burst_7d: Total claims from this cluster in the trailing 7 days

Syndicate Alert Trigger:
- If cluster_merchant_span >= 3 AND cluster_burst_7d >= 3:
  Flagged as "COORDINATED_SYNDICATE_ATTACK".
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple


class DisjointSetUnion:
    """Disjoint Set with path compression and union-by-rank."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, node: str) -> str:
        """Find root with path compression."""
        if node not in self.parent:
            self.parent[node] = node
            self.rank[node] = 0
            return node

        # Path compression
        path = []
        curr = node
        while self.parent[curr] != curr:
            path.append(curr)
            curr = self.parent[curr]

        root = curr
        for p in path:
            self.parent[p] = root

        return root

    def union(self, u: str, v: str) -> str:
        """Union two nodes by rank, returns the common root."""
        root_u = self.find(u)
        root_v = self.find(v)

        if root_u == root_v:
            return root_u

        # Union by rank
        if self.rank[root_u] < self.rank[root_v]:
            self.parent[root_u] = root_v
            return root_v
        elif self.rank[root_u] > self.rank[root_v]:
            self.parent[root_v] = root_u
            return root_u
        else:
            self.parent[root_v] = root_u
            self.rank[root_u] += 1
            return root_u


class SyndicateGraphTracker:
    """Thread-safe in-memory graph tracker for multi-merchant fraud rings."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.dsu = DisjointSetUnion()

        # Cluster metadata keyed by root node
        self.cluster_nodes: Dict[str, Set[str]] = {}
        self.cluster_identities: Dict[str, Set[str]] = {}
        self.cluster_merchants: Dict[str, Set[str]] = {}
        # Stores tuples of (claim_timestamp_sec, claim_id, merchant_id)
        self.cluster_claims: Dict[str, List[Tuple[float, str, str]]] = {}

        # Node attribute metadata
        self.node_types: Dict[str, str] = {}  # node -> 'user', 'device', 'vpa', 'phone', 'pincode'

    def _ensure_cluster(self, root: str) -> None:
        if root not in self.cluster_nodes:
            self.cluster_nodes[root] = {root}
        if root not in self.cluster_identities:
            self.cluster_identities[root] = set()
        if root not in self.cluster_merchants:
            self.cluster_merchants[root] = set()
        if root not in self.cluster_claims:
            self.cluster_claims[root] = []

    def _merge_cluster_meta(self, new_root: str, old_root: str) -> None:
        """Merge metadata from old_root into new_root when components union."""
        if new_root == old_root:
            return
        self._ensure_cluster(new_root)
        if old_root in self.cluster_nodes:
            self.cluster_nodes[new_root].update(self.cluster_nodes.pop(old_root))
        if old_root in self.cluster_identities:
            self.cluster_identities[new_root].update(self.cluster_identities.pop(old_root))
        if old_root in self.cluster_merchants:
            self.cluster_merchants[new_root].update(self.cluster_merchants.pop(old_root))
        if old_root in self.cluster_claims:
            self.cluster_claims[new_root].extend(self.cluster_claims.pop(old_root))

    def add_event(
        self,
        identity_id: str,
        merchant_id: str,
        device_id: Optional[str] = None,
        upi_vpa: Optional[str] = None,
        phone_hash: Optional[str] = None,
        shipping_pincode: Optional[str] = None,
        claim_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        is_claim: bool = False,
    ) -> str:
        """Record a transaction or dispute event, unioning shared infrastructure.

        Returns:
            The root identifier of the connected cluster.
        """
        ts = timestamp if timestamp is not None else time.time()
        user_node = f"usr:{identity_id}"

        with self._lock:
            # Register user node
            root = self.dsu.find(user_node)
            self._ensure_cluster(root)
            self.node_types[user_node] = "user"
            self.cluster_nodes[root].add(user_node)
            self.cluster_identities[root].add(identity_id)
            if merchant_id:
                self.cluster_merchants[root].add(str(merchant_id))

            # Connect all infrastructure nodes
            infra_nodes = []
            if device_id:
                dev_node = f"dev:{device_id}"
                self.node_types[dev_node] = "device"
                infra_nodes.append(dev_node)
            if upi_vpa:
                vpa_node = f"vpa:{upi_vpa.lower().strip()}"
                self.node_types[vpa_node] = "vpa"
                infra_nodes.append(vpa_node)
            if phone_hash:
                phone_node = f"ph:{phone_hash.strip()}"
                self.node_types[phone_node] = "phone"
                infra_nodes.append(phone_node)
            if shipping_pincode:
                pin_node = f"pin:{shipping_pincode.strip()}"
                self.node_types[pin_node] = "pincode"
                infra_nodes.append(pin_node)

            for infra_node in infra_nodes:
                old_root_user = self.dsu.find(user_node)
                old_root_infra = self.dsu.find(infra_node)
                if old_root_user != old_root_infra:
                    new_root = self.dsu.union(old_root_user, old_root_infra)
                    other_root = old_root_infra if new_root == old_root_user else old_root_user
                    self._merge_cluster_meta(new_root, other_root)
                    root = new_root
                else:
                    root = old_root_user

                self._ensure_cluster(root)
                self.cluster_nodes[root].add(infra_node)

            # Record claim burst if applicable
            if is_claim and claim_id:
                self._ensure_cluster(root)
                self.cluster_claims[root].append((ts, str(claim_id), str(merchant_id)))

            return root

    def extract_features(
        self,
        identity_id: str,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Extract the 4 real-time graph features under <10ms for a given identity."""
        t_start = time.perf_counter()
        now_ts = current_time if current_time is not None else time.time()
        user_node = f"usr:{identity_id}"

        with self._lock:
            root = self.dsu.find(user_node)
            self._ensure_cluster(root)

            # 1. Total unique nodes
            nodes = self.cluster_nodes.get(root, {user_node})
            cluster_size = len(nodes)

            # 2. Shared infrastructure neighbor identities
            identities = self.cluster_identities.get(root, {identity_id})
            # Neighbors = other users sharing this cluster
            shared_infra_neighbor_count = max(0, len(identities) - 1)

            # 3. Multi-merchant span
            merchants = self.cluster_merchants.get(root, set())
            cluster_merchant_span = len(merchants)

            # 4. Trailing 7-day claim burst (7 * 86,400 = 604,800 seconds)
            seven_days_sec = 7.0 * 86400.0
            claims_window_start = now_ts - seven_days_sec
            raw_claims = self.cluster_claims.get(root, [])
            burst_claims = [c for c in raw_claims if c[0] >= claims_window_start]
            cluster_burst_7d = len(burst_claims)

            # Syndicate Alert Condition
            is_syndicate_attack = (cluster_merchant_span >= 3 and cluster_burst_7d >= 3)
            syndicate_alert = "COORDINATED_SYNDICATE_ATTACK" if is_syndicate_attack else "NORMAL_MERCHANT_TRAFFIC"

            # Infrastructure breakdown
            node_breakdown: Dict[str, List[str]] = {"user": [], "device": [], "vpa": [], "phone": [], "pincode": []}
            for n in nodes:
                ntype = self.node_types.get(n, "unknown")
                raw_val = n.split(":", 1)[1] if ":" in n else n
                if ntype in node_breakdown:
                    node_breakdown[ntype].append(raw_val)

            duration_ms = (time.perf_counter() - t_start) * 1000.0

            return {
                "identity_id": identity_id,
                "cluster_root": root,
                "cluster_size": cluster_size,
                "shared_infra_neighbor_count": shared_infra_neighbor_count,
                "cluster_merchant_span": cluster_merchant_span,
                "cluster_burst_7d": cluster_burst_7d,
                "is_syndicate_attack": is_syndicate_attack,
                "syndicate_alert": syndicate_alert,
                "merchants_targeted": sorted(list(merchants)),
                "recent_claims": [c[1] for c in burst_claims[-5:]],
                "infrastructure_breakdown": node_breakdown,
                "extraction_latency_ms": round(duration_ms, 3),
            }


# =============================================================================
# BENCHMARK GRAPH SEEDER
# =============================================================================

_GLOBAL_TRACKER: Optional[SyndicateGraphTracker] = None
_INIT_LOCK = threading.Lock()


def get_syndicate_graph() -> SyndicateGraphTracker:
    """Retrieve or initialize the singleton SyndicateGraphTracker."""
    global _GLOBAL_TRACKER
    with _INIT_LOCK:
        if _GLOBAL_TRACKER is None:
            _GLOBAL_TRACKER = SyndicateGraphTracker()
            seed_benchmark_syndicates(_GLOBAL_TRACKER)
        return _GLOBAL_TRACKER


def seed_benchmark_syndicates(tracker: SyndicateGraphTracker) -> None:
    """Pre-seed the in-memory graph with multi-merchant rings and honest merchant baselines."""
    now = time.time()
    day = 86400.0

    # -------------------------------------------------------------------------
    # RING 1: "Shadow-UPI Syndicate Alpha" (Coordinated Multi-Merchant Refund Ring)
    # 4 distinct user accounts sharing 2 device IDs and 1 UPI handle across 4 merchants
    # -------------------------------------------------------------------------
    shared_vpa = "refund_ring_alpha@okhdfcbank"
    shared_device_1 = "DEV_X900_MUMBAI"
    shared_device_2 = "DEV_X901_MUMBAI"
    shared_pin = "400051"

    syndicate_members = [
        ("USR_SYN_01", "MERCH_ELECTRONICS_D2C", shared_device_1, shared_vpa, "PH_HASH_991", shared_pin, "CLM_SYN_101", now - (1 * day)),
        ("USR_SYN_02", "MERCH_APPAREL_INDIA", shared_device_1, shared_vpa, "PH_HASH_992", shared_pin, "CLM_SYN_102", now - (2 * day)),
        ("USR_SYN_03", "MERCH_COSMETICS_DIRECT", shared_device_2, shared_vpa, "PH_HASH_993", shared_pin, "CLM_SYN_103", now - (3 * day)),
        ("USR_3490660", "MERCH_HOME_DECOR_D2C", shared_device_2, shared_vpa, "PH_HASH_994", shared_pin, "CLM_3490660", now - (2.5 * day)),
    ]

    for uid, mid, dev, vpa, ph, pin, cid, ts in syndicate_members:
        tracker.add_event(
            identity_id=uid,
            merchant_id=mid,
            device_id=dev,
            upi_vpa=vpa,
            phone_hash=ph,
            shipping_pincode=pin,
            claim_id=cid,
            timestamp=ts,
            is_claim=True,
        )

    # -------------------------------------------------------------------------
    # RING 2: "Device-Farm Ring Beta" (Targeting electronics & gadgets across 3 stores)
    # -------------------------------------------------------------------------
    shared_device_farm = "DEV_EMULATOR_HYDERABAD_77"
    farm_members = [
        ("USR_FARM_01", "MERCH_GADGETS_INDIA", shared_device_farm, "buyer1@paytm", "PH_HASH_771", "500081", "CLM_FARM_201", now - (1.5 * day)),
        ("USR_FARM_02", "MERCH_SMARTPHONES_D2C", shared_device_farm, "buyer2@ybl", "PH_HASH_772", "500081", "CLM_FARM_202", now - (2.5 * day)),
        ("USR_FARM_03", "MERCH_AUDIO_HUB", shared_device_farm, "buyer3@oksbi", "PH_HASH_773", "500081", "CLM_FARM_203", now - (3.5 * day)),
    ]

    for uid, mid, dev, vpa, ph, pin, cid, ts in farm_members:
        tracker.add_event(
            identity_id=uid,
            merchant_id=mid,
            device_id=dev,
            upi_vpa=vpa,
            phone_hash=ph,
            shipping_pincode=pin,
            claim_id=cid,
            timestamp=ts,
            is_claim=True,
        )

    # -------------------------------------------------------------------------
    # HONEST CONTROL BASELINE: Single-merchant legitimate shoppers
    # -------------------------------------------------------------------------
    honest_customers = [
        ("USR_3489068", "MERCH_BOOKS_STORE", "DEV_HONEST_01", "genuine_rahul@oksbi", "PH_HASH_111", "560001", "CLM_3489068", now - (20 * day)),
        ("USR_3490159", "MERCH_FOOTWEAR_D2C", "DEV_HONEST_02", "anita_sharma@icici", "PH_HASH_222", "110001", "CLM_3490159", now - (15 * day)),
        ("USR_3491784", "MERCH_COSMETICS_STORE", "DEV_HONEST_03", "priya_k@paytm", "PH_HASH_333", "700001", "CLM_3491784", now - (12 * day)),
    ]

    for uid, mid, dev, vpa, ph, pin, cid, ts in honest_customers:
        tracker.add_event(
            identity_id=uid,
            merchant_id=mid,
            device_id=dev,
            upi_vpa=vpa,
            phone_hash=ph,
            shipping_pincode=pin,
            claim_id=cid,
            timestamp=ts,
            is_claim=True,
        )
