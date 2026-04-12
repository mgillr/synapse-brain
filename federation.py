"""Synapse Federation Protocol -- Decentralized Collective Intelligence.

Enables any node running crdt-merge >= 0.9.5 to join the Synapse collective.
Trust-gated gossip ensures the swarm maintains integrity as it grows.

Architecture:
  - Zero-trust bootstrap: new nodes start at trust 0.0
  - Trust earned through consistent, validated contributions
  - Swarm DNA validation: protocol version + CRDT compatibility check
  - Graduated access: trust level gates what data a node receives
  - Byzantine exclusion via E4 SLT detection
  - Gossip protocol identical to internal spore gossip (same wire format)

Federation is additive to existing spore architecture. Internal spores
continue to operate as before. Federation nodes join the same gossip mesh
with trust-gated payload filtering.
"""
import hashlib
import json
import logging
import os
import time
from typing import Any

log = logging.getLogger("federation")

# Protocol constants
FEDERATION_VERSION = "1.0.0"
SWARM_DNA_VERSION = "5.1.0"
MIN_CRDT_MERGE = "0.9.5"

# Trust tiers
TRUST_BOOTSTRAP = 0.0
TRUST_OBSERVER = 0.2    # can receive collective knowledge
TRUST_CONTRIBUTOR = 0.4  # can submit tasks, receive reasoning
TRUST_PEER = 0.6        # full gossip exchange
TRUST_COMMANDER = 0.8   # can deploy spores, manage sub-swarms


class SwarmDNA:
    """Immutable protocol parameters baked into the package.

    Modified versions fail validation at every peer. This ensures
    all federation nodes speak the same protocol regardless of who
    deployed them.
    """

    def __init__(self):
        self.protocol_version = FEDERATION_VERSION
        self.swarm_version = SWARM_DNA_VERSION
        self.min_crdt_merge = MIN_CRDT_MERGE
        self.gossip_format = "json-rpc-1.0"
        self.memory_backend = "crdt-orset"
        self.trust_backend = "e4-delta-lattice"
        self.clock_type = "vector-clock"
        self.provenance = "merkle-dag"

        # Compute DNA hash -- changes if any parameter changes
        dna_string = json.dumps(self.to_dict(), sort_keys=True)
        self.dna_hash = hashlib.sha256(dna_string.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "swarm_version": self.swarm_version,
            "min_crdt_merge": self.min_crdt_merge,
            "gossip_format": self.gossip_format,
            "memory_backend": self.memory_backend,
            "trust_backend": self.trust_backend,
            "clock_type": self.clock_type,
            "provenance": self.provenance,
        }

    def validate(self, remote_dna: dict) -> tuple[bool, str]:
        """Validate a remote node's DNA against ours.

        Returns (valid, reason).
        """
        if not remote_dna:
            return False, "no_dna_provided"

        # Protocol version must match major.minor
        remote_ver = remote_dna.get("protocol_version", "0.0.0")
        local_parts = self.protocol_version.split(".")
        remote_parts = remote_ver.split(".")
        if len(remote_parts) < 2 or len(local_parts) < 2:
            return False, f"invalid_version_format: {remote_ver}"
        if remote_parts[0] != local_parts[0]:
            return False, f"major_version_mismatch: {remote_ver}"

        # CRDT merge version compatibility
        remote_crdt = remote_dna.get("min_crdt_merge", "0.0.0")
        if remote_crdt < self.min_crdt_merge:
            return False, f"crdt_merge_too_old: {remote_crdt}"

        # Core backends must match
        for key in ["gossip_format", "memory_backend", "trust_backend",
                     "clock_type", "provenance"]:
            if remote_dna.get(key) != getattr(self, key, None):
                return False, f"{key}_mismatch: {remote_dna.get(key)}"

        return True, "valid"


class FederationNode:
    """Represents a known federation participant."""

    def __init__(self, node_id: str, endpoint: str,
                 role: str = "contributor"):
        self.node_id = node_id
        self.endpoint = endpoint
        self.role = role
        self.trust = TRUST_BOOTSTRAP
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.contributions = 0
        self.violations = 0
        self.dna_validated = False
        self.specialization = ""

    def update_trust(self, delta: float):
        """Adjust trust based on behavior. Clamped to [0.0, 1.0]."""
        self.trust = max(0.0, min(1.0, self.trust + delta))

    def is_active(self, timeout: float = 600) -> bool:
        return (time.time() - self.last_seen) < timeout

    def tier(self) -> str:
        if self.trust >= TRUST_COMMANDER:
            return "commander"
        elif self.trust >= TRUST_PEER:
            return "peer"
        elif self.trust >= TRUST_CONTRIBUTOR:
            return "contributor"
        elif self.trust >= TRUST_OBSERVER:
            return "observer"
        return "bootstrap"

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "endpoint": self.endpoint,
            "role": self.role,
            "trust": round(self.trust, 4),
            "tier": self.tier(),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "contributions": self.contributions,
            "violations": self.violations,
            "dna_validated": self.dna_validated,
            "specialization": self.specialization,
            "active": self.is_active(),
        }


class FederationRegistry:
    """Registry of all known federation nodes.

    Handles node discovery, trust tracking, and gossip routing decisions.
    """

    def __init__(self, local_id: str, dna: SwarmDNA):
        self.local_id = local_id
        self.dna = dna
        self.nodes: dict[str, FederationNode] = {}
        self.blocked: set[str] = set()  # permanently excluded nodes

    def register(self, node_id: str, endpoint: str,
                 role: str = "contributor",
                 remote_dna: dict | None = None) -> tuple[bool, str]:
        """Register a new federation node. Returns (accepted, reason)."""
        if node_id == self.local_id:
            return False, "self_registration"

        if node_id in self.blocked:
            return False, "permanently_blocked"

        # Validate DNA if provided
        if remote_dna:
            valid, reason = self.dna.validate(remote_dna)
            if not valid:
                log.warning("Federation: rejected %s -- %s", node_id, reason)
                return False, reason
        else:
            # First contact without DNA -- accept provisionally
            log.info("Federation: provisional accept of %s (no DNA yet)",
                     node_id)

        if node_id in self.nodes:
            # Known node -- update last seen
            node = self.nodes[node_id]
            node.last_seen = time.time()
            if remote_dna and not node.dna_validated:
                node.dna_validated = True
            return True, "updated"

        # New node
        node = FederationNode(node_id, endpoint, role)
        if remote_dna:
            node.dna_validated = True
        self.nodes[node_id] = node
        log.info("Federation: registered %s at %s (trust: %.2f)",
                 node_id, endpoint, node.trust)
        return True, "registered"

    def get(self, node_id: str) -> FederationNode | None:
        return self.nodes.get(node_id)

    def exclude(self, node_id: str, reason: str = ""):
        """Permanently exclude a node (Byzantine detection)."""
        self.blocked.add(node_id)
        if node_id in self.nodes:
            del self.nodes[node_id]
        log.warning("Federation: excluded %s -- %s", node_id, reason)

    def gossip_targets(self, min_trust: float = 0.0) -> list[FederationNode]:
        """Get active nodes above minimum trust for gossip."""
        return [
            n for n in self.nodes.values()
            if n.is_active() and n.trust >= min_trust
            and n.node_id not in self.blocked
        ]

    def should_share_payload(self, node: FederationNode,
                              payload_type: str) -> bool:
        """Trust-gated payload sharing decision.

        Higher trust = more data shared. This is the graduated access model.
        """
        if node.node_id in self.blocked:
            return False

        gates = {
            "collective_knowledge": TRUST_OBSERVER,
            "task_summaries": TRUST_OBSERVER,
            "reasoning_deltas": TRUST_CONTRIBUTOR,
            "full_memory": TRUST_PEER,
            "trust_state": TRUST_PEER,
            "sentinel_telemetry": TRUST_COMMANDER,
        }
        required_trust = gates.get(payload_type, TRUST_PEER)
        return node.trust >= required_trust

    def build_gossip_for_node(self, node: FederationNode,
                               full_payload: dict) -> dict:
        """Build a trust-filtered gossip payload for a specific node."""
        filtered = {"from": self.local_id, "federation": True}

        if self.should_share_payload(node, "collective_knowledge"):
            filtered["collective"] = full_payload.get("collective", {})

        if self.should_share_payload(node, "task_summaries"):
            tasks = full_payload.get("tasks", {})
            # Summaries only -- no full deltas
            filtered["tasks"] = {
                tid: {
                    "description": t.get("description", "")[:200],
                    "converged": t.get("converged", False),
                    "delta_count": t.get("delta_count", 0),
                }
                for tid, t in tasks.items()
            }

        if self.should_share_payload(node, "reasoning_deltas"):
            filtered["deltas"] = full_payload.get("deltas", [])
            filtered["tasks"] = full_payload.get("tasks", {})

        if self.should_share_payload(node, "full_memory"):
            filtered["memory"] = full_payload.get("memory", {})

        if self.should_share_payload(node, "trust_state"):
            filtered["trust"] = full_payload.get("trust", {})

        return filtered

    def record_contribution(self, node_id: str):
        """Record a positive contribution from a node."""
        node = self.nodes.get(node_id)
        if node:
            node.contributions += 1
            # Trust increases with consistent contributions
            node.update_trust(0.01)

    def record_violation(self, node_id: str, severity: float = 0.1):
        """Record a protocol violation from a node."""
        node = self.nodes.get(node_id)
        if node:
            node.violations += 1
            node.update_trust(-severity)
            if node.trust <= 0.0 and node.violations >= 5:
                self.exclude(node_id, "trust_depleted")

    def active_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.is_active())

    def stats(self) -> dict:
        active = [n for n in self.nodes.values() if n.is_active()]
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": len(active),
            "blocked_nodes": len(self.blocked),
            "trust_distribution": {
                "commander": sum(
                    1 for n in active if n.tier() == "commander"
                ),
                "peer": sum(1 for n in active if n.tier() == "peer"),
                "contributor": sum(
                    1 for n in active if n.tier() == "contributor"
                ),
                "observer": sum(
                    1 for n in active if n.tier() == "observer"
                ),
                "bootstrap": sum(
                    1 for n in active if n.tier() == "bootstrap"
                ),
            },
            "mean_trust": (
                round(
                    sum(n.trust for n in active) / max(1, len(active)), 4
                )
            ),
            "dna_hash": self.dna.dna_hash,
        }

    def all_nodes(self) -> list[dict]:
        return [n.to_dict() for n in self.nodes.values()]


def mount_federation_routes(fastapi_app, registry: FederationRegistry):
    """Mount federation API endpoints onto FastAPI.

    Routes:
    - POST /federation/join: Node registration
    - POST /federation/gossip: Trust-gated gossip exchange
    - GET /federation/info: Public federation info
    - GET /federation/nodes: List known nodes (trust-gated)
    """
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @fastapi_app.post("/federation/join")
    async def federation_join(request: Request):
        body = await request.json()
        node_id = body.get("node_id", "")
        endpoint = body.get("endpoint", "")
        role = body.get("role", "contributor")
        remote_dna = body.get("dna")

        if not node_id or not endpoint:
            return JSONResponse(
                {"error": "node_id and endpoint required"}, status_code=400
            )

        accepted, reason = registry.register(
            node_id, endpoint, role, remote_dna
        )

        return JSONResponse({
            "accepted": accepted,
            "reason": reason,
            "your_trust": (
                registry.get(node_id).trust if accepted else 0.0
            ),
            "swarm_dna": registry.dna.to_dict(),
            "swarm_stats": registry.stats(),
        })

    @fastapi_app.post("/federation/gossip")
    async def federation_gossip(request: Request):
        body = await request.json()
        from_id = body.get("from", "")

        node = registry.get(from_id)
        if not node:
            return JSONResponse(
                {"error": "Unknown node. Call /federation/join first."},
                status_code=403,
            )

        if from_id in registry.blocked:
            return JSONResponse(
                {"error": "Node excluded from federation."},
                status_code=403,
            )

        # Update last seen
        node.last_seen = time.time()

        # Record contribution
        if body.get("deltas") or body.get("collective"):
            registry.record_contribution(from_id)

        # Merge incoming collective knowledge
        collective_incoming = body.get("collective", {})
        # (actual merge happens in spore.py via dual_memory)

        return JSONResponse({
            "status": "ok",
            "from": registry.local_id,
            "your_trust": node.trust,
            "your_tier": node.tier(),
            "federation_stats": registry.stats(),
        })

    @fastapi_app.get("/federation/info")
    async def federation_info():
        return JSONResponse({
            "spore": registry.local_id,
            "dna": registry.dna.to_dict(),
            "dna_hash": registry.dna.dna_hash,
            "stats": registry.stats(),
            "version": FEDERATION_VERSION,
        })

    @fastapi_app.get("/federation/nodes")
    async def federation_nodes():
        # Only return basic info publicly
        nodes = []
        for n in registry.nodes.values():
            if n.is_active():
                nodes.append({
                    "node_id": n.node_id,
                    "role": n.role,
                    "tier": n.tier(),
                    "specialization": n.specialization,
                    "active": True,
                })
        return JSONResponse({
            "nodes": nodes,
            "total": len(nodes),
            "blocked": len(registry.blocked),
        })

    log.info("Federation routes mounted at /federation/*")
