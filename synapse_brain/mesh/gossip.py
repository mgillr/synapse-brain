"""Gossip protocol for state propagation between spores.

Uses a pull-push gossip model:
1. Each cycle, a spore picks random peers and sends its Merkle root
2. If roots differ, spores exchange missing deltas
3. CRDT merge guarantees convergence regardless of message ordering

Transport: HTTP/JSON over httpx. Lightweight enough for free-tier services.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """Track information about a known peer."""

    peer_id: str
    url: str
    last_seen: float = 0.0
    merkle_root: str = ""
    clock: int = 0
    failures: int = 0
    latency_ms: float = 0.0

    @property
    def is_alive(self) -> bool:
        """Consider a peer dead if not seen in 5 minutes."""
        return (time.time() - self.last_seen) < 300

    @property
    def is_healthy(self) -> bool:
        """Healthy if alive and fewer than 3 consecutive failures."""
        return self.is_alive and self.failures < 3


class GossipProtocol:
    """Pull-push gossip with Merkle-based anti-entropy.

    Each spore runs one GossipProtocol instance. It:
    - Listens for incoming sync requests on its HTTP port
    - Periodically pushes state digests to random peers
    - Pulls missing deltas when Merkle roots diverge
    """

    def __init__(
        self,
        spore_id: str,
        port: int = 8470,
        seeds: list[str] | None = None,
        on_delta: Callable | None = None,
        fanout: int = 3,
        gossip_interval: float = 10.0,
    ):
        self.spore_id = spore_id
        self.port = port
        self.seeds = seeds or []
        self.on_delta = on_delta  # callback: (delta, source_spore) -> bool
        self.fanout = fanout
        self.gossip_interval = gossip_interval

        self._peers: dict[str, PeerInfo] = {}
        self._client = httpx.AsyncClient(timeout=10.0)

        # Register seed peers
        for seed_url in self.seeds:
            peer_id = f"seed-{seed_url}"
            self._peers[peer_id] = PeerInfo(peer_id=peer_id, url=seed_url)

        logger.info(
            "Gossip protocol initialized for %s with %d seeds",
            spore_id, len(self.seeds),
        )

    @property
    def peer_count(self) -> int:
        return len(self._peers)

    @property
    def healthy_peers(self) -> list[PeerInfo]:
        return [p for p in self._peers.values() if p.is_healthy]

    def add_peer(self, peer_id: str, url: str) -> None:
        """Register a new peer."""
        if peer_id not in self._peers and peer_id != self.spore_id:
            self._peers[peer_id] = PeerInfo(peer_id=peer_id, url=url)
            logger.info("Discovered peer %s at %s", peer_id, url)

    async def broadcast(self, state_snapshot: dict[str, Any]) -> int:
        """Push state digest to random subset of peers.

        Returns number of peers successfully contacted.
        """
        targets = self.healthy_peers
        if not targets:
            targets = list(self._peers.values())

        if not targets:
            return 0

        # Select random fanout peers
        selected = random.sample(targets, min(self.fanout, len(targets)))
        success = 0

        digest = {
            "type": "digest",
            "spore_id": self.spore_id,
            "clock": state_snapshot.get("clock", 0),
            "merkle_root": state_snapshot.get("merkle_root", ""),
            "task_summaries": {
                tid: info["delta_count"]
                for tid, info in state_snapshot.get("tasks", {}).items()
            },
        }

        for peer in selected:
            try:
                start = time.time()
                resp = await self._client.post(
                    f"{peer.url}/gossip/digest",
                    json=digest,
                )
                peer.latency_ms = (time.time() - start) * 1000
                peer.last_seen = time.time()
                peer.failures = 0

                if resp.status_code == 200:
                    success += 1
                    reply = resp.json()

                    # Process any deltas the peer sent back
                    for delta_data in reply.get("missing_deltas", []):
                        from synapse_brain.spore.runtime import ReasoningDelta
                        delta = ReasoningDelta.from_dict(delta_data)
                        if self.on_delta:
                            self.on_delta(delta, peer.peer_id)

                    # Learn about new peers
                    for p in reply.get("peers", []):
                        self.add_peer(p["peer_id"], p["url"])

            except Exception as e:
                peer.failures += 1
                logger.debug("Failed to gossip with %s: %s", peer.peer_id, e)

        return success

    async def handle_digest(self, digest: dict[str, Any], local_snapshot: dict[str, Any]) -> dict[str, Any]:
        """Handle an incoming digest from a peer.

        Compare Merkle roots. If they differ, identify missing deltas
        and return them.
        """
        peer_id = digest["spore_id"]
        peer_root = digest.get("merkle_root", "")
        local_root = local_snapshot.get("merkle_root", "")

        response: dict[str, Any] = {
            "spore_id": self.spore_id,
            "missing_deltas": [],
            "peers": [
                {"peer_id": p.peer_id, "url": p.url}
                for p in self.healthy_peers[:10]
            ],
        }

        if peer_root == local_root:
            return response

        # Roots differ: find deltas the peer is missing
        peer_tasks = digest.get("task_summaries", {})
        local_tasks = local_snapshot.get("tasks", {})
        local_store = local_snapshot.get("delta_store", {})

        for task_id, task_info in local_tasks.items():
            local_count = task_info["delta_count"]
            peer_count = peer_tasks.get(task_id, 0)

            if local_count > peer_count:
                # Send deltas the peer does not have
                for fp in task_info["fingerprints"]:
                    if fp in local_store:
                        response["missing_deltas"].append(local_store[fp])

        return response

    async def pull_from_peer(self, peer: PeerInfo, local_snapshot: dict[str, Any]) -> int:
        """Actively pull missing state from a specific peer.

        Returns number of new deltas received.
        """
        try:
            resp = await self._client.post(
                f"{peer.url}/gossip/pull",
                json={
                    "spore_id": self.spore_id,
                    "merkle_root": local_snapshot.get("merkle_root", ""),
                    "known_fingerprints": list(local_snapshot.get("delta_store", {}).keys()),
                },
            )

            if resp.status_code == 200:
                data = resp.json()
                count = 0
                for delta_data in data.get("deltas", []):
                    from synapse_brain.spore.runtime import ReasoningDelta
                    delta = ReasoningDelta.from_dict(delta_data)
                    if self.on_delta:
                        if self.on_delta(delta, peer.peer_id):
                            count += 1
                return count

        except Exception as e:
            peer.failures += 1
            logger.debug("Pull from %s failed: %s", peer.peer_id, e)

        return 0

    async def listen(self):
        """Start HTTP listener for incoming gossip.

        Uses a lightweight ASGI server (or falls back to raw asyncio).
        """
        from synapse_brain.mesh.server import create_gossip_server

        server = create_gossip_server(self)
        await server.serve(port=self.port)

    def status(self) -> dict[str, Any]:
        """Return gossip protocol status."""
        return {
            "spore_id": self.spore_id,
            "total_peers": len(self._peers),
            "healthy_peers": len(self.healthy_peers),
            "peers": {
                pid: {
                    "alive": p.is_alive,
                    "healthy": p.is_healthy,
                    "latency_ms": round(p.latency_ms, 1),
                    "failures": p.failures,
                    "last_seen": p.last_seen,
                }
                for pid, p in self._peers.items()
            },
        }
