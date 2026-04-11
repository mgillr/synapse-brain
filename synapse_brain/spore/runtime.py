"""Spore runtime -- the agent execution loop.

Each spore runs this loop:
1. Pull tasks from mesh
2. Reason about task fragments
3. Produce deltas
4. Push deltas to mesh
5. Merge incoming deltas from peers
6. Sleep until next cycle
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from crdt_merge import ORSet
from crdt_merge.e4.causal_trust_clock import CausalTrustClock
from crdt_merge.e4.delta_trust_lattice import DeltaTrustLattice
from crdt_merge.e4.trust_bound_merkle import TrustBoundMerkle

from synapse_brain.cortex.protocol import (
    CognitivePhase,
    SporeRole,
    build_context_prompt,
    build_system_prompt,
    parse_reasoning_output,
    assign_roles,
)
from synapse_brain.mesh.gossip import GossipProtocol
from synapse_brain.providers.router import ProviderRouter

logger = logging.getLogger(__name__)


class SporeState(Enum):
    IDLE = "idle"
    REASONING = "reasoning"
    GOSSIPING = "gossiping"
    MERGING = "merging"
    SLEEPING = "sleeping"


@dataclass
class ReasoningDelta:
    """A single unit of reasoning produced by a spore."""

    delta_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    spore_id: str = ""
    task_id: str = ""
    content: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    parent_deltas: list[str] = field(default_factory=list)

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True).encode()

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_id": self.delta_id,
            "spore_id": self.spore_id,
            "task_id": self.task_id,
            "content": self.content,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "parent_deltas": self.parent_deltas,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReasoningDelta:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()[:16]


@dataclass
class SporeConfig:
    """Configuration for a single spore instance."""

    spore_id: str = field(default_factory=lambda: f"spore-{uuid.uuid4().hex[:8]}")
    mesh_seeds: list[str] = field(default_factory=list)
    listen_port: int = 8470
    cycle_interval: float = 30.0
    max_concurrent_tasks: int = 3
    provider_keys: dict[str, str] = field(default_factory=dict)
    role: SporeRole = SporeRole.GENERALIST

    @classmethod
    def from_env(cls) -> SporeConfig:
        """Build config from environment variables."""
        seeds = os.getenv("SYNAPSE_MESH_SEEDS", "").split(",")
        seeds = [s.strip() for s in seeds if s.strip()]

        provider_keys = {}
        for key_name in [
            "GOOGLE_AI_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
            "CEREBRAS_API_KEY", "MISTRAL_API_KEY", "HF_TOKEN",
            "CLOUDFLARE_API_TOKEN", "GITHUB_TOKEN", "TOGETHER_API_KEY",
            "NVIDIA_API_KEY", "COHERE_API_KEY", "GLM_API_KEY",
        ]:
            val = os.getenv(key_name)
            if val:
                provider_keys[key_name] = val

        return cls(
            spore_id=os.getenv("SYNAPSE_SPORE_ID", f"spore-{uuid.uuid4().hex[:8]}"),
            mesh_seeds=seeds,
            listen_port=int(os.getenv("SYNAPSE_PORT", "8470")),
            cycle_interval=float(os.getenv("SYNAPSE_CYCLE_INTERVAL", "30")),
            provider_keys=provider_keys,
        )


class Spore:
    """The atomic unit of the Synapse Brain swarm.

    Runs an async event loop that:
    - Listens for incoming gossip from peers
    - Pulls task fragments from its local CRDT state
    - Reasons about them using available LLM providers
    - Produces reasoning deltas and inserts them into CRDT state
    - Gossips updated state to peers
    """

    def __init__(self, config: SporeConfig | None = None):
        self.config = config or SporeConfig.from_env()
        self.state = SporeState.IDLE

        # Cognitive protocol -- loaded once, used on every reasoning call
        self._role = self.config.role
        self._system_prompt = build_system_prompt(self._role)
        self._cognitive_phase = CognitivePhase.EXPLORATION

        # CRDT state: OR-Set of reasoning deltas keyed by task_id
        self._deltas: dict[str, ORSet] = {}  # task_id -> ORSet of delta fingerprints
        self._delta_store: dict[str, ReasoningDelta] = {}  # fingerprint -> delta

        # Distributed Synapsis Map fragments (CRDT-merged)
        self._synapsis_nodes: dict[str, dict] = {}  # node_id -> node dict

        # E4 trust infrastructure
        self._clock = CausalTrustClock(self.config.spore_id)
        self._lattice = DeltaTrustLattice(
            self.config.spore_id,
            initial_peers={self.config.spore_id},
        )
        self._merkle = TrustBoundMerkle()

        # Mesh and reasoning
        self._gossip: GossipProtocol | None = None
        self._provider: ProviderRouter | None = None

        # Task tracking
        self._active_tasks: set[str] = set()
        self._completed_tasks: set[str] = set()

        # Attached documents for context (user-provided)
        self._documents: list[dict[str, str]] = []

        logger.info(
            "Spore %s initialized (role=%s)",
            self.config.spore_id, self._role.value,
        )

    @property
    def spore_id(self) -> str:
        return self.config.spore_id

    def get_delta_set(self, task_id: str) -> ORSet:
        """Get or create the CRDT set for a task's reasoning deltas."""
        if task_id not in self._deltas:
            self._deltas[task_id] = ORSet()
        return self._deltas[task_id]

    def insert_delta(self, delta: ReasoningDelta) -> str:
        """Insert a reasoning delta into CRDT state and Merkle tree."""
        fp = delta.fingerprint()
        delta_set = self.get_delta_set(delta.task_id)
        delta_set.add(fp)

        self._delta_store[fp] = delta
        self._merkle.insert_leaf(fp, delta.to_bytes(), self.config.spore_id)

        # Tick the causal clock
        self._clock = self._clock.increment()

        logger.debug(
            "Spore %s inserted delta %s for task %s (clock=%d)",
            self.spore_id, fp, delta.task_id, self._clock.logical_time,
        )
        return fp

    def attach_document(self, name: str, content: str) -> None:
        """Attach a document to this spore's context for reasoning."""
        self._documents.append({"name": name, "content": content})
        logger.info("Spore %s attached document: %s", self.spore_id, name)

    def clear_documents(self) -> None:
        """Clear all attached documents."""
        self._documents.clear()

    async def reason(
        self,
        task_id: str,
        prompt: str,
        subtask: str = "",
        context: list[ReasoningDelta] | None = None,
    ) -> ReasoningDelta:
        """Use an LLM provider to reason about a task fragment.

        The cognitive protocol is injected as the system prompt.
        Role-specific bias shapes HOW the spore reasons.
        Existing swarm deltas and Synapsis fragments provide context.
        Attached documents are included in the context window.

        Produces a ReasoningDelta containing structured reasoning output.
        """
        self.state = SporeState.REASONING

        if not self._provider:
            self._provider = ProviderRouter(self.config.provider_keys)

        # Gather existing deltas for context
        parent_ids = []
        existing_delta_dicts = []
        if context:
            parent_ids = [d.delta_id for d in context]
            existing_delta_dicts = [d.to_dict() for d in context]

        # Gather relevant Synapsis Map fragments
        synapsis_frags = [
            node for node in self._synapsis_nodes.values()
            if node.get("task_id") == task_id or not node.get("task_id")
        ]

        # Build the full protocol-aware prompt
        context_prompt = build_context_prompt(
            task_description=prompt,
            subtask=subtask or prompt,
            existing_deltas=existing_delta_dicts,
            synapsis_fragments=synapsis_frags,
            attached_documents=self._documents if self._documents else None,
        )

        # Route to best available provider with protocol system prompt
        response = await self._provider.reason(
            context_prompt,
            system_prompt=self._system_prompt,
        )

        # Parse the structured reasoning output
        parsed = parse_reasoning_output(response["text"])

        delta = ReasoningDelta(
            spore_id=self.spore_id,
            task_id=task_id,
            content={
                "hypothesis": parsed.get("hypothesis", ""),
                "evidence": parsed.get("evidence", []),
                "contradictions": parsed.get("contradictions", []),
                "connections": parsed.get("connections", []),
                "primitives": parsed.get("primitives", []),
                "model": response["model"],
                "role": self._role.value,
                "phase": self._cognitive_phase.value,
            },
            confidence=parsed.get("confidence", response.get("confidence", 0.5)),
            parent_deltas=parent_ids,
        )

        fp = self.insert_delta(delta)

        # Advance cognitive phase for next cycle
        phases = list(CognitivePhase)
        current_idx = phases.index(self._cognitive_phase)
        self._cognitive_phase = phases[(current_idx + 1) % len(phases)]

        self.state = SporeState.IDLE
        return delta

    def merge_remote_delta(self, delta: ReasoningDelta, source_spore: str) -> bool:
        """Merge a delta received from another spore via gossip.

        Returns True if the delta was new, False if already known.
        """
        self.state = SporeState.MERGING
        fp = delta.fingerprint()

        if fp in self._delta_store:
            self.state = SporeState.IDLE
            return False

        # Trust check: only accept deltas from trusted peers
        trust = self._lattice.get_trust(source_spore)
        if trust.overall_trust() < 0.1:
            logger.warning(
                "Rejecting delta from untrusted spore %s (trust=%.3f)",
                source_spore, trust.overall_trust(),
            )
            self.state = SporeState.IDLE
            return False

        # Insert into local state
        self._delta_store[fp] = delta
        delta_set = self.get_delta_set(delta.task_id)
        delta_set.add(fp)
        self._merkle.insert_leaf(fp, delta.to_bytes(), source_spore)

        # Update trust lattice: register new peer if unknown
        if source_spore not in self._lattice.known_peers():
            self._lattice = DeltaTrustLattice(
                self.config.spore_id,
                initial_peers=self._lattice.known_peers() | {source_spore},
            )

        self.state = SporeState.IDLE
        logger.debug("Merged delta %s from spore %s", fp, source_spore)
        return True

    def get_task_consensus(self, task_id: str) -> list[ReasoningDelta]:
        """Get all reasoning deltas for a task, sorted by confidence."""
        delta_set = self.get_delta_set(task_id)
        elements = delta_set.value  # ORSet.value is a set property

        deltas = []
        for fp in elements:
            if fp in self._delta_store:
                deltas.append(self._delta_store[fp])

        return sorted(deltas, key=lambda d: d.confidence, reverse=True)

    def state_snapshot(self) -> dict[str, Any]:
        """Export current state for gossip transmission."""
        return {
            "spore_id": self.spore_id,
            "clock": self._clock.logical_time,
            "merkle_root": self._merkle.recompute(),
            "tasks": {
                task_id: {
                    "delta_count": len(dset.value),
                    "fingerprints": list(dset.value),
                }
                for task_id, dset in self._deltas.items()
            },
            "delta_store": {
                fp: d.to_dict() for fp, d in self._delta_store.items()
            },
        }

    async def run(self):
        """Main event loop. Runs forever."""
        logger.info("Spore %s starting on port %d", self.spore_id, self.config.listen_port)

        self._gossip = GossipProtocol(
            spore_id=self.spore_id,
            port=self.config.listen_port,
            seeds=self.config.mesh_seeds,
            on_delta=self.merge_remote_delta,
        )

        self._provider = ProviderRouter(self.config.provider_keys)

        # Start gossip listener
        gossip_task = asyncio.create_task(self._gossip.listen())

        # Start main processing loop
        cycle_task = asyncio.create_task(self._cycle_loop())

        await asyncio.gather(gossip_task, cycle_task)

    async def _cycle_loop(self):
        """Periodic cycle: gossip state, check for new tasks, process."""
        while True:
            try:
                # Gossip current state to peers
                if self._gossip:
                    self.state = SporeState.GOSSIPING
                    await self._gossip.broadcast(self.state_snapshot())

                self.state = SporeState.SLEEPING
                await asyncio.sleep(self.config.cycle_interval)

            except Exception as e:
                logger.error("Cycle error: %s", e)
                await asyncio.sleep(5)

    def metrics(self) -> dict[str, Any]:
        """Return operational metrics for monitoring."""
        return {
            "spore_id": self.spore_id,
            "state": self.state.value,
            "clock": self._clock.logical_time,
            "total_deltas": len(self._delta_store),
            "active_tasks": len(self._active_tasks),
            "completed_tasks": len(self._completed_tasks),
            "merkle_root": self._merkle.recompute(),
            "peer_count": self._lattice.peer_count,
        }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Synapse Brain Spore")
    parser.add_argument("--port", type=int, default=8470)
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seed URLs")
    parser.add_argument("--id", type=str, default=None, help="Spore ID")
    args = parser.parse_args()

    config = SporeConfig.from_env()
    if args.port:
        config.listen_port = args.port
    if args.seeds:
        config.mesh_seeds = [s.strip() for s in args.seeds.split(",")]
    if args.id:
        config.spore_id = args.id

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    spore = Spore(config)
    asyncio.run(spore.run())


if __name__ == "__main__":
    main()
