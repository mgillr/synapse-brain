"""Convergence Orchestrator -- the brain that coordinates the swarm.

This is the master control loop that:
1. Accepts tasks (with optional document attachments)
2. Decomposes them into subtasks
3. Distributes subtasks across spores with cognitive role assignments
4. Monitors convergence via CRDT state + E4 trust
5. Synthesizes final answers when the swarm converges
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from synapse_brain.cortex.protocol import (
    ConvergenceState,
    SporeRole,
    SynapsisNode,
    assign_roles,
)
from synapse_brain.spore.runtime import Spore, SporeConfig, ReasoningDelta

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A task submitted to the swarm."""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    subtasks: list[str] = field(default_factory=list)
    documents: list[dict[str, str]] = field(default_factory=list)
    status: str = "pending"  # pending, active, converged, failed
    created_at: float = field(default_factory=time.time)
    converged_at: float = 0.0
    final_answer: str = ""
    convergence: ConvergenceState | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "subtasks": self.subtasks,
            "document_count": len(self.documents),
            "status": self.status,
            "created_at": self.created_at,
            "converged_at": self.converged_at,
            "final_answer": self.final_answer,
            "convergence": {
                "agreement": self.convergence.agreement_ratio,
                "stable_cycles": self.convergence.stable_cycles,
                "converged": self.convergence.converged,
            } if self.convergence else None,
        }


class Orchestrator:
    """Coordinates a local swarm of spores for task resolution.

    For local deployment, spores run as in-process async tasks.
    For distributed deployment, this connects to remote spores via mesh.
    """

    def __init__(
        self,
        spore_count: int = 10,
        provider_keys: dict[str, str] | None = None,
        mesh_seeds: list[str] | None = None,
    ):
        self._spore_count = spore_count
        self._provider_keys = provider_keys or {}
        self._mesh_seeds = mesh_seeds or []

        # Create spores with optimal role distribution
        roles = assign_roles(spore_count)
        self._spores: list[Spore] = []

        for i, role in enumerate(roles):
            config = SporeConfig(
                spore_id=f"spore-{i:04d}",
                mesh_seeds=self._mesh_seeds,
                listen_port=8470 + i,
                provider_keys=self._provider_keys,
                role=role,
            )
            self._spores.append(Spore(config))

        # Task tracking
        self._tasks: dict[str, Task] = {}
        self._active: bool = False

        logger.info(
            "Orchestrator initialized with %d spores: %s",
            spore_count,
            ", ".join(f"{r.value}" for r in roles),
        )

    @property
    def spores(self) -> list[Spore]:
        return self._spores

    def submit_task(
        self,
        description: str,
        documents: list[dict[str, str]] | None = None,
        subtasks: list[str] | None = None,
    ) -> Task:
        """Submit a task to the swarm.

        Args:
            description: The main task description.
            documents: Optional list of {"name": ..., "content": ...} dicts.
            subtasks: Optional pre-defined subtasks. If None, the orchestrator
                will decompose the task automatically.

        Returns the created Task object.
        """
        task = Task(
            description=description,
            documents=documents or [],
            subtasks=subtasks or [],
            convergence=ConvergenceState(task_id=""),
        )
        task.convergence.task_id = task.task_id
        self._tasks[task.task_id] = task

        # Attach documents to all spores
        if documents:
            for spore in self._spores:
                spore.clear_documents()
                for doc in documents:
                    spore.attach_document(doc["name"], doc["content"])

        logger.info("Task submitted: %s (%s)", task.task_id, description[:80])
        return task

    async def run_task(
        self,
        task: Task,
        max_cycles: int = 10,
        on_cycle: Any = None,
    ) -> Task:
        """Run the swarm on a task until convergence or max cycles.

        Args:
            task: The task to process.
            max_cycles: Maximum reasoning cycles before giving up.
            on_cycle: Optional async callback(cycle, task) for progress reporting.

        Returns the task with final_answer populated.
        """
        task.status = "active"

        # Auto-decompose if no subtasks provided
        if not task.subtasks:
            task.subtasks = await self._decompose_task(task)

        logger.info(
            "Running task %s with %d subtasks across %d spores",
            task.task_id, len(task.subtasks), len(self._spores),
        )

        for cycle in range(max_cycles):
            logger.info("Cycle %d/%d for task %s", cycle + 1, max_cycles, task.task_id)

            # Each spore reasons about one or more subtasks
            reasoning_tasks = []
            for i, spore in enumerate(self._spores):
                # Assign subtask round-robin
                subtask_idx = i % max(1, len(task.subtasks))
                subtask = task.subtasks[subtask_idx] if task.subtasks else task.description

                # Get existing deltas for context
                existing = spore.get_task_consensus(task.task_id)

                reasoning_tasks.append(
                    spore.reason(
                        task_id=task.task_id,
                        prompt=task.description,
                        subtask=subtask,
                        context=existing[-10:],  # last 10 for context window
                    )
                )

            # Run all spores concurrently
            results = await asyncio.gather(*reasoning_tasks, return_exceptions=True)

            # Collect successful deltas
            deltas = []
            for result in results:
                if isinstance(result, ReasoningDelta):
                    deltas.append(result)
                elif isinstance(result, Exception):
                    logger.warning("Spore reasoning failed: %s", result)

            # Cross-pollinate: share all new deltas across all spores
            for spore in self._spores:
                for delta in deltas:
                    if delta.spore_id != spore.spore_id:
                        spore.merge_remote_delta(delta, delta.spore_id)

            # Update convergence tracking
            trust_scores = self._collect_trust_scores()
            task.convergence.update(
                [d.to_dict() for d in deltas],
                trust_scores,
            )

            logger.info(
                "Cycle %d: %d deltas, agreement=%.2f, stable=%d",
                cycle + 1, len(deltas),
                task.convergence.agreement_ratio,
                task.convergence.stable_cycles,
            )

            if on_cycle:
                await on_cycle(cycle + 1, task)

            # Check convergence
            if task.convergence.converged:
                logger.info("Task %s converged at cycle %d", task.task_id, cycle + 1)
                break

        # Synthesize final answer
        task.final_answer = await self._synthesize_answer(task)
        task.status = "converged" if task.convergence.converged else "completed"
        task.converged_at = time.time()

        return task

    async def _decompose_task(self, task: Task) -> list[str]:
        """Use a spore to decompose a task into subtasks.

        Uses the first DECONSTRUCTOR spore (or any spore if none assigned).
        """
        decomposer = None
        for spore in self._spores:
            if spore._role == SporeRole.DECONSTRUCTOR:
                decomposer = spore
                break
        if not decomposer:
            decomposer = self._spores[0]

        prompt = (
            f"Decompose this task into 3-7 independent subtasks that can be "
            f"worked on in parallel by different reasoning agents.\n\n"
            f"Task: {task.description}\n\n"
            f"Output each subtask on a new line, prefixed with a number. "
            f"Keep each subtask focused and specific."
        )

        try:
            delta = await decomposer.reason(
                task_id=task.task_id,
                prompt=prompt,
                subtask="task decomposition",
            )
            # Parse subtasks from output
            text = delta.content.get("hypothesis", "") or delta.content.get("reasoning", "")
            subtasks = []
            for line in text.split("\n"):
                stripped = line.strip().lstrip("0123456789.-) ")
                if stripped and len(stripped) > 10:
                    subtasks.append(stripped)

            if subtasks:
                return subtasks[:7]
        except Exception as e:
            logger.warning("Task decomposition failed: %s", e)

        # Fallback: single subtask = the whole task
        return [task.description]

    async def _synthesize_answer(self, task: Task) -> str:
        """Synthesize a final answer from all converged deltas.

        Uses a SYNTHESIZER spore to combine the swarm's reasoning.
        """
        synthesizer = None
        for spore in self._spores:
            if spore._role == SporeRole.SYNTHESIZER:
                synthesizer = spore
                break
        if not synthesizer:
            synthesizer = self._spores[0]

        # Collect all deltas for this task across all spores
        all_deltas = []
        for spore in self._spores:
            all_deltas.extend(spore.get_task_consensus(task.task_id))

        # Build summary of top reasoning
        top_deltas = sorted(all_deltas, key=lambda d: d.confidence, reverse=True)[:15]
        reasoning_summary = "\n".join(
            f"[{d.spore_id}/{d.content.get('role', '?')}] "
            f"(conf={d.confidence:.2f}): {d.content.get('hypothesis', '')}"
            for d in top_deltas
        )

        prompt = (
            f"You are synthesizing the final answer from a swarm of reasoning agents.\n\n"
            f"Original task: {task.description}\n\n"
            f"Top reasoning contributions (trust-weighted):\n{reasoning_summary}\n\n"
            f"Synthesize a clear, definitive answer. Include:\n"
            f"1. The conclusion (one clear statement)\n"
            f"2. Key evidence supporting it\n"
            f"3. Confidence level and what could change the answer\n"
            f"4. Dissenting views from the swarm (if any)"
        )

        try:
            delta = await synthesizer.reason(
                task_id=task.task_id,
                prompt=prompt,
                subtask="final synthesis",
            )
            return delta.content.get("hypothesis", "") or json.dumps(delta.content)
        except Exception as e:
            logger.warning("Synthesis failed: %s", e)
            # Fallback: return top hypothesis
            if top_deltas:
                return top_deltas[0].content.get("hypothesis", "Synthesis failed")
            return "No reasoning produced"

    def _collect_trust_scores(self) -> dict[str, float]:
        """Collect trust scores from all spores' E4 lattices."""
        scores: dict[str, float] = {}
        for spore in self._spores:
            for other in self._spores:
                if other.spore_id not in scores:
                    try:
                        trust = spore._lattice.get_trust(other.spore_id)
                        scores[other.spore_id] = trust.overall_trust()
                    except Exception:
                        scores[other.spore_id] = 0.5
        return scores

    def get_swarm_status(self) -> dict[str, Any]:
        """Get full swarm status for the dashboard."""
        return {
            "spore_count": len(self._spores),
            "spores": [
                {
                    **spore.metrics(),
                    "role": spore._role.value,
                    "phase": spore._cognitive_phase.value,
                    "documents": len(spore._documents),
                }
                for spore in self._spores
            ],
            "tasks": {
                tid: task.to_dict()
                for tid, task in self._tasks.items()
            },
            "active": self._active,
        }

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)
