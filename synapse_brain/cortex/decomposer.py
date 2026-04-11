"""Task decomposition -- breaks complex tasks into spore-sized fragments.

Uses an LLM to analyze a task and produce a decomposition tree:
- Root: the original task
- Branches: subtask categories (research, analysis, synthesis, verification)
- Leaves: atomic reasoning prompts suitable for individual spores

Each leaf becomes a task fragment distributed across the mesh.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskPhase(Enum):
    DECOMPOSE = "decompose"
    DISTRIBUTE = "distribute"
    REASON = "reason"
    CONVERGE = "converge"
    SYNTHESIZE = "synthesize"
    VERIFY = "verify"


@dataclass
class TaskFragment:
    """An atomic unit of work for a single spore."""

    fragment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = ""
    prompt: str = ""
    category: str = "general"
    priority: int = 0
    dependencies: list[str] = field(default_factory=list)
    assigned_spore: str | None = None
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "category": self.category,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "assigned_spore": self.assigned_spore,
            "status": self.status,
        }


@dataclass
class Task:
    """A complete task with its decomposition tree."""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    original_prompt: str = ""
    phase: TaskPhase = TaskPhase.DECOMPOSE
    fragments: list[TaskFragment] = field(default_factory=list)
    synthesis: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_fragments(self) -> int:
        return len(self.fragments)

    @property
    def completed_fragments(self) -> int:
        return sum(1 for f in self.fragments if f.status == "completed")

    @property
    def progress(self) -> float:
        if self.total_fragments == 0:
            return 0.0
        return self.completed_fragments / self.total_fragments


class TaskDecomposer:
    """Decomposes complex tasks into spore-sized fragments.

    Strategy:
    1. Analyze the task to identify reasoning dimensions
    2. Generate diverse perspective prompts for each dimension
    3. Add verification fragments that cross-check other fragments
    4. Return a set of independent fragments + dependency graph
    """

    DECOMPOSITION_PROMPT = """Analyze this task and decompose it into 3-8 independent reasoning subtasks.
Each subtask should be answerable by a single focused reasoning step.
Include at least one verification subtask that cross-checks the others.

Task: {task}

Return a JSON array of objects with fields:
- "prompt": the subtask prompt (be specific and self-contained)
- "category": one of "research", "analysis", "synthesis", "verification", "creative"
- "priority": 0 (highest) to 2 (lowest)
- "dependencies": array of indices of subtasks this depends on (empty for independent)

Return ONLY the JSON array, no other text."""

    def __init__(self, provider_router):
        self._router = provider_router

    async def decompose(self, task_prompt: str) -> Task:
        """Decompose a task into fragments."""
        task = Task(original_prompt=task_prompt)

        try:
            response = await self._router.reason(
                self.DECOMPOSITION_PROMPT.format(task=task_prompt),
                system="You are a task decomposition engine. Return only valid JSON.",
            )

            text = response["text"].strip()
            # Extract JSON from response
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            fragments_data = json.loads(text)

            for i, fdata in enumerate(fragments_data):
                fragment = TaskFragment(
                    task_id=task.task_id,
                    prompt=fdata["prompt"],
                    category=fdata.get("category", "general"),
                    priority=fdata.get("priority", 1),
                    dependencies=[
                        fragments_data[d]["prompt"][:20]
                        for d in fdata.get("dependencies", [])
                        if d < len(fragments_data)
                    ],
                )
                task.fragments.append(fragment)

            task.phase = TaskPhase.DISTRIBUTE
            logger.info(
                "Decomposed task %s into %d fragments",
                task.task_id, len(task.fragments),
            )

        except json.JSONDecodeError:
            # Fallback: treat the whole task as a single fragment
            logger.warning("Failed to decompose task, using single fragment")
            task.fragments.append(TaskFragment(
                task_id=task.task_id,
                prompt=task_prompt,
                category="general",
            ))
            task.phase = TaskPhase.DISTRIBUTE

        except Exception as e:
            logger.error("Decomposition failed: %s", e)
            task.fragments.append(TaskFragment(
                task_id=task.task_id,
                prompt=task_prompt,
                category="general",
            ))
            task.phase = TaskPhase.DISTRIBUTE

        return task


class TaskSynthesizer:
    """Synthesizes converged reasoning deltas into a final answer.

    Takes all reasoning deltas for a task, weighs them by confidence
    and trust scores, and produces a unified synthesis.
    """

    SYNTHESIS_PROMPT = """You are synthesizing reasoning from multiple distributed agents.
Each agent reasoned about a subtask independently. Their outputs follow.

{reasoning_context}

Produce a unified, coherent answer to the original task:
{original_task}

Weight contributions by stated confidence. Resolve contradictions by
preferring higher-confidence reasoning. Flag any unresolved disagreements."""

    def __init__(self, provider_router):
        self._router = provider_router

    async def synthesize(self, task: Task, deltas: list[dict[str, Any]]) -> str:
        """Produce a final synthesis from converged reasoning deltas."""
        context_parts = []
        for d in sorted(deltas, key=lambda x: x.get("confidence", 0), reverse=True):
            spore_id = d.get("spore_id", "unknown")
            confidence = d.get("confidence", 0)
            content = d.get("content", {})
            reasoning = content.get("reasoning", str(content))
            context_parts.append(
                f"[Spore {spore_id}] (confidence={confidence:.2f}):\n{reasoning}"
            )

        context = "\n\n---\n\n".join(context_parts)

        response = await self._router.reason(
            self.SYNTHESIS_PROMPT.format(
                reasoning_context=context,
                original_task=task.original_prompt,
            ),
        )

        task.synthesis = response["text"]
        task.confidence = sum(d.get("confidence", 0) for d in deltas) / max(len(deltas), 1)
        task.phase = TaskPhase.VERIFY

        return response["text"]
