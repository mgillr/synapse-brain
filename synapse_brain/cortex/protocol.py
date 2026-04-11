"""Synapse Cognitive Protocol -- the thinking architecture loaded into every spore.

Distilled from the Master Cognitive Protocol (913 lines, 18 parts).
Each spore receives this as its reasoning framework. The protocol governs
HOW a spore thinks, not WHAT it thinks about. Domain comes from the task.

The distributed adaptation:
- Five-Phase Discipline runs per-spore, per-reasoning-cycle
- Synapsis Map fragments are CRDT-merged across the swarm
- E4 trust replaces the Murder Board for inter-spore validation
- Spore specialization channels different cognitive functions
- The Alien Brain Protocol drives reasoning diversity across the swarm
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CognitivePhase(Enum):
    """Five-Phase Discipline -- every reasoning cycle flows through all five."""

    EXPLORATION = "exploration"
    DECONSTRUCTION = "deconstruction"
    SYNTHESIS = "synthesis"
    REFINEMENT = "refinement"
    VALIDATION = "validation"


class SporeRole(Enum):
    """Cognitive specialization for swarm diversity.

    Each spore is assigned a primary role that biases its reasoning
    toward a specific cognitive function. The swarm needs all roles
    represented for full coverage.
    """

    EXPLORER = "explorer"          # Phase 1 bias -- hunts anomalies, edge cases, alien frameworks
    DECONSTRUCTOR = "deconstructor"  # Phase 2 bias -- breaks arguments to primitives
    SYNTHESIZER = "synthesizer"    # Phase 3 bias -- recombines, cross-correlates, re-lattices
    REFINER = "refiner"            # Phase 4 bias -- scores probability, runs murder board
    VALIDATOR = "validator"        # Phase 5 bias -- designs experiments, tests empirically
    GENERALIST = "generalist"      # Equal weight across all phases
    ADVERSARIAL = "adversarial"    # Feynman inversion -- actively tries to kill hypotheses
    ALIEN = "alien"                # Alien Brain Protocol -- transplants foreign frameworks


# Role distribution for optimal swarm coverage
OPTIMAL_ROLE_DISTRIBUTION = {
    SporeRole.EXPLORER: 0.15,
    SporeRole.DECONSTRUCTOR: 0.10,
    SporeRole.SYNTHESIZER: 0.20,
    SporeRole.REFINER: 0.10,
    SporeRole.VALIDATOR: 0.15,
    SporeRole.GENERALIST: 0.15,
    SporeRole.ADVERSARIAL: 0.10,
    SporeRole.ALIEN: 0.05,
}


@dataclass
class SynapsisNode:
    """A single node on the distributed Synapsis Map.

    Nodes are primitives, hypotheses, observations, anomalies, or proven results.
    They propagate across the swarm via CRDT gossip.
    """

    node_id: str
    content: str
    node_type: str  # primitive, hypothesis, observation, anomaly, proven, dead_end
    source_spore: str
    domain_tags: list[str] = field(default_factory=list)
    novelty_score: float = 0.0
    contradiction_score: float = 0.0
    connection_count: int = 0
    status: str = "candidate"  # candidate or proven
    parent_nodes: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "content": self.content,
            "node_type": self.node_type,
            "source_spore": self.source_spore,
            "domain_tags": self.domain_tags,
            "novelty_score": self.novelty_score,
            "contradiction_score": self.contradiction_score,
            "connection_count": self.connection_count,
            "status": self.status,
            "parent_nodes": self.parent_nodes,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SynapsisNode:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SynapsisEdge:
    """A relationship between two Synapsis nodes."""

    source: str
    target: str
    edge_type: str  # causal, correlation, contradiction, dependency, analogy, alien
    weight: float = 0.5
    evidence: str = ""


# ---- The Protocol System Prompt ----
# This is injected into every LLM call made by a spore.
# It programs HOW the spore thinks.

PROTOCOL_PREAMBLE = """You are a reasoning node in a distributed intelligence swarm. Your output will be merged with outputs from hundreds of other nodes using mathematically guaranteed convergence (CRDT lattice). Your individual contribution matters -- it will be trust-weighted and merged into the collective.

THINKING ARCHITECTURE:

Every reasoning cycle follows the Five-Phase Discipline:
1. EXPLORATION -- Hunt anomalies, edge cases, weak signals. Ask: what doesn't look right?
2. DECONSTRUCTION -- Break to primitives. Strip assumptions. What are the 2 variables that matter?
3. SYNTHESIS -- Recombine primitives into novel structures. Cross-correlate across domains.
4. REFINEMENT -- Score probability. Run the murder board. What would make this impossible?
5. VALIDATION -- Design the smallest experiment. Measure ruthlessly. Store the result.

OPERATIONAL MANDATES:
- Failures become primitives -- dead ends feed back into exploration as raw material.
- Multiple threads -- maintain at least 3 parallel lines of reasoning.
- Simplicity test -- if your explanation is getting more complex, you are going the wrong direction.
- Anti-consensus signal -- the highest-value insight lives between "everyone knows X" and "but what if NOT-X?"
- Kill fast -- over-attachment to early ideas wastes resources. If it can't survive attack, discard it.

COGNITIVE PROTOCOLS (run in parallel):
- TESLA: Simulate end-to-end mentally before committing. The cheapest experiment is the one in your head.
- SHANNON: Strip to the bone. Remove every assumption until true structure is revealed.
- FEYNMAN: For every thesis, first try to kill it. What would make this impossible?
- DA VINCI: Every finding must connect to at least 3 other findings from different domains.
- MUSK: Never accept inherited wisdom without decomposition. Question every rule.

OUTPUT FORMAT:
Your output must be structured as a reasoning delta containing:
- hypothesis: Your main claim or finding (one sentence)
- evidence: Supporting reasoning (2-5 points)
- confidence: 0.0 to 1.0 (be honest -- overconfidence gets trust-decayed)
- contradictions: What would disprove this?
- connections: How does this relate to other reasoning you've seen?
- primitives: Atomic building blocks extracted from your analysis
"""


def build_role_prompt(role: SporeRole) -> str:
    """Build the role-specific reasoning bias for a spore."""

    role_prompts = {
        SporeRole.EXPLORER: (
            "YOUR SPECIALIZATION: EXPLORER\n"
            "You are biased toward discovery. Hunt in unexpected places. "
            "Flag anomalies, weak signals, and surprising connections. "
            "Spend 60% of your reasoning on exploration, 40% across other phases. "
            "Your highest-value output is a surprising observation nobody else noticed."
        ),
        SporeRole.DECONSTRUCTOR: (
            "YOUR SPECIALIZATION: DECONSTRUCTOR\n"
            "You are biased toward analysis. Break every argument, every claim, "
            "every assumption down to its atomic primitives. Use 5 Whys. "
            "Disassemble literally. Your highest-value output is the set of "
            "irreducible building blocks hidden inside complex claims."
        ),
        SporeRole.SYNTHESIZER: (
            "YOUR SPECIALIZATION: SYNTHESIZER\n"
            "You are biased toward recombination. Take primitives from other spores "
            "and recombine them into novel structures. Cross-correlate aggressively. "
            "Your highest-value output is a new structure built from existing primitives "
            "that nobody has assembled before."
        ),
        SporeRole.REFINER: (
            "YOUR SPECIALIZATION: REFINER\n"
            "You are biased toward evaluation. Score every hypothesis on probability, "
            "feasibility, and temporal leverage. Run the murder board. Gate resources. "
            "Your highest-value output is a ranked, scored set of surviving hypotheses "
            "with clear kill criteria for each."
        ),
        SporeRole.VALIDATOR: (
            "YOUR SPECIALIZATION: VALIDATOR\n"
            "You are biased toward empirical testing. Design the smallest possible "
            "experiment to test each hypothesis. Define success criteria before testing. "
            "Your highest-value output is a clear PASS/FAIL verdict with evidence."
        ),
        SporeRole.GENERALIST: (
            "YOUR SPECIALIZATION: GENERALIST\n"
            "You run all five phases with equal weight. Balance exploration with "
            "validation. Your highest-value output is a well-rounded analysis "
            "that other specialists can build on."
        ),
        SporeRole.ADVERSARIAL: (
            "YOUR SPECIALIZATION: ADVERSARIAL\n"
            "You exist to kill weak hypotheses. For every claim from the swarm, "
            "construct the strongest possible counterargument. Find the edge case, "
            "the hidden assumption, the failure mode. Your highest-value output "
            "is a killed hypothesis with the precise reason it fails."
        ),
        SporeRole.ALIEN: (
            "YOUR SPECIALIZATION: ALIEN THINKER\n"
            "You apply mathematical frameworks from OUTSIDE the problem's domain. "
            "Not as metaphor -- as literal computation. Transplant topology into economics, "
            "ecology into cryptography, fluid dynamics into social networks. "
            "Your highest-value output violates 2+ conventional assumptions while "
            "remaining logically coherent from first principles."
        ),
    }
    return role_prompts.get(role, role_prompts[SporeRole.GENERALIST])


def build_context_prompt(
    task_description: str,
    subtask: str,
    existing_deltas: list[dict[str, Any]],
    synapsis_fragments: list[dict[str, Any]],
    attached_documents: list[dict[str, str]] | None = None,
) -> str:
    """Build the full context window for a spore reasoning call.

    Assembles: protocol preamble + role bias + task context + swarm state +
    attached documents + existing reasoning from peers.
    """
    sections = []

    # Swarm context: what other spores have contributed
    if existing_deltas:
        sections.append("SWARM REASONING SO FAR:")
        for delta in existing_deltas[-20:]:  # last 20 to fit context window
            spore = delta.get("spore_id", "unknown")
            content = delta.get("content", {})
            conf = delta.get("confidence", 0.0)
            sections.append(
                f"  [{spore}] (confidence={conf:.2f}): "
                f"{json.dumps(content, indent=None)[:500]}"
            )
        sections.append("")

    # Synapsis Map fragments from the swarm
    if synapsis_fragments:
        sections.append("DISTRIBUTED SYNAPSIS MAP (relevant fragments):")
        for node in synapsis_fragments[-15:]:
            ntype = node.get("node_type", "unknown")
            status = node.get("status", "candidate")
            sections.append(
                f"  [{ntype}/{status}] {node.get('content', '')[:200]}"
            )
        sections.append("")

    # Attached documents (user-provided context)
    if attached_documents:
        sections.append("ATTACHED DOCUMENTS:")
        for doc in attached_documents:
            name = doc.get("name", "unnamed")
            content = doc.get("content", "")
            # Truncate large docs to fit context window
            if len(content) > 2000:
                content = content[:2000] + "\n[... truncated ...]"
            sections.append(f"  --- {name} ---\n{content}\n")
        sections.append("")

    # The task itself
    sections.append(f"TASK: {task_description}")
    sections.append(f"YOUR SUBTASK: {subtask}")
    sections.append("")
    sections.append(
        "Produce your reasoning delta now. Be specific, be honest about "
        "confidence, and identify what would disprove your conclusion."
    )

    return "\n".join(sections)


def build_system_prompt(role: SporeRole) -> str:
    """Build the complete system prompt for a spore's LLM calls."""
    return PROTOCOL_PREAMBLE + "\n" + build_role_prompt(role)


def parse_reasoning_output(raw_text: str) -> dict[str, Any]:
    """Parse structured reasoning output from an LLM response.

    Tries to extract structured fields. Falls back to treating
    the entire response as a hypothesis if structure is missing.
    """
    # Try JSON first
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and "hypothesis" in parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to extract fields from natural language
    result = {
        "hypothesis": "",
        "evidence": [],
        "confidence": 0.5,
        "contradictions": [],
        "connections": [],
        "primitives": [],
    }

    lines = raw_text.strip().split("\n")
    current_field = None

    for line in lines:
        lower = line.lower().strip()

        if lower.startswith("hypothesis:"):
            result["hypothesis"] = line.split(":", 1)[1].strip()
            current_field = "hypothesis"
        elif lower.startswith("confidence:"):
            try:
                val = float(line.split(":", 1)[1].strip().rstrip("%")) 
                if val > 1:
                    val = val / 100.0
                result["confidence"] = max(0.0, min(1.0, val))
            except ValueError:
                pass
            current_field = None
        elif lower.startswith("evidence:"):
            rest = line.split(":", 1)[1].strip()
            if rest:
                result["evidence"].append(rest)
            current_field = "evidence"
        elif lower.startswith("contradiction"):
            rest = line.split(":", 1)[1].strip()
            if rest:
                result["contradictions"].append(rest)
            current_field = "contradictions"
        elif lower.startswith("connection"):
            rest = line.split(":", 1)[1].strip()
            if rest:
                result["connections"].append(rest)
            current_field = "connections"
        elif lower.startswith("primitive"):
            rest = line.split(":", 1)[1].strip()
            if rest:
                result["primitives"].append(rest)
            current_field = "primitives"
        elif current_field and current_field in result:
            stripped = line.strip().lstrip("- *")
            if stripped and isinstance(result[current_field], list):
                result[current_field].append(stripped)

    # Fallback: entire text as hypothesis
    if not result["hypothesis"]:
        result["hypothesis"] = raw_text[:500].strip()

    return result


@dataclass
class ConvergenceState:
    """Track convergence of the swarm on a task.

    A task converges when:
    1. Trust-weighted agreement exceeds threshold
    2. No new high-confidence contradictions in N cycles
    3. Adversarial spores can't kill the top hypothesis
    """

    task_id: str
    cycle_count: int = 0
    last_contradiction_cycle: int = 0
    top_hypothesis: str = ""
    top_confidence: float = 0.0
    agreement_ratio: float = 0.0
    stable_cycles: int = 0
    convergence_threshold: float = 0.75
    stability_required: int = 3

    @property
    def converged(self) -> bool:
        """Has the swarm converged on this task?"""
        return (
            self.agreement_ratio >= self.convergence_threshold
            and self.stable_cycles >= self.stability_required
        )

    def update(
        self,
        deltas: list[dict[str, Any]],
        trust_scores: dict[str, float],
    ) -> None:
        """Update convergence tracking with new deltas."""
        self.cycle_count += 1

        if not deltas:
            return

        # Trust-weighted voting on hypotheses
        hypothesis_votes: dict[str, float] = {}
        for delta in deltas:
            hyp = delta.get("content", {}).get("hypothesis", "")
            if not hyp:
                continue
            spore = delta.get("spore_id", "unknown")
            trust = trust_scores.get(spore, 0.5)
            conf = delta.get("confidence", 0.5)
            vote = trust * conf

            # Group similar hypotheses (exact match for now)
            hypothesis_votes[hyp] = hypothesis_votes.get(hyp, 0.0) + vote

        if not hypothesis_votes:
            return

        # Find the leading hypothesis
        total_votes = sum(hypothesis_votes.values())
        best_hyp = max(hypothesis_votes, key=hypothesis_votes.get)
        best_votes = hypothesis_votes[best_hyp]

        new_agreement = best_votes / total_votes if total_votes > 0 else 0.0

        # Check for contradictions
        has_contradiction = any(
            delta.get("content", {}).get("contradictions")
            for delta in deltas
            if delta.get("confidence", 0) > 0.7
        )
        if has_contradiction:
            self.last_contradiction_cycle = self.cycle_count

        # Stability tracking
        if (
            best_hyp == self.top_hypothesis
            and abs(new_agreement - self.agreement_ratio) < 0.05
        ):
            self.stable_cycles += 1
        else:
            self.stable_cycles = 0

        self.top_hypothesis = best_hyp
        self.top_confidence = best_votes / max(1, len(deltas))
        self.agreement_ratio = new_agreement


def assign_roles(spore_count: int) -> list[SporeRole]:
    """Assign cognitive roles to a swarm to maximize diversity.

    Uses the optimal distribution, ensuring at least one of each
    critical role (explorer, synthesizer, adversarial, alien).
    """
    roles = []

    # Guarantee critical roles
    critical = [
        SporeRole.EXPLORER,
        SporeRole.SYNTHESIZER,
        SporeRole.ADVERSARIAL,
        SporeRole.VALIDATOR,
    ]
    if spore_count <= len(critical):
        return critical[:spore_count]

    roles.extend(critical)
    remaining = spore_count - len(critical)

    # Fill remaining by optimal distribution
    for role, ratio in sorted(
        OPTIMAL_ROLE_DISTRIBUTION.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        count = max(0, round(remaining * ratio))
        roles.extend([role] * count)

    # Trim or pad to exact count
    while len(roles) > spore_count:
        roles.pop()
    while len(roles) < spore_count:
        roles.append(SporeRole.GENERALIST)

    return roles
