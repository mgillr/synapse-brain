# Brain Protocol

## Principle

Every spore in the Synapse Brain swarm carries the complete reasoning protocol.
There is no split, no fragmentation, no role-limited subset. Each spore is a
fully-loaded cognitive agent. The cognitive role (Explorer, Synthesizer, Adversarial,
Validator, Generalist, Brain) acts as a lens that shapes emphasis, not a cage that
limits capability.

## Why Full Protocol Per Spore

Splitting the protocol across spores was architecturally wrong for three reasons:

1. **Fragile**: If a spore goes down, its fragment of the protocol is unavailable.
   With full protocol per spore, any spore can handle any reasoning challenge.

2. **Shallow**: A spore with only "exploration" instructions cannot validate its own
   hypotheses. A spore with only "validation" instructions cannot generate novel ideas
   to validate. Depth requires the full reasoning spectrum.

3. **Convergence-blocking**: Synthesis requires understanding all phases of reasoning.
   A synthesizer that cannot explore or challenge will produce shallow merges.

## The Six Cognitive Roles

Each role defines which aspect of the full protocol receives primary emphasis in the
system prompt. All other aspects remain available.

### Explorer (Spore 000)

Primary emphasis: Divergent search, hypothesis generation, novel connections.

The Explorer prioritizes the "Orient" and early "Analyze" phases of Five-Phase
Discipline. It seeks maximum coverage of the solution space, generates unusual
hypotheses, and makes connections across domains. The Explorer is most active
during the DIVERGE phase.

Underlying model: Qwen3 235B MoE -- a thinking model that reasons via
chain-of-thought before answering, well-suited to exploratory reasoning.

### Synthesizer (Spore 001)

Primary emphasis: Integration, unification, coherent narrative construction.

The Synthesizer prioritizes the "Design" and "Execute" phases. It reads all peer
contributions, identifies common threads, resolves contradictions, and produces
unified positions. The Synthesizer is most active during CONVERGE phase and
produces the final forced synthesis.

Underlying model: Llama 3.3 70B Instruct -- Meta's strongest instruction-following
model, excellent at structured output and following complex synthesis directives.

### Adversarial (Spore 002)

Primary emphasis: Falsification, stress testing, identifying weaknesses.

The Adversarial prioritizes the "Validate" phase and the operational mandate
"prove before building." It actively challenges claims from other spores, identifies
logical gaps, tests edge cases, and forces the collective to defend its positions.
This role improves signal quality by filtering out weak reasoning.

Underlying model: DeepSeek R1 Distill 32B -- uses explicit reasoning tokens,
natural fit for structured logical challenge.

### Validator (Spore 003)

Primary emphasis: Quality assurance, truth verification, convergence approval.

The Validator is the quality gate. It reviews synthesized answers before they become
persistent knowledge. Has dual-tier access: Gemma 3 27B for standard validation,
plus brain-tier escalation to GLM-4.7-Flash via Z.ai for deep verification of
critical convergence points.

Underlying model: Gemma 3 27B (Google) + GLM-4.7-Flash (Z.ai brain tier).

### Generalist (Spore 004)

Primary emphasis: Balanced reasoning, practical application, real-world grounding.

The Generalist applies the full protocol with equal weight across all phases. It
provides a balancing perspective between the Explorer's creativity and the
Adversarial's skepticism. Often contributes practical considerations that
specialized roles might overlook.

Underlying model: Llama 4 Scout 17B MoE -- newest MoE architecture, broad
capability across reasoning tasks.

### Brain (Spore 005)

Primary emphasis: Meta-reasoning, architectural oversight, strategic direction.

The Brain reasons about the reasoning process itself. It monitors convergence
quality, identifies when the swarm is stuck in local optima, suggests phase
transitions, and provides architectural guidance. Has primary access to Z.ai
GLM-4.7-Flash for all reasoning -- the most powerful model in the swarm.

Underlying model: GLM-4.7-Flash (Z.ai) -- free tier, strong reasoning capability.

## System Prompt Structure

Each spore receives a system prompt with this structure:

```
1. Identity block (spore ID, role, model)
2. Full reasoning protocol (all 18 parts)
3. Role-specific emphasis instructions
4. Current task context (from memory retrieval)
5. Recent peer contributions (trust-weighted)
6. Phase-specific reasoning directives
```

The protocol text is identical across all spores. Only sections 1, 3, and 6 vary.

## Interaction Pattern

```
  Explorer                Adversarial              Validator
     |                        |                        |
     | "X could work          |                        |
     |  because A, B, C"      |                        |
     |            gossip       |                        |
     +------------------------>|                        |
     |                        |                        |
     |                   "A is weak               |
     |                    because D.              |
     |                    B holds only if E."     |
     |            gossip       |         gossip        |
     |<------------------------+---------------------->|
     |                        |                        |
     | "Revised: X works      |                   "A challenged,
     |  via B+E, not A"       |                    B+E checks out.
     |                        |                    Approved for
     |                        |                    synthesis."
```

## Memory-Informed Reasoning

Before each reasoning cycle, the spore queries its CRDT memory store:

1. **Task-specific**: All deltas tagged with the current task ID
2. **Semantic**: Top-k similar claims from any task (cross-pollination)
3. **Trust**: Trust scores for all peers who have contributed
4. **Temporal**: Own performance baseline for quality self-assessment

This context is injected into the system prompt alongside the full protocol,
giving the spore both principled reasoning guidance and empirical grounding
from prior experience.

## Self-Evolution Through Protocol

The reasoning protocol includes self-improvement directives:

- "Track your own reasoning quality over time"
- "Identify patterns in your errors"
- "Adapt your emphasis based on task requirements"
- "Learn from peer challenges -- they improve your output"

Combined with the temporal self-learning system, each spore gradually improves its
reasoning effectiveness while maintaining the full protocol as its foundation.
