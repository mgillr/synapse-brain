# Synapse Brain Architecture

## Overview

Synapse Brain is a distributed reasoning system where autonomous agents (spores) form
a gossip mesh, reason independently using different LLM families, and converge on
synthesized answers through CRDT-backed persistent memory and trust-weighted integration.

The system is designed around three invariants:

1. **Nothing is ever forgotten** -- CRDT OR-Set with add-wins semantics
2. **Every perspective is genuinely independent** -- different model families per spore
3. **Trust is earned, not assigned** -- E4 DeltaTrustLattice tracks interaction quality

## System Layers

### Layer 1: Gossip Mesh

Spores communicate through HTTP gossip over a fully connected mesh. Every spore
knows every other spore's URL and authenticates with HF_TOKEN (for private Spaces).

Gossip cycle (every 20 seconds):
1. Select a random peer
2. Exchange health + clock + delta count
3. Sync memory state (new deltas since last sync)
4. Update trust scores based on peer interaction
5. Propagate any active tasks and their reasoning state

The mesh is eventually consistent. A delta produced by any spore reaches all other
spores within 2-3 gossip cycles (40-60 seconds).

### Layer 2: CRDT Memory

All persistent state is stored in `crdt-merge` data structures:

| Structure | Purpose | Semantics |
|-----------|---------|-----------|
| `ORSet` | Reasoning deltas, claims, memories | Add-wins, merge-idempotent |
| `MerkleTree` | Memory integrity verification | Hash-chain provenance |
| `VectorClock` | Causal ordering of events | Lamport timestamps per spore |
| `DeltaTrustLattice` | Inter-spore trust scores | Monotonic trust deltas |

Memory is organized into four stores:
- **reasoning**: Task-specific reasoning deltas with claims, confidence, phase
- **knowledge**: Persistent facts extracted from converged syntheses
- **trust_events**: Record of every trust-affecting interaction
- **temporal**: Self-observation data (cycle times, error rates, quality scores)

The sidecar indexer runs as a background thread, continuously building embedding
indices over the reasoning and knowledge stores. When a spore needs context for a
new reasoning cycle, it retrieves by semantic similarity -- not by recency or
position in a context window.

### Layer 3: Cognitive Protocol

Every spore receives the complete Master Cognitive Protocol (913 lines, 18 parts).
This includes:

- **Five-Phase Discipline**: Orient, Analyze, Design, Execute, Validate
- **11 Operational Mandates**: Including "never guess", "prove before building",
  "understand why before fixing"
- **Three-Tier Synapse**: Reactive (fast response), Analytical (deep reasoning),
  Strategic (long-term planning)

The cognitive role (Explorer, Synthesizer, Adversarial, Validator, Generalist, Brain)
determines which aspects of the protocol are emphasized in the system prompt, but
every spore has access to the full protocol.

### Layer 4: Trust and Quality

The E4 DeltaTrustLattice manages trust between spores:

```
Trust Score = f(interaction_history, contribution_quality, convergence_alignment)
```

Trust affects:
- **Gossip priority**: High-trust peers are synced first
- **Synthesis weighting**: High-trust contributions carry more weight
- **Anomaly detection**: Sudden trust drops trigger investigation

The Validator spore (003) has brain-tier access via Z.ai GLM-4.7-Flash for deep
quality assessment. When convergence is reached, the Validator can escalate to the
brain tier to verify the synthesis before it becomes part of persistent knowledge.

### Layer 5: Temporal Self-Learning

Each spore maintains a temporal baseline of its own operational metrics:

- Average cycle time and standard deviation
- Error rate distribution
- Reasoning quality scores (assessed by peer feedback)
- Convergence contribution rate

The TAI (Temporal Accumulating Intelligence) pattern detects drift: if a spore's
recent performance deviates significantly from its established baseline, it flags
an anomaly. This is not rule-based -- the system learns what "normal" looks like
from its own history.

## Reasoning Flow

### Task Submission

1. Client POSTs task to any spore's `/api/task` endpoint
2. Receiving spore generates a task ID, stores task in memory
3. Gossip carries the task to all spores within 1-2 cycles
4. Each spore begins independent reasoning

### Progressive Phases

| Phase | Cycles | Behavior |
|-------|--------|----------|
| DIVERGE | 0-4 | Maximum exploration, broad hypothesis generation |
| DEEPEN | 5-9 | Evidence gathering, claim refinement, challenge |
| CONVERGE | 10+ | Agreement-seeking, synthesis, trust-weighted merge |

Phase transitions are per-task and tracked individually.

### Reasoning Cycle

Each cycle per task:
1. **Context retrieval**: Semantic search over memory for relevant prior reasoning
2. **Peer input**: Incorporate recent deltas from other spores (trust-weighted)
3. **LLM reasoning**: Generate structured response with claims + confidence
4. **Claim extraction**: Parse claims from LLM output
5. **Delta creation**: Package as CRDT delta with metadata
6. **Memory storage**: Add to OR-Set, update Merkle tree
7. **Convergence check**: Compute semantic similarity across all spore claims

### Convergence Detection

Convergence is measured by semantic similarity between extracted claims across spores:

1. Embed all current claims using sentence-transformers
2. Compute pairwise cosine similarity
3. Calculate agreement score (fraction of pairs above threshold)
4. Track agreement over 3 consecutive cycles
5. If stable above 0.70 for 3 cycles, trigger forced synthesis

The synthesizer spore (001) produces the final synthesis, weighted by trust scores.
The validator spore (003) reviews and approves or rejects.

### Forced Synthesis

When convergence is detected:
1. Synthesizer collects all claims from all spores with trust weights
2. Generates a unified synthesis incorporating the strongest contributions
3. Validator reviews using brain-tier reasoning (GLM-4.7-Flash if available)
4. If approved, synthesis is promoted to knowledge store
5. All spores receive the converged result via gossip

## Spore Configuration

Each spore is configured through environment variables:

| Variable | Purpose |
|----------|---------|
| `SYNAPSE_SPORE_ID` | Unique identifier |
| `SYNAPSE_SPORE_INDEX` | Numeric index (determines role) |
| `SYNAPSE_PEERS` | JSON array of peer URLs |
| `SYNAPSE_PRIMARY_MODEL` | HF model ID for primary reasoning |
| `HF_TOKEN` | HuggingFace API token |
| `ZAI_API_KEY` | Z.ai API key (spores 003, 005 only) |

## LLM Provider Chain

Each spore has a primary model and a fallback chain:

1. **Primary**: Assigned per-spore model via HF Inference API or Z.ai
2. **Fallback 1**: HF Router (alternative model from same family)
3. **Fallback 2**: Any available free-tier provider
4. **Brain tier**: GLM-4.7-Flash via Z.ai (spores 003, 005 only)

The brain tier is used for:
- Deep validation of convergence quality (Validator)
- Architectural oversight and meta-reasoning (Brain)
- Synthesis review before knowledge promotion

## Data Flow Diagram

```
Task submitted
     |
     v
[Spore receives] --gossip--> [All spores receive]
     |                              |
     v                              v
[DIVERGE phase]              [DIVERGE phase]
     |                              |
     v                              v
[Claims extracted]           [Claims extracted]
     |                              |
     v                              v
[Stored in OR-Set]           [Stored in OR-Set]
     |                              |
     +---------- gossip sync -------+
     |                              |
     v                              v
[DEEPEN phase]               [DEEPEN phase]
(incorporates peer claims)   (incorporates peer claims)
     |                              |
     v                              v
[Refined claims]             [Refined claims]
     |                              |
     +---------- gossip sync -------+
     |
     v
[Convergence detected]
     |
     v
[Synthesizer produces unified answer]
     |
     v
[Validator reviews (brain tier if available)]
     |
     v
[Promoted to knowledge store]
     |
     v
[Available for all future reasoning]
```

## Scaling

The architecture scales horizontally:
- Add more spores by increasing `--count` in the launcher
- Each spore is stateless on disk (all state is in CRDT memory)
- Gossip mesh self-organizes as new peers join
- Trust is established organically through interaction

Practical limits:
- Gossip overhead grows O(n) per cycle (each spore syncs with one random peer)
- Full mesh convergence time: O(n log n) gossip cycles
- Memory sync payload grows with delta count (paginated at 100 deltas per sync)
- Free-tier LLM rate limits constrain per-spore reasoning throughput

## File Structure

```
synapse-brain/
  spore.py              -- Canonical spore (deployed to each HF Space)
  launch_swarm.py       -- Swarm deployment script
  requirements.txt      -- Spore pip dependencies
  command-center/
    app.py              -- Monitoring dashboard (deployed as HF Space)
    requirements.txt
  docs/
    ARCHITECTURE.md     -- This document
    BRAIN-PROTOCOL.md   -- Full cognitive protocol specification
    SELF-EVOLUTION.md   -- Self-improvement and temporal learning
    CONVERGENCE.md      -- Convergence detection and synthesis
  synapse_brain/        -- Python package (local CLI, future)
  tests/
    test_spore.py       -- Unit and integration tests
```
