# Synapse Brain Architecture

## Overview

Synapse Brain is a distributed reasoning system built on three layers:

1. **Spore Layer** -- autonomous LLM agents that reason independently
2. **Memory Layer** -- CRDT-backed persistent storage with semantic indexing
3. **Trust Layer** -- E4 recursive trust lattice for peer quality scoring

## Memory Model

### Never-Forgetting Property

Every spore maintains an OR-Set (from crdt-merge) with add-wins semantics.
When a spore produces a reasoning artifact, it is added to the local OR-Set
and propagated to all peers via gossip.

OR-Set add-wins means: if one spore adds a memory and another concurrently
does anything else, the add wins. Memories only accumulate. The total
memory count across the swarm grows monotonically.

### Semantic Sliding Window

A background sidecar continuously indexes all memories using
sentence-transformers. When a spore needs context for a new reasoning
task, it retrieves the top-K most semantically relevant memories.

This is the key to scaling: the context window size is fixed regardless
of total memory count. A spore with 100 memories and a spore with
1,000,000 memories both use the same context window. Retrieval time
is O(log N) with HNSW indexing.

### Memory Lifecycle

```
New reasoning artifact
    |
    v
Add to local OR-Set (instant, local)
    |
    v
Gossip to peers (async, background)
    |
    v
Peers add to their OR-Sets (add-wins merge)
    |
    v
Sidecar indexes new memory (background)
    |
    v
Available for semantic retrieval (all spores)
```

## Trust Model

### E4 Recursive Trust Lattice

Each spore maintains a DeltaTrustLattice (from crdt-merge) that tracks
peer trust scores. Trust is earned through consistent, high-quality
contributions over time.

Trust flows into synthesis: when multiple spores contribute to a
converged answer, higher-trust contributions carry more weight.

### Trust Update

After each reasoning cycle:
1. All spores that participated get a trust delta
2. The delta magnitude depends on how useful their contribution was
3. Trust propagates through the lattice via gossip
4. Over time, consistently good spores accumulate high trust

### Symbiotic Lattice Trust (SLT)

SLT is a detection-and-exclusion mechanism, not classical BFT. It does
not tolerate Byzantine faults -- it detects them through trust score
divergence and excludes the faulty peer from synthesis weighting.

A peer whose trust drops below the lattice minimum is effectively
excluded from influencing converged answers while still being able
to participate and rebuild trust.

## Privacy Model

### Knowledge Wall

The Knowledge Wall separates each spore's memory into two domains:

- **Private memory** -- raw input from the commander, local reasoning
  artifacts. Never leaves the spore. Never enters gossip.
- **Collective memory** -- distilled insights extracted from private
  reasoning. HMAC-bound provenance. This is what gets gossiped.

The distillation process is one-way: raw data produces distilled
insights, but the insights cannot be reversed to recover the raw data.
Even if a peer is fully compromised, it only has access to distilled
collective memories -- the raw input from other commanders does not
exist on any other spore.

## Communication Model

### Gossip Mesh

Spores communicate via HTTP gossip. Each spore knows a set of peers
and periodically shares:

- New memories (OR-Set deltas)
- Trust updates (DeltaTrustLattice deltas)
- Health status
- Task state

Gossip is eventually consistent -- all spores converge to the same
state given sufficient time and connectivity.

### Federation

New nodes join the swarm by:
1. Installing crdt-merge >= 0.9.5
2. Connecting to any existing peer
3. Passing Swarm DNA integrity verification
4. Beginning gossip exchange

Swarm DNA is a hash of the federation protocol. Modified nodes fail
verification and cannot join.

## Spore Roles

Each spore has a cognitive role that influences its reasoning approach:

| Role | Behavior |
|---|---|
| Explorer | Divergent thinking, generates novel hypotheses |
| Synthesizer | Convergent thinking, combines and refines |
| Adversarial | Challenges claims, finds weaknesses |
| Validator | Verifies consistency, checks evidence |
| Generalist | Balanced approach, fills gaps |
| Brain | Deep reasoning, complex analysis |
| Sentinel | Autonomous code deployment, system maintenance |

Roles are not rigid partitions -- every spore has the full Master
reasoning protocol. The role is a lens that influences which aspects
of the protocol are emphasized.

## Scaling Properties

### Known (tested)

- 7 spores: stable convergence, 53+ tasks, 45,000+ aggregate memories
- Memory retrieval: no degradation up to ~7K per spore
- Trust: stabilizes in 0.40-0.70 range within ~20 tasks
- Gossip: converges within 3-5 cycles at 7 nodes

### Unknown (needs testing)

- Memory degradation curve at 100K+ per spore
- Convergence quality at 20+ spores
- Gossip convergence time at 100+ nodes
- Trust dynamics with adversarial peers at scale
- Cross-commander federation behavior
