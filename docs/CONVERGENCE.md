# Convergence Protocol

## How the Swarm Reaches Agreement

Convergence in Synapse Brain is semantic, not mechanical. The system does not vote.
It does not average. It measures whether the claims produced by independent spores
are saying the same thing in different words -- and when they are, it synthesizes.

## Semantic Similarity Engine

Claims are extracted from each spore's reasoning output as structured text. These
claims are embedded using sentence-transformers (all-MiniLM-L6-v2 by default) into
a shared vector space.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

claims_a = ["Recursive trust is the key innovation"]
claims_b = ["The trust propagation mechanism is the primary contribution"]

embeddings = model.encode(claims_a + claims_b)
similarity = cosine_similarity(embeddings[0], embeddings[1])
# similarity ~ 0.82 -- semantically aligned despite different words
```

Keyword Jaccard overlap would score these two claims near zero (different vocabulary).
Semantic similarity captures the actual meaning alignment.

## Agreement Score Computation

For each active task, at each reasoning cycle:

1. Collect all claims from all spores for the current phase
2. Embed all claims
3. For each pair of spores, compute max cosine similarity between their claims
4. Agreement score = fraction of spore pairs with max similarity above threshold (0.65)

```
agreement = |{(i,j) : max_sim(claims_i, claims_j) > 0.65}| / |total_pairs|
```

## Convergence Trigger

Convergence is declared when:

1. Agreement score exceeds 0.70
2. Score has been stable (within 0.05) for 3 consecutive cycles
3. Current phase is CONVERGE (cycle 10+)

All three conditions must hold. This prevents premature convergence during DIVERGE
(where low agreement is expected and healthy) and requires stability (not just a
spike).

## Forced Synthesis

When convergence triggers:

1. **Claim collection**: All claims from all spores, weighted by trust score
2. **Cluster analysis**: Claims are clustered by semantic similarity
3. **Synthesis prompt**: The Synthesizer (spore 001) receives:
   - All claim clusters with trust weights
   - Points of agreement (high-similarity clusters)
   - Points of disagreement (unresolved tensions)
   - Directive to produce a unified position
4. **LLM synthesis**: Synthesizer generates a structured synthesis
5. **Validation**: Validator (spore 003) reviews the synthesis
   - Standard review via Gemma 3 27B
   - Brain-tier review via GLM-4.7-Flash for high-stakes tasks
6. **Promotion**: Approved syntheses enter the knowledge store

## Phase Transitions

| Transition | Trigger | Effect |
|-----------|---------|--------|
| DIVERGE to DEEPEN | Cycle 5 reached | Shift from exploration to evidence gathering |
| DEEPEN to CONVERGE | Cycle 10 reached | Shift from deepening to agreement-seeking |
| CONVERGE to DONE | Convergence detected | Forced synthesis triggered |
| Any to EXTENDED | Agreement dropping | Phase extended if convergence regresses |

Phase transitions are per-task. Multiple tasks can be in different phases
simultaneously.

## Convergence Failure Modes

### Premature Convergence

All spores agree too quickly, usually because they share similar biases.

**Detection**: Agreement above 0.70 during DIVERGE phase.
**Response**: Extend DIVERGE phase. Inject adversarial prompts.

### Oscillation

Agreement score fluctuates around the threshold without stabilizing.

**Detection**: Score crosses 0.70 more than 3 times without holding for 3 cycles.
**Response**: Reduce threshold to 0.60 for this task. Accept weaker consensus.

### Permanent Disagreement

Spores cannot reach agreement even after extended CONVERGE phase.

**Detection**: 20+ cycles in CONVERGE without triggering synthesis.
**Response**: Force synthesis anyway with explicit disagreement documentation.
The synthesis notes which claims are contested and by whom.

### Single-Spore Domination

One spore's claims dominate the similarity scores because of high volume.

**Detection**: One spore contributes more than 40% of total claims.
**Response**: Normalize claim counts per spore before computing agreement.

## Trust Weighting in Synthesis

During forced synthesis, each claim carries a weight:

```
claim_weight = trust_score(originating_spore) * confidence(claim)
```

High-trust spores with high-confidence claims dominate the synthesis.
Low-trust spores with low-confidence claims are noted but deprioritized.

This is not a filter -- every claim reaches the synthesis prompt. But the
Synthesizer is instructed to weight contributions proportionally.

## Cross-Task Convergence

Knowledge from converged tasks informs reasoning on new tasks:

1. When a new task arrives, the sidecar indexer retrieves semantically similar
   knowledge from the knowledge store
2. This prior knowledge is injected into the reasoning context
3. Spores can build on prior convergence rather than starting from scratch

Over time, the knowledge store creates a growing foundation of converged
reasoning that accelerates convergence on new tasks.
