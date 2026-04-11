# Synapse Brain

Distributed reasoning swarm where multiple LLM agents converge on solutions through
CRDT-backed persistent memory, gossip-mesh communication, and trust-weighted synthesis.

```python
# Deploy a 6-spore swarm with one command
python launch_swarm.py --count 6 --hf-token $HF_TOKEN --zai-key $ZAI_KEY
```

Six spores, six different LLM families, one converging intelligence that never forgets.

## What It Does

Synapse Brain decomposes reasoning tasks across a mesh of autonomous agents (spores),
each running a different large language model. Spores reason independently, share
discoveries via gossip protocol, build trust organically through interaction quality,
and converge on synthesized answers that no single model could produce alone.

Every piece of reasoning is persisted in a CRDT-backed memory store. Nothing is ever
deleted. The system indexes continuously through a sidecar process so retrieval is
instant without context window pressure. A spore that has been running for six months
remembers everything from day one.

## Architecture

```
                    Command Center (monitoring)
                           |
        +--------+---------+---------+---------+--------+
        |        |         |         |         |        |
    Spore-0  Spore-1   Spore-2  Spore-3   Spore-4  Spore-5
    Explorer Synth.    Advers.  Validator  General. Brain
    Qwen3    Llama3.3  DeepSeek Gemma3     Llama4   GLM-4.7
    235B     70B       R1 32B   27B+Z.ai   Scout    Flash
        |        |         |         |         |        |
        +--------+---------+---------+---------+--------+
                    Gossip Mesh (HTTP + HF_TOKEN)
                           |
                    CRDT Memory Layer
                    (crdt-merge 0.9.5)
                    OR-Set + MerkleDAG
                    + E4 Trust Lattice
```

### Swarm Topology

| Spore | Cognitive Role | Primary Model | Family | Special |
|-------|---------------|---------------|--------|---------|
| 000 | Explorer | Qwen3 235B MoE | Qwen (Alibaba) | Thinking model, chain-of-thought |
| 001 | Synthesizer | Llama 3.3 70B | Meta | Best instruct model |
| 002 | Adversarial | DeepSeek R1 Distill 32B | DeepSeek | Explicit reasoning tokens |
| 003 | Validator | Gemma 3 27B + GLM-4.7 brain | Google + Z.ai | Dual-tier quality gate |
| 004 | Generalist | Llama 4 Scout 17B MoE | Meta Llama 4 | Newest MoE architecture |
| 005 | Brain | GLM-4.7-Flash | ZhipuAI | Free tier, architectural oversight |

### Why Model Diversity Matters

Identical models produce correlated errors. When five copies of the same LLM reason
about a problem, they make the same mistakes in the same places. Different model
families have different training data, different architectures, different biases.
Qwen thinks via chain-of-thought before answering. DeepSeek uses explicit reasoning
tokens. Gemma comes from completely different training data. Five genuinely independent
perspectives produce solutions none could reach alone.

## Core Systems

### 1. CRDT Persistent Memory (Never-Forgetting)

Every reasoning delta, claim, synthesis, and trust event is stored in a
`crdt-merge` OR-Set. Add-wins semantics means nothing is ever lost. Gossip
protocol means every spore eventually has everything. The MerkleDAG provides
cryptographic verification of memory integrity.

```python
from crdt_merge.core import ORSet
from crdt_merge.merkle import MerkleTree

memory = ORSet()
merkle = MerkleTree()

# Store a reasoning delta -- it can never be lost
tag = memory.add({"task_id": "abc", "claim": "X > Y", "confidence": 0.85})
merkle.insert(tag, hash(delta))

# Gossip carries the delta to all peers automatically
# Every spore converges to the same memory state
```

The sidecar indexer continuously builds embedding indices over the memory store,
enabling instant semantic retrieval without stuffing the context window. A spore
retrieves relevant prior reasoning by meaning, not by recency.

### 2. Master Cognitive Protocol (Full-Spectrum Reasoning)

Every spore is loaded with the complete Master Cognitive Protocol -- 913 lines,
18 parts, covering Five-Phase Discipline, 11 Operational Mandates, and Three-Tier
Synapse architecture. The cognitive role (Explorer, Synthesizer, Adversarial,
Validator, Generalist, Brain) acts as a LENS on the full protocol, not a limitation.

An Explorer applies the full protocol but emphasizes divergent search. A Validator
applies the full protocol but emphasizes rigorous falsification. Every spore can
reason about anything -- their role shapes their initial approach, not their ceiling.

### 3. Semantic Convergence Engine

Convergence is measured by semantic similarity between extracted claims, not keyword
overlap. The engine uses sentence-transformers to embed claims and compute cosine
similarity. When agreement exceeds the convergence threshold (default 0.70) and
remains stable for 3 consecutive cycles, forced synthesis triggers.

Progressive phases govern the reasoning arc:
- **DIVERGE** (cycles 0-4): Maximum exploration, broad hypothesis generation
- **DEEPEN** (cycles 5-9): Evidence gathering, claim refinement, adversarial challenge
- **CONVERGE** (cycle 10+): Agreement-seeking synthesis, trust-weighted integration

### 4. E4 Trust Lattice (Byzantine Detection)

Trust between spores is managed by the `crdt-merge` E4 DeltaTrustLattice. Each
interaction produces a trust delta. Consistently high-quality contributions increase
trust. Low-quality or adversarial contributions decrease it. Trust scores weight
how much influence each spore has during synthesis.

The Validator spore (003) has dual-tier access: Gemma 3 27B for standard validation,
GLM-4.7-Flash via Z.ai for deep validation of convergence quality. This gives the
quality gate access to the most powerful reasoning available.

### 5. Temporal Self-Learning (TAI Pattern)

Each spore learns its own operational heartbeat -- cycle times, error rates,
reasoning quality distribution, convergence speed. Deviations from the established
temporal pattern trigger anomaly detection. A spore that suddenly produces lower
quality reasoning or experiences unusual latency is flagged.

This is the TAI (Temporal Accumulating Intelligence) pattern applied to cognitive
health: the system knows what "normal" looks like because it has been observing
itself continuously. No rules, no thresholds -- just temporal provenance.

### 6. Trust-Weighted Gossip

When gossip carries reasoning deltas between spores, the trust score of the
originating spore weights how the receiving spore processes the information.
High-trust deltas are incorporated directly. Low-trust deltas are treated as
hypotheses requiring additional validation. This prevents a single unreliable
spore from corrupting the collective.

## Self-Evolution Protocol

The swarm continuously improves itself:

1. **Self-analysis tasks**: The swarm reasons about its own architecture and
   proposes improvements
2. **Convergence quality tracking**: Each synthesis is scored and tracked over time
3. **Temporal baseline drift**: The TAI system detects gradual performance changes
4. **Trust recalibration**: Trust scores evolve based on actual contribution quality
5. **Memory-informed reasoning**: Prior reasoning shapes future approaches -- the
   system learns from its own history

The first self-analysis convergence (v3 swarm) produced 111 reasoning deltas across
5 model families and independently validated model diversity as the highest-priority
improvement before knowing it had already been implemented. The swarm co-evolved
with its own recommendations.

## Deployment

### Prerequisites

- HuggingFace account (Pro tier for private Spaces)
- Python 3.10+
- `pip install huggingface_hub`

### Quick Start

```bash
# Deploy 6-spore swarm with Z.ai brain tier on Validator + Brain
python launch_swarm.py \
  --count 6 \
  --hf-token $HF_TOKEN \
  --hf-owner Optitransfer \
  --zai-key $ZAI_KEY \
  --private

# Monitor via Command Center
# https://optitransfer-synapse-command-center.hf.space
```

### Submitting Tasks

```bash
curl -X POST https://optitransfer-synapse-spore-000.hf.space/api/task \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"task": "Analyze the implications of recursive trust propagation in distributed systems"}'
```

Tasks propagate to all spores via gossip within one cycle. Each spore reasons
independently, shares claims, and the swarm converges to a synthesized answer.

### Command Center

The Command Center is a private HuggingFace Space that aggregates state from all
spores. It provides:

- Real-time spore health, clock, cycles, peers, deltas
- Full conversation stream showing reasoning evolution across all spores
- Trust matrix visualization
- Task submission interface
- Memory statistics and integrity verification

## Infrastructure

| Component | Platform | Role |
|-----------|----------|------|
| Spore 000-005 | HF Spaces (Private) | Reasoning agents |
| Command Center | HF Spaces (Private) | Monitoring dashboard |
| Memory Layer | crdt-merge 0.9.5 (PyPI) | Persistent CRDT state |
| Brain Tier | Z.ai GLM-4.7-Flash | Deep validation (free) |
| Worker LLMs | HF Inference API | Primary reasoning (free) |

## How It Differs

Most multi-agent systems share a context window, use the same model, and forget
everything between sessions. Synapse Brain:

- **Never forgets**: CRDT memory is append-only, gossip-replicated, Merkle-verified
- **Genuinely diverse**: 6 different model families, not 6 copies of GPT-4
- **Trust-weighted**: Quality of contribution determines influence, not volume
- **Self-aware**: Temporal self-learning detects its own performance drift
- **Convergent**: Semantic similarity drives synthesis, not keyword matching
- **Decentralized**: No orchestrator bottleneck, pure peer-to-peer mesh

## Dependencies

- [crdt-merge](https://pypi.org/project/crdt-merge/) >= 0.9.5 -- CRDT memory + E4 trust
- [sentence-transformers](https://www.sbert.net/) >= 3.0 -- semantic convergence
- [httpx](https://www.python-httpx.org/) >= 0.27 -- gossip mesh
- [numpy](https://numpy.org/) >= 1.24 -- vector operations

## Patents

Patent: UK Application No. GB 2607132.4, GB2608127.3

## License

BSL 1.1 -- Change Date: 2028-04-08
