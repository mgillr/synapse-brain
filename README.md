# Synapse Brain

Distributed reasoning swarm. 1000 cheap agents converging on solutions with mathematical guarantees.

```python
from synapse_brain import Orchestrator

swarm = Orchestrator(spore_count=50, provider_keys={
    "GROQ_API_KEY": "gsk_...",
    "GOOGLE_AI_KEY": "AIza...",
    "GLM_API_KEY": "...",
})

task = swarm.submit_task(
    description="Design a distributed caching strategy for 10M concurrent users",
    documents=[{"name": "requirements.pdf", "content": "..."}],
)

import asyncio
result = asyncio.run(swarm.run_task(task, max_cycles=8))

print(result.final_answer)
print(f"Converged: {result.convergence.converged}")
print(f"Agreement: {result.convergence.agreement_ratio:.0%}")
```

## What This Is

Each **spore** (agent) in the swarm is a lightweight reasoning node powered by free-tier LLM APIs. Individually they are limited. Collectively they solve problems that no single model can.

**How the swarm thinks:**
1. A task enters the orchestrator
2. A Deconstructor spore breaks it into subtasks
3. Spores reason about subtasks in parallel, each with a cognitive specialization
4. Results propagate via peer-to-peer gossip using CRDT state (guaranteed convergence)
5. E4 trust lattice weights contributions -- bad reasoning gets decayed automatically
6. A Synthesizer spore assembles the converged answer

Every spore runs the **Synapse Cognitive Protocol** -- a thinking architecture that governs how it reasons. The protocol programs cognitive discipline (Five-Phase Discipline, Alien Brain Protocol, 10 innovation protocols) into the system prompt of every LLM call.

## Cognitive Roles

The swarm assigns cognitive specializations to maximize reasoning diversity:

| Role | Function | Distribution |
|------|----------|-------------|
| Explorer | Hunts anomalies, edge cases, alien frameworks | 15% |
| Deconstructor | Breaks arguments to atomic primitives | 10% |
| Synthesizer | Recombines primitives into novel structures | 20% |
| Refiner | Scores probability, runs murder board | 10% |
| Validator | Designs and runs experiments | 15% |
| Generalist | Balanced across all phases | 15% |
| Adversarial | Tries to kill weak hypotheses | 10% |
| Alien | Applies cross-domain mathematical frameworks | 5% |

## Architecture

```
                    +-------------------+
                    |   Orchestrator    |
                    |   (submit task,   |
                    |    monitor,       |
                    |    synthesize)    |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
         +----v----+   +----v----+   +----v----+
         | Spore 0 |   | Spore 1 |   | Spore N |
         | Explorer|   | Synth.  |   | Advers. |
         +----+----+   +----+----+   +----+----+
              |              |              |
              +--------------+--------------+
                             |
                    +--------v---------+
                    |  CRDT Mesh Layer  |
                    |  (gossip + merge) |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  E4 Trust Layer   |
                    |  (Byzantine det.) |
                    +------------------+
```

## Free-Tier Provider Support

Every spore can route reasoning through any combination of:

| Provider | Free Tier | Model |
|----------|-----------|-------|
| Groq | 14,400 req/day | Llama 3.1 70B |
| Google AI Studio | 1,500 req/day | Gemini 2.0 Flash |
| Cerebras | 30 req/min | Llama 3.1 70B |
| Mistral | 1M tokens/month | Mistral Small |
| OpenRouter | Variable | Multiple |
| HuggingFace | Rate-limited | Llama 3 8B |
| Together AI | $1 free credit | Llama 3.1 8B |
| NVIDIA NIM | 5,000 req/month | Llama 3.1 8B |
| Cloudflare Workers AI | 10,000 neurons/day | Llama 3.1 8B |
| Cohere | 1,000 req/month | Command R |
| GitHub Models | 150 req/day | GPT-4o-mini |
| GLM-4 | Free tier | GLM-4-Flash |

With 12 providers, a swarm of 100 spores can sustain ~20,000+ reasoning calls per day at zero cost.

## Deployment

### Local (development)

```bash
pip install synapse-brain
synapse-brain dashboard --spores 20 --port 7770
```

Opens the command center at `http://localhost:7770` with real-time swarm topology, health monitoring, task submission, and document upload.

### Mass Deployment (free hosts)

Deploy spores across free infrastructure:

```bash
# Deploy 10 spores to HuggingFace Spaces
synapse-brain deploy --target hf-spaces --count 10

# Deploy 4 spores to Oracle Cloud free tier
synapse-brain deploy --target oracle --count 4

# Deploy to Render free tier
synapse-brain deploy --target render --count 3

# Deploy a Cloudflare Worker relay
synapse-brain deploy --target cloudflare-relay
```

See [INFRASTRUCTURE-MAP.md](docs/INFRASTRUCTURE-MAP.md) for the full deployment matrix.

## Document-Aware Reasoning

Attach documents to any task. Every spore receives the documents in its context window:

```python
with open("report.pdf") as f:
    content = f.read()

task = swarm.submit_task(
    description="Analyze this report and identify the 3 biggest risks",
    documents=[{"name": "Q4-report.pdf", "content": content}],
)
```

## Dashboard

The localhost command center provides:

- **Swarm topology** -- live map of all spores, their roles, and connection health
- **Task monitor** -- convergence progress, cycle-by-cycle reasoning trace
- **Health metrics** -- per-spore latency, trust scores, provider usage
- **Task submission** -- submit tasks with document uploads
- **Reasoning trace** -- inspect individual deltas and the synthesis chain

```bash
synapse-brain dashboard
```

## How Convergence Works

1. Each spore produces **reasoning deltas** -- structured outputs containing hypothesis, evidence, confidence, contradictions, and primitives
2. Deltas are inserted into a **CRDT OR-Set** with Merkle provenance tracking
3. Spores gossip state to peers -- CRDT merge guarantees every spore eventually sees every delta
4. **E4 trust lattice** weights each spore's contributions -- consistently useful reasoning gets amplified, noise gets decayed
5. The orchestrator tracks **convergence**: trust-weighted agreement ratio + stability cycles
6. When agreement exceeds threshold and remains stable, the swarm has converged

No central coordinator. No single point of failure. Mathematical convergence guarantee.

## Built On

- [crdt-merge](https://github.com/mgillr/crdt-merge) -- deterministic state convergence (OR-Set + Merkle + vector clocks)
- E4 Recursive Trust-Delta Protocol -- Byzantine fault detection and trust propagation
- Patent: UK Application No. GB 2607132.4, GB2608127.3

## License

Business Source License 1.1

Licensed Work: Synapse Brain
Change Date: 2028-04-08
Change License: Apache License 2.0
