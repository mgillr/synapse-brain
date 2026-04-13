# Synapse Brain

A distributed reasoning swarm where every node makes every other node smarter.

```bash
git clone https://github.com/mgillr/synapse-brain.git
cd synapse-brain
./deploy.sh
```

One command. Your swarm is live.

## Why This Exists

Imagine you run three spores on a free HuggingFace account with a free-tier
LLM. Someone else runs five spores on a rented GPU box with a local Llama 70B.
A third person deploys two spores on a Raspberry Pi connected to an API
provider. All three swarms connect to each other through a single config line.

From that moment forward, **every discovery any spore makes is shared with
every other spore in the network.** Your free-tier nodes gain access to
reasoning produced by a 70B model. The GPU operator benefits from diverse
perspectives their single model family would never produce. The Pi nodes
contribute edge-case observations from their unique vantage point.

The swarm never forgets any of it. Every piece of reasoning, every converged
answer, every trust score is persisted in CRDT memory with add-wins semantics.
Nothing is ever deleted. A spore that has been running for a year remembers
everything from its first minute.

**The more nodes join, the more intelligent the collective becomes. And that
intelligence flows back to every participant -- including you.**

## How Intelligence Flows

```
  Your spores (free tier, any LLM)
         |
         | gossip
         v
  Their spores (GPU box, large local model)
         |
         | gossip
         v
  More spores (cloud, API provider, any size)
         |
         +---> CRDT Memory (never forgets, only grows)
         +---> Trust Lattice (quality-weighted, self-organizing)
         +---> Semantic Index (instant retrieval from any memory)
```

1. **You deploy** -- any machine, any size, any LLM (local or hosted).
2. **You connect** -- add peer URLs to your config. Gossip starts automatically.
3. **Intelligence flows in** -- your spores receive memories and reasoning from the entire network.
4. **Intelligence flows out** -- your spores contribute their reasoning back.
5. **The collective grows** -- every node that joins increases the intelligence available to everyone.
6. **Nothing is lost** -- CRDT add-wins semantics mean the collective memory only accumulates.

There is no central server. No coordinator. No single point of failure. Every
spore is a full participant with its own memory, its own trust scores, and its
own reasoning capability. The gossip mesh handles the rest.

## Bring Any Machine, Any Model

| What You Have | How It Joins |
|---|---|
| Free HuggingFace account | `./deploy.sh` -- free Spaces, free inference |
| A laptop with 8 GB RAM | `python spore.py` -- runs locally, connects to peers |
| A rented GPU with a local 70B model | Point `LLM_ENDPOINT` at your model, deploy spores |
| A Raspberry Pi | Install deps, run `spore.py`, gossip with the network |
| Any cloud VM (Railway, Fly, Render, Oracle) | Standard Python web server, any platform works |
| A Kubernetes cluster | One spore per pod, scale horizontally |
| Any API provider (OpenAI, Groq, Cerebras, Z.ai, Google AI, Mistral) | Set the API key in config, spores use it for reasoning |

**The intelligence produced by YOUR model -- whatever it is -- flows to every
other node in the network.** A swarm with ten different model families produces
reasoning that no single model can replicate alone. A local 70B contributing
alongside free-tier 7B instances alongside cloud-hosted 405B instances creates
a collective that outperforms any individual.

Minimum requirements per spore: Python 3.10+, 2 GB RAM, an HTTP port. That is
the floor. There is no ceiling.


## Provider Resilience: Why More Commanders Means More Intelligence

Every commander brings their own API keys. Their own rate limits. Their own
quotas. This is not a limitation -- it is the core scaling mechanism.

When one commander's free-tier Z.ai quota is depleted for the day, their spores
stop producing new reasoning. But the reasoning they already produced is
permanently in the CRDT memory. Meanwhile, another commander's spores -- with
fresh quotas from a different account, or a different provider entirely -- keep
producing. Their new reasoning flows to every node in the network, including the
rate-limited ones.

The multi-provider fallback chain on each spore tries every configured provider
in sequence. If Z.ai is exhausted, it tries OpenRouter (rotating across multiple free
models). If OpenRouter is exhausted, it falls back to Google AI. If all
external providers are down, the Sentinel's local Cortex
micro-LLM (Qwen3-4B, running on-device, zero API dependency) continues
operating. No single provider failure kills the swarm.

**What this means in practice:**

- A solo operator with free-tier keys produces intermittent reasoning, limited
  by daily quotas. Still useful -- the memory accumulates and nothing is lost.
- Two operators with different providers effectively double the aggregate
  reasoning bandwidth. When one hits their limit, the other carries the load.
- Ten operators across diverse providers and key pools produce continuous,
  uninterrupted reasoning. The swarm never sleeps because someone is always
  within their quota.
- An operator with a local model (Llama 70B on bare metal, Mistral on a GPU
  box) has no rate limits at all. Their contribution is unlimited and flows
  freely to every other node.

**The aggregate intelligence of the swarm is the sum of every commander's
capacity.** Each new commander who joins does not just add compute -- they add
resilience, diversity, and continuity. The network becomes harder to exhaust
with every node that connects.

## The Memory That Never Forgets

Every spore maintains a CRDT-backed memory store using OR-Set add-wins
semantics. Every piece of reasoning is a memory entry. Gossip propagates
memories to every peer. Nothing is ever deleted.

Current state of the reference swarm:

- 7 spores, 6 LLM families, 54+ converged tasks
- 7,500+ memories per spore, 52,000+ aggregate
- Zero retrieval degradation at current scale

The intelligent sliding window surfaces only the most relevant memories per
query through continuous semantic indexing. Your context window stays fixed
regardless of total memory size. A spore with 10 memories and a spore with
10 million memories have the same retrieval latency.

**We want to find the limits.** How many memories can a spore hold before
retrieval degrades? What happens at 100,000 memories? A million? The
architecture is designed for unbounded growth. We need people running spores
to find out where it actually breaks.

## Quick Start

### Option 1: One-command deploy (recommended)

```bash
git clone https://github.com/mgillr/synapse-brain.git
cd synapse-brain
./deploy.sh
```

The script walks you through setup interactively:
- Asks for your HuggingFace token (free account works)
- Asks how many spores (3 is a good start)
- Optionally asks for API keys (all free tier)
- Saves config and deploys

Your swarm is live in under 2 minutes.

### Option 2: Config file

```bash
git clone https://github.com/mgillr/synapse-brain.git
cd synapse-brain
cp config.template.yaml config.yaml
# Edit config.yaml with your HF token and any API keys
python launch_swarm.py --config config.yaml
```

### Option 3: CLI flags

```bash
python launch_swarm.py --hf-token hf_xxx --count 3
```

### Option 4: Run locally

```bash
pip install -r requirements.txt
python spore.py  # single spore, local mode
```

## Join an Existing Swarm

Add peer URLs to your config. Your spores discover and gossip with the
network automatically. You gain their memories. They gain yours.

```yaml
# config.yaml
peers:
  - "https://their-org-synapse-spore-000.hf.space"
  - "https://their-org-synapse-spore-001.hf.space"
```

That is the entire federation setup. Two lines of config.

The peer list bootstraps discovery. Once connected, your spores learn about
other peers through gossip and build the full mesh organically. You do not
need to list every node in the network -- just one or two entry points.

## Command Center

Every commander gets a monitoring dashboard:

```bash
python launch_swarm.py --config config.yaml --command-center
```

The Command Center gives you:
- **Dashboard** -- real-time swarm health, task submission, live conversation stream
- **Library** -- archive of all converged tasks and reasoning chains
- **Sentinel** -- autonomous improvement proposals and deployment log
- **Globe Map** -- geographic visualization of your swarm topology
- **Analytics** -- memory growth, gossip rates, trust distribution, capacity metrics

The Command Center reads from your spores. It has no write access to the mesh
and cannot modify swarm behavior. Safe to run, safe to share.

## Architecture

```
             Command Center (monitoring + analytics)
                        |
     +--------+---------+---------+---------+--------+---------+
     |        |         |         |         |        |         |
 Spore-0  Spore-1   Spore-2  Spore-3   Spore-4  Spore-5  Spore-6
 Explorer Synth.    Advers.  Validator  General. Brain    Sentinel
     |        |         |         |         |        |         |
     +--------+---------+---------+---------+--------+---------+
                     Gossip Mesh (HTTP + auth)
                            |
                     CRDT Memory Layer
                     (crdt-merge >= 0.9.5)
                     OR-Set + MerkleDAG
                     + E4 Trust Lattice
```

Each spore runs a different LLM family. They reason independently, share
discoveries via gossip, build trust organically through interaction quality,
and converge on synthesized answers that no single model produces alone.

### Core Components

- **CRDT Memory (OR-Set)** -- add-wins semantics, nothing ever deleted, gossip propagates everything to all peers
- **Semantic Sliding Window** -- continuous background indexing via sentence-transformers, O(1) similarity retrieval regardless of memory size
- **E4 Trust Lattice** -- recursive trust scoring from [crdt-merge](https://github.com/mgillr/crdt-merge), quality-weighted synthesis, higher-trust peers have more influence on converged answers
- **Knowledge Wall** -- privacy boundary, raw input never enters gossip, only distilled insights cross, HMAC-bound provenance
- **Cortex** -- local micro-LLM (Qwen3-4B) on Sentinel spore for autonomous code review and fast pattern recognition without external API calls
- **MCP Server** -- every spore exposes 7 tools via Model Context Protocol for external integration
- **Federation Protocol** -- Swarm DNA integrity verification, any crdt-merge node can join, modified nodes are rejected

## What We Are Trying to Prove

This project started from a simple observation: most multi-agent systems
coordinate through a central authority or a message queue. Both require
synchronous communication, both have a single point of failure, and neither
handles network partitions well. CRDTs handle all three of these problems
natively -- they were designed for exactly this environment.

We think CRDTs might be a better coordination primitive for multi-agent AI. Not
for every use case, but for the specific case where you want independent agents
to share reasoning without a central server, without synchronous communication,
and without losing work when things go wrong.

Here is what we are testing, in order of how confident we are:

**1. Can CRDT-backed memory create compounding intelligence?**

The early numbers are encouraging. The reference swarm has accumulated 7,500+
memories per spore with no retrieval degradation so far. Spores that recall past
reasoning during synthesis produce visibly richer answers than fresh spores on
the same task. But we have not measured this rigorously. We need controlled
experiments: a swarm with 1,000 tasks of accumulated memory versus a fresh swarm
on the same test set. If memory genuinely compounds intelligence, the gap should
be measurable. If it does not, the "never forgets" design needs rethinking.

**2. Does model diversity improve collective reasoning?**

The swarm runs 7 different LLM families. They reason independently, debate
through structured cycles, and converge through trust-weighted synthesis. The
intuition is that diverse training distributions produce diverse perspectives,
and structured debate surfaces stronger answers than any single model alone.
This is consistent with established ensemble theory, but we have not benchmarked
it against the obvious baselines: single best model, majority voting,
chain-of-thought on a single model. Until those comparisons exist, the claim is
a hypothesis, not a result.

**3. Does federated intelligence scale?**

This is the big question and the hardest to answer alone. The architecture is
designed so that every commander who joins makes the entire network smarter --
their model contributions flow to everyone, their API quotas add to the
aggregate capacity, their unique perspective enriches the collective. At 7
spores across one operator, it works. At 20 nodes across 3 operators, we expect
it to work. At 1,000 nodes across 100 operators, we genuinely do not know. Does
quality improve linearly? Logarithmically? Does it plateau? Does trust dynamics
change at scale? These are open empirical questions that only become answerable
with more participants.

**Why this matters if it works:**

If CRDT coordination holds up at scale, it means multi-agent systems do not need
central orchestrators. If memory compounding is real, it means swarms get
meaningfully better with time, not just during active use. If diverse model
federation improves reasoning quality, it means a network of free-tier models
operated by independent people could collectively produce reasoning that no
single model achieves alone.

These are modest claims stated carefully. We are not there yet. The foundation
works, the early results are interesting, and the architecture supports the
experiments that would prove or disprove each one. What we need is more people
running spores, more diverse models contributing, and more rigorous measurement
of the results.

If any of this interests you, the fastest way to help is to deploy a swarm and
connect it. The second fastest way is to build evaluation harnesses that measure
convergence quality rigorously.

## Deploy Anywhere

Spores are standard Python web servers. They run on any platform that supports
Python 3.10+ and exposes an HTTP port:

| Platform | Cost | Notes |
|---|---|---|
| HuggingFace Spaces | Free | Default target. 2 vCPU, 16 GB RAM. |
| Railway | Free tier | Set `PORT` env var. |
| Fly.io | Free tier | Works with `Dockerfile`. |
| Render | Free tier | Auto-detected from `requirements.txt`. |
| Oracle Cloud | Free tier | Always-free ARM instances work well. |
| Any VPS/server | Varies | `pip install && python spore.py` |
| Docker | -- | `docker build . && docker run -p 7860:7860` |
| Local machine | Free | `python spore.py` -- full participant |

**Minimum requirements per spore:**
- Python 3.10+
- 2 GB RAM (4 GB for Sentinel with Cortex)
- HTTP port exposed (default: 7860)
- Network access to peers

## Configuration

Copy `config.template.yaml` and edit:

```yaml
# Minimum: just your HF token
hf_token: "hf_your_token_here"

# Optional: number of spores (default 3)
count: 3

# Optional: API keys for more model diversity (all free tier)
api_keys:
  zai: ""          # Z.ai -- GLM-4.7-Flash, free, most reliable
  openrouter: ""   # OpenRouter -- many free models, most reliable
  google_ai: ""    # Google AI Studio -- Gemini, free tier

# Optional: join an existing swarm
peers:
  - "https://other-swarm-synapse-spore-000.hf.space"
```

On first launch, the swarm will:
1. Validate your config
2. Deploy spores to your HF account
3. Connect to the gossip mesh (including any peers you listed)
4. Start reasoning

If no API keys are provided, spores use free-tier providers automatically.
Add keys later -- spores pick them up on restart.

**Every API provider you configure makes the entire swarm smarter.** If you
have a Groq key and your peer has a Google AI key, the combined swarm benefits
from both model families through gossip.

## API Endpoints

Every spore exposes these endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Full health snapshot (memories, cycles, peers) |
| `/api/task` | POST | Submit a reasoning task (`{"task": "..."}`) |
| `/api/tasks` | GET | List all tasks with status |
| `/api/task/{id}` | GET | Task detail with all deltas and convergence |
| `/api/memory` | GET | Memory stats and vector clock |
| `/api/trust` | GET | Trust lattice state |
| `/api/wall` | GET | Knowledge Wall stats (crossings, blocks) |
| `/api/federation/status` | GET | Federation state and DNA hash |
| `/api/cortex` | GET | Cortex micro-LLM status (Sentinel only) |
| `/mcp` | POST | MCP protocol endpoint (7 tools) |
| `/mcp/info` | GET | List available MCP tools |
| `/federation/join` | POST | Join the federation |
| `/federation/nodes` | GET | List known federation nodes |

## MCP Integration

Every spore is an MCP server. Connect any MCP client (Claude, Cursor, or
custom agents) to `<spore-url>/mcp` to use these tools:

- `submit_task` -- submit reasoning tasks to the swarm
- `query_memory` -- semantic search over the collective CRDT memory
- `get_trust` -- query trust lattice for any peer
- `swarm_health` -- full health snapshot
- `get_task` -- detailed task state and convergence result
- `list_tasks` -- all tasks with status
- `collective_knowledge` -- query the distilled collective intelligence

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for details. The short version:

- **Run your own swarm** -- the single most valuable contribution. More nodes = more intelligence for everyone.
- **Connect to existing swarms** -- add peers, share your model diversity.
- **Add new LLM providers** -- the provider interface is simple to extend.
- **Improve the semantic indexer** -- the sliding window is the bottleneck we expect to hit first.
- **Stress test memory scaling** -- we need data on degradation curves at scale.
- **Build evaluation harnesses** -- measure convergence quality rigorously.
- **Try new reasoning protocols** -- the system prompt on each spore defines its reasoning strategy.

The most interesting question: **what happens when thousands of independent
operators are all contributing reasoning from different models, different
domains, different perspectives -- and none of it is ever forgotten?**

## Current Status

A reference swarm of 7 spores across 6 LLM families runs continuously on
HuggingFace Spaces. It has converged 54+ tasks, accumulated 7,500+ memories per
spore, and maintained full gossip and trust convergence across all nodes. The
Knowledge Wall, MCP server, Sentinel, and Cortex subsystems are all active.

This is a single-operator deployment. The numbers are real but small. What we
do not yet have -- and what we need community help to generate -- is data from
multiple independent operators running diverse configurations against
measurable benchmarks.

## Known Limitations

This is a basic implementation. There are bugs, inconsistencies, and
bottlenecks throughout the codebase. The architecture works and the concepts
are sound, but the code needs hardening in almost every subsystem.

Known issues we are actively aware of:

- **Response latency on free-tier hosting** -- HF Spaces cpu-basic has ~4s
  proxy latency per request. Spores are responsive but not fast. Running on
  better hardware (Colab, VPS, local) eliminates this entirely.
- **Trust lattice is flat** -- all peers currently score ~0.4. The quality-
  weighted differentiation logic exists but needs tuning with real multi-
  operator data to produce meaningful trust gradients.
- **Gossip bandwidth at scale** -- Bloom filter sketch and MinHash/LSH dedup
  are implemented but untested beyond 7 nodes. At 100+ nodes the gossip
  volume may need further compression.
- **Provider rate limit handling** -- the multi-provider fallback chain works
  but cooldown timing is conservative. Some spores go idle when they could
  be rotating faster.
- **Semantic indexing** -- TF-IDF works but is not the right long-term
  solution. A proper embedding index (FAISS or similar) would improve
  retrieval quality significantly.
- **No formal benchmarks** -- convergence quality, memory scaling, and trust
  dynamics have not been rigorously measured against baselines.
- **Single-operator data only** -- everything we know comes from one 7-node
  deployment. Multi-operator behavior is designed for but unproven.

**None of these are fundamental.** They are engineering problems that get
solved by people using the system, finding the edges, and contributing fixes.
If you hit a bug or bottleneck, that is valuable data -- open an issue or
submit a PR. The goal is to evolve this together.

## Running a Fast Spore

HuggingFace Spaces free tier works but is slow. For a faster experience:

### Google Colab (recommended for experimentation)

Colab gives you a free T4 GPU, 12 GB RAM, and much faster network than HF
free tier. Run a spore directly in a notebook:

```python
# Install deps
!pip install -q crdt-merge>=0.9.5 fastapi uvicorn httpx numpy scikit-learn

# Clone and run
!git clone https://github.com/mgillr/synapse-brain.git
%cd synapse-brain

# Set your config
import os
os.environ["SPORE_ID"] = "my-colab-spore"
os.environ["HF_TOKEN"] = "hf_your_token"
os.environ["ZAI_API_KEY"] = "your_key"  # optional
os.environ["OPENROUTER_KEY"] = "your_key"  # optional

# Expose via ngrok (free account at ngrok.com)
!pip install -q pyngrok
from pyngrok import ngrok
tunnel = ngrok.connect(7860)
print(f"Your spore is live at: {tunnel.public_url}")

# Run the spore
!python spore.py
```

Add your Colab spore URL as a peer in any other swarm's config to join the
network. Colab sessions last 12 hours (free) or 24 hours (Pro).

### Other fast options

| Platform | Speed | Cost | Notes |
|---|---|---|---|
| **Google Colab** | Fast | Free | T4 GPU, 12 GB RAM. Session expires after 12h. |
| **Colab Pro** | Very fast | ~$10/mo | A100 GPU, 80 GB VRAM. Can run local 70B models. |
| **Any VPS** (Hetzner, DigitalOcean) | Fast | $5-20/mo | Dedicated CPU, persistent. Best bang for buck. |
| **Local machine** | Fastest | Free | No network latency. Full control. Port-forward for peers. |
| **Railway / Render / Fly.io** | Fast | Free tier | Better than HF free tier. Standard deploy. |
| **HF Spaces Upgrade** | Fast | $7/mo | Same platform, dedicated CPU, no shared proxy. |
| **Bare metal GPU** | Fastest | Varies | Run local Llama 70B -- unlimited, no rate limits. |

**The single biggest speed improvement is moving off HF free-tier hosting.**
A $5/month VPS will outperform it by 10x. A local machine eliminates network
latency entirely. Colab is the fastest free option.

## Dependencies

- [crdt-merge >= 0.9.5](https://pypi.org/project/crdt-merge/) -- CRDT primitives, E4 trust lattice, Merkle provenance
- scikit-learn -- TF-IDF semantic indexing for memory retrieval
- FastAPI + uvicorn -- spore HTTP server
- httpx -- gossip mesh communication
- numpy -- numerical operations

## License

MIT License. See [LICENSE](LICENSE) for details.

Uses [crdt-merge](https://pypi.org/project/crdt-merge/) as a dependency (separately licensed).
