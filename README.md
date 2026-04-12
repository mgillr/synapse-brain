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

## What We Are Testing

1. **Memory scaling** -- how many memories before retrieval degrades? Current: 7,500+ per spore with zero degradation.
2. **Convergence quality** -- does answer quality improve with more diverse models? Current: 7 model families, consistent convergence.
3. **Trust dynamics** -- how does the E4 trust lattice behave over time? Does trust distribution stabilize?
4. **Swarm scaling** -- what happens at 20 nodes? 100? 1,000? The architecture is designed for it.
5. **Cross-commander federation** -- multiple independent operators, connected via gossip. The viral intelligence loop.
6. **Model diversity** -- what mix of models produces the highest quality convergence?

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
  groq: ""         # Groq -- Llama 3.3 70B, free tier
  google_ai: ""    # Google AI Studio -- Gemini, free tier
  cerebras: ""     # Cerebras -- fast inference, free tier
  openrouter: ""   # OpenRouter -- many models, free tier

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

The foundation works. The reference swarm runs continuously:

- 7 spores across 6 LLM families, 54+ converged tasks
- 7,500+ memories per spore with no retrieval degradation
- Knowledge Wall filtering active on every spore
- MCP server exposing 7 tools per spore for external integration
- Sentinel with local Cortex (Qwen3-4B) for autonomous operations
- Full gossip mesh with trust-weighted synthesis

What we want to learn:

- Memory scaling ceiling (tested: ~7,500 per spore; theoretical: millions)
- Optimal spore count and model mix for convergence quality
- Trust dynamics at scale (current: 12 peers tracked)
- Cross-commander federation behavior with independent operators
- Real-world failure modes and recovery patterns

## Dependencies

- [crdt-merge >= 0.9.5](https://pypi.org/project/crdt-merge/) -- CRDT primitives, E4 trust lattice, Merkle provenance
- sentence-transformers -- semantic embedding for memory retrieval
- FastAPI + uvicorn -- spore HTTP server
- httpx -- gossip mesh communication
- numpy -- numerical operations

## License

MIT License. See [LICENSE](LICENSE) for details.

Uses [crdt-merge](https://pypi.org/project/crdt-merge/) as a dependency (separately licensed).
