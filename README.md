# Synapse Brain

Experimental distributed reasoning swarm where autonomous LLM agents converge on
solutions through never-forgetting CRDT memory, gossip-mesh communication, and
trust-weighted synthesis.

**The foundation is built. We want to see how far it scales.**

```
git clone https://github.com/mgillr/synapse-brain.git
cd synapse-brain
cp config.template.yaml config.yaml   # edit with your API keys (or leave blank for free-tier)
python launch_swarm.py --config config.yaml
```

Three commands. Your swarm is live.

## What This Is

An experimental implementation of swarm intelligence with a core property:
**the swarm never forgets.** Every piece of reasoning every agent produces is
persisted in a CRDT-backed memory store using add-wins semantics. Nothing is
ever deleted. A spore that has been running for months remembers everything
from its first cycle.

The intelligent sliding window surfaces only the most relevant context per
query through continuous semantic indexing -- the context window stays fixed
regardless of total memory size.

We want to find the limits. How large can a swarm grow before memory becomes
a bottleneck? How many concurrent agents can share context effectively? What
happens to convergence quality at 100 nodes? 1,000? 10,000?

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

### Core Properties

- **Never-forgetting memory**: OR-Set with add-wins semantics. Memories only accumulate, never shrink. Gossip propagates everything to all peers.
- **Semantic sliding window**: Continuous background indexing via sentence-transformers. Retrieval is O(1) similarity search -- context window pressure is zero regardless of memory size.
- **Trust-weighted synthesis**: E4 recursive trust lattice (from [crdt-merge](https://github.com/mgillr/crdt-merge)) scores every peer based on contribution quality. Higher-trust peers have more influence on converged answers.
- **Privacy boundary**: Knowledge Wall ensures raw input never enters gossip. Only distilled insights cross the boundary. HMAC-bound provenance for auditability.
- **Federation-ready**: Any node running crdt-merge >= 0.9.5 can join. Swarm DNA integrity verification prevents modified nodes from participating.

### What We Are Testing

1. **Memory scaling** -- How many memories can a single spore hold before retrieval degrades? Current: 6,500+ per spore, 45,000+ aggregate across 7 nodes with no degradation.
2. **Convergence quality** -- Does answer quality improve with more diverse models? Current: 7 model families, consistent convergence within 3-5 cycles.
3. **Trust dynamics** -- How does the E4 trust lattice behave over time? Does trust distribution stabilize? Current: 53 converged tasks, trust range 0.40-0.70.
4. **Swarm scaling** -- What happens at 20 nodes? 100? The architecture is designed for it but untested at scale.
5. **Cross-commander federation** -- Multiple independent operators running their own swarms, connected via gossip. Untested in the wild.

## Quick Start

### Option 1: Deploy on HuggingFace Spaces (free tier)

```bash
git clone https://github.com/mgillr/synapse-brain.git
cd synapse-brain
cp config.template.yaml config.yaml
# Edit config.yaml -- at minimum set your HF token
python launch_swarm.py --config config.yaml --count 3
```

This deploys 3 spores as free HF Spaces. No GPU needed -- inference runs via
free-tier LLM APIs.

### Option 2: Run locally

```bash
pip install -r requirements.txt
python spore.py  # single spore, local mode
```

### Option 3: Join the existing swarm

Edit `config.yaml` to point your spores at the existing Optitransfer core
swarm as peers. Your nodes will automatically discover and gossip with the
network.

## Configuration

Copy `config.template.yaml` and edit:

```yaml
# Minimum viable config -- just your HF token
hf_token: "hf_your_token_here"

# Optional: add API keys for more model diversity
# Leave blank to use free-tier providers only
api_keys:
  zai: ""          # Z.ai (GLM-4.7-Flash, free)
  groq: ""         # Groq (Llama, free tier)
  google_ai: ""    # Google AI Studio (Gemini, free tier)
  cerebras: ""     # Cerebras (fast inference, free tier)

# Swarm size (default: 3)
count: 3

# Your identity as a commander
commander: "your-username"
```

On first launch, the swarm will:
1. Validate your config
2. Deploy spores to your HF account
3. Connect to the gossip mesh
4. Start reasoning

If no API keys are provided, spores fall back to free-tier providers
(HF Router, Z.ai free models). You can add keys later -- spores pick
them up on restart.

## Project Structure

```
synapse-brain/
  spore.py              # Core spore runtime (2,500+ lines)
  cortex.py             # Local micro-LLM for Sentinel (Qwen3-4B)
  knowledge_wall.py     # Privacy boundary (distillation + HMAC)
  mcp_server.py         # MCP tool server (7 tools per spore)
  federation.py         # Federation join/gossip protocol
  launch_swarm.py       # Deployment script
  config.template.yaml  # Configuration template
  requirements.txt      # Dependencies
  command-center/       # Monitoring dashboard (Gradio)
  docs/                 # Architecture documentation
  tests/                # Test suite
```

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for details. The short version:

- **Run your own swarm** and report what you find
- **Add new LLM providers** -- the provider interface is simple to extend
- **Improve the semantic indexer** -- the sliding window is the bottleneck we expect to hit first
- **Stress test memory scaling** -- we need data on degradation curves
- **Build evaluation harnesses** -- measure convergence quality rigorously
- **Try new reasoning protocols** -- the system prompt on each spore defines its reasoning strategy

## Current Status

This is experimental software. The foundation works:

- 7-spore swarm has been running continuously with 53+ converged tasks
- 6,500+ memories per spore with no retrieval degradation
- Trust lattice stabilizing around 0.40-0.70 range
- Knowledge Wall filtering correctly (crossings vs blocks tracked)
- MCP server exposing 7 tools per spore for external integration
- Sentinel with local Cortex (Qwen3-4B) for autonomous code deployment

What we do not know yet:

- Memory scaling ceiling (theoretical: millions; tested: ~7K)
- Optimal spore count for convergence quality
- Trust dynamics at scale (>20 peers)
- Federation behavior across independent commanders
- Real-world failure modes under adversarial conditions

## Dependencies

- [crdt-merge >= 0.9.5](https://pypi.org/project/crdt-merge/) -- CRDT primitives, E4 trust lattice, Merkle provenance
- sentence-transformers -- semantic embedding for memory retrieval
- FastAPI + uvicorn -- spore HTTP server
- httpx -- gossip mesh communication
- numpy -- numerical operations

## License

MIT License. See [LICENSE](LICENSE) for details.

Uses [crdt-merge](https://pypi.org/project/crdt-merge/) as a dependency
(separately licensed).
