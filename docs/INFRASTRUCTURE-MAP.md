# Synapse Brain -- Free Infrastructure Map

Deployment targets for mass agent swarm. Every platform below is free-tier,
no credit card, permanent (unless noted). Combined capacity: 1000+ concurrent
agents at zero cost.

---

## Tier 1: Always-On Compute (Persistent Agents)

These platforms host long-running Python processes. Each runs a Spore agent
that maintains CRDT state, gossips with peers, and processes reasoning tasks.

### Oracle Cloud Always Free

- **Capacity**: 4 ARM Ampere A1 instances (total 4 OCPU, 24 GB RAM) + 2 AMD
  micro instances (1/8 OCPU, 1 GB RAM each)
- **Storage**: 200 GB block volume (total)
- **Network**: 10 TB/month outbound
- **Persistence**: Full VMs, survive reboots, SSH access
- **Best for**: Coordinator nodes, persistent state stores, mesh relays
- **Agents per account**: 6 VMs = run 20-50 lightweight spores per VM = **~120-300 agents**
- **Notes**: Most generous always-free tier. ARM is perfect for Python. Can
  run systemd services. One account per person but multiple email signups
  have worked historically.

### HuggingFace Spaces (CPU)

- **Capacity**: Unlimited free CPU Spaces (2 vCPU, 16 GB RAM each)
- **Sleep**: After 48h of inactivity (keep-alive ping prevents this)
- **Storage**: Ephemeral (use external persistence)
- **Best for**: Worker spores, inference endpoints, public-facing nodes
- **Agents per account**: Soft limit ~50 Spaces realistically
- **Deploy**: `git push` to HF repo, auto-builds from Dockerfile or app.py
- **Notes**: ZeroGPU available for inference bursts. Gradio/FastAPI native.

### Render

- **Capacity**: 750 free instance hours/month (enough for 1 always-on service)
- **Sleep**: After 15min idle (keep-alive ping solves this)
- **Storage**: Ephemeral filesystem, free Postgres (1 GB, expires 30 days)
- **Best for**: HTTP relay nodes, webhook receivers, lightweight spores
- **Agents per account**: 1 always-on (or several with sleep rotation)
- **Deploy**: GitHub auto-deploy
- **Caution**: Suspends services that initiate too much outbound traffic

### Koyeb

- **Capacity**: 1 free service, 512 MB RAM, 0.1 vCPU
- **Sleep**: Scale-to-zero when idle
- **Storage**: Ephemeral + free Postgres
- **Best for**: Coordinator/relay, lightweight API endpoint
- **Agents per account**: 1
- **Deploy**: GitHub or Docker

### Vercel Serverless Functions

- **Capacity**: 100K function invocations/day, 10s execution limit
- **Best for**: Gossip relay, task routing, webhook endpoints
- **Agents per account**: N/A (stateless functions)
- **Deploy**: `vercel deploy`
- **Notes**: Not for long-running agents. Perfect for message routing layer.

### Cloudflare Workers

- **Capacity**: 100K requests/day, 10ms CPU time per request
- **Storage**: KV (1 GB free), Durable Objects, R2 (10 GB free)
- **Best for**: Gossip relay, state sync routing, edge presence
- **Agents per account**: Unlimited workers
- **Deploy**: `wrangler deploy`
- **Notes**: 10ms CPU limit means no heavy reasoning. Ideal for message
  routing, peer discovery, lightweight consensus checks.

---

## Tier 2: Notebook Compute (Batch/GPU Agents)

These platforms run heavier workloads -- model inference, tensor operations,
deep reasoning chains. Agents here wake up, process a batch, push results
to the mesh, and go idle.

### Google Colab Free

- **Capacity**: T4 GPU (16 GB VRAM), ~12h session limit
- **Best for**: Inference-heavy spores, model evaluation, tensor merge
- **Agents per session**: 1 (but can run many sequential tasks)
- **Notes**: Can connect to mesh via HTTP. Session reconnect needed.

### Kaggle Notebooks

- **Capacity**: 30h GPU/week (T4 or P100), 20h TPU/week
- **Storage**: 20 GB local, persistent datasets
- **Best for**: Scheduled heavy compute, evaluation runs
- **Agents per account**: Multiple notebooks concurrent

### Lightning.ai

- **Capacity**: Free tier with CPU Studios
- **Best for**: Persistent dev environments, model serving
- **Notes**: More limited free tier than Colab/Kaggle

---

## Tier 3: Free LLM APIs (Reasoning Providers)

Each spore can use these as its reasoning backend. By spreading across
providers, the swarm avoids any single rate limit. Combined free capacity
below is massive.

| Provider | Free Limits | Best Models | Notes |
|---|---|---|---|
| Google AI Studio | 1.5M tokens/min, 1500 req/day | Gemini 2.0 Flash, Gemini 3 Flash | Most generous. Primary reasoning engine. |
| Groq | 1000 req/day, 6K tokens/min | DeepSeek R1 70B, Llama 3.3 70B | Ultra-fast inference. |
| OpenRouter | 20 req/min, 200 req/day | 29+ free models | Multi-model routing. Good fallback. |
| Cerebras | 30 req/min, 60K tokens/min | Llama 3.3 70B | Fast, high throughput. |
| Mistral | 1 req/sec, 500K tokens/min | Mistral Large, Mistral 8B | Strong reasoning. |
| HuggingFace Inference | Variable monthly credits | Any model <10GB | Run your own fine-tunes. |
| Cloudflare Workers AI | 10K neurons/day | Llama 3.1 8B, Mistral 7B | Edge inference. |
| GitHub Models | Free with GH account | GPT-4o, various | Good for prototyping. |
| Together AI | Free tier | Llama 3.2, DeepSeek R1 70B | Collaborative. |
| NVIDIA NIM | Free prototyping | DeepSeek, Llama, Gemma | GPU-optimized inference. |
| Cohere | 1000 calls/month | Command R+, embeddings | Good for RAG tasks. |
| Scaleway | 100 req/min | Llama 3.1 70B | EU-based. |
| OVH AI | 12 req/min | CodeLlama, Llama 70B | EU-based. |
| AI21 | $10 credits for 3 months | Jamba Large/Mini | Trial credits. |
| Fireworks AI | $1 free credits | Llama 405B, DeepSeek R1 | Fast serverless. |

**Combined daily capacity (conservative):**
- Google: ~1500 complex reasoning calls
- Groq: ~1000 calls
- OpenRouter: ~200 calls
- Cerebras: ~43K calls (30/min * 1440 min)
- Mistral: ~86K calls (1/sec * 86400 sec)
- Others: ~2000 calls combined

**Total: ~130K+ free reasoning calls per day across all providers**

With 1000 agents, each agent gets ~130 reasoning calls/day. For gossip and
state sync (which do not require LLM calls), throughput is effectively
unlimited.

---

## Tier 4: Persistence Layer

Agent state must survive restarts. Options:

| Service | Free Tier | Best For |
|---|---|---|
| Cloudflare KV | 1 GB, 100K reads/day | Agent registry, peer lists |
| Cloudflare R2 | 10 GB, 1M reads/month | CRDT state snapshots |
| Supabase | 500 MB Postgres, 1 GB storage | Structured agent state |
| PlanetScale | 1 billion row reads/month | High-read workloads |
| Neon Postgres | 0.5 GB, always free | Lightweight state |
| Upstash Redis | 10K commands/day | Fast KV, pub/sub |
| GitHub repos | Unlimited private repos | State as git commits |

---

## Deployment Strategy

### Phase 1: Foundation (Week 1)
- 6 Oracle Cloud VMs as persistent mesh backbone
- 10 HuggingFace Spaces as worker spores
- 1 Cloudflare Worker as gossip relay
- 1 Supabase instance as state store
- Total: ~70 agents

### Phase 2: Scale (Week 2-3)
- 30 more HF Spaces
- Kaggle notebooks for heavy compute
- Add Groq + Google AI Studio as reasoning providers
- Render for HTTP relay
- Total: ~300 agents

### Phase 3: Mass (Week 4+)
- Multiple Oracle Cloud accounts (family/friends)
- 50+ HF Spaces
- Full LLM provider rotation
- Cloudflare edge presence worldwide
- Total: ~1000+ agents

---

## Cost Analysis

| Component | Monthly Cost |
|---|---|
| Compute (Oracle + HF + Render + Koyeb) | $0 |
| Relay (Cloudflare Workers + Vercel) | $0 |
| Reasoning (all LLM APIs combined) | $0 |
| Persistence (Supabase + Cloudflare KV/R2) | $0 |
| **Total** | **$0** |

---

## Platform-Specific Agent Configurations

### Oracle Cloud Spore
```yaml
runtime: python3.12
memory: 4096  # 4GB per VM (6 VMs available)
spores_per_vm: 20-50
process: systemd service
persistence: local SQLite + mesh sync
```

### HuggingFace Space Spore
```yaml
runtime: docker or gradio
memory: 16384  # 16GB per Space
spores_per_space: 1 (dedicated)
process: FastAPI server
persistence: mesh sync only (ephemeral disk)
keep_alive: ping every 30min
```

### Cloudflare Worker Relay
```yaml
runtime: javascript (or python via pyodide)
cpu_limit: 10ms per request
role: gossip_relay, peer_discovery
persistence: KV store
```

### Render Spore
```yaml
runtime: python3.12
memory: 512  # free tier limit
spores_per_instance: 1
process: FastAPI server
persistence: mesh sync (ephemeral disk)
keep_alive: ping every 10min
```
