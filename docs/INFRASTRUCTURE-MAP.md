# Infrastructure Map

## Current Deployment (v5)

### Spore Spaces (Private)

| Space | Role | Model | Z.ai Access |
|-------|------|-------|-------------|
| `Optitransfer/synapse-spore-000` | Explorer | Qwen3 235B MoE | No |
| `Optitransfer/synapse-spore-001` | Synthesizer | Llama 3.3 70B | No |
| `Optitransfer/synapse-spore-002` | Adversarial | DeepSeek R1 32B | No |
| `Optitransfer/synapse-spore-003` | Validator | Gemma 3 27B | Yes (brain tier) |
| `Optitransfer/synapse-spore-004` | Generalist | Llama 4 Scout 17B | No |
| `Optitransfer/synapse-spore-005` | Brain | GLM-4.7-Flash | Yes (primary) |

### Monitoring

| Space | Purpose |
|-------|---------|
| `Optitransfer/synapse-command-center` | Real-time dashboard + conversation stream |

### External Dependencies

| Service | Purpose | Tier |
|---------|---------|------|
| HuggingFace Inference API | LLM reasoning (5 models) | Free / Pro |
| Z.ai API | GLM-4.7-Flash brain tier | Free |
| crdt-merge 0.9.5 (PyPI) | CRDT memory + E4 trust | Published |

### Secrets Distribution

| Secret | Spores |
|--------|--------|
| `HF_TOKEN` | All (000-005) |
| `ZAI_API_KEY` | 003, 005 only |

### Network Topology

```
Spore 000 <---> Spore 001 <---> Spore 002
  ^   \           ^   \           ^   \
  |    \          |    \          |    \
  v     v         v     v         v     v
Spore 005 <---> Spore 004 <---> Spore 003
                                 (Z.ai)

All connections are bidirectional HTTP via HF Spaces URLs.
Authentication: Bearer token (HF_TOKEN) on every request.
Gossip interval: 20 seconds.
Full mesh convergence: 2-3 gossip cycles (40-60s).
```

### Resource Consumption

| Resource | Per Spore | Total (6 spores) |
|----------|-----------|-------------------|
| CPU | 2 vCPU | 12 vCPU |
| Memory | 16 GB | 96 GB |
| Disk | Ephemeral | Ephemeral |
| LLM calls | ~3/min | ~18/min |
| Gossip bandwidth | ~10 KB/cycle | ~60 KB/cycle |

### Scaling Path

Phase 1 (current): 6 spores on HF Spaces Pro
Phase 2: 10-20 spores + Google Cloud Run gossip relay
Phase 3: 50-100 spores + Redis-backed gossip + persistent storage
Phase 4: Cloudflare Workers edge routing + Firestore shared state
