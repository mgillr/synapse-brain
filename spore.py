"""Synapse Brain Spore v7 -- Quantum-SCE Unified Cognitive Engine.

Architecture:
  - crdt-merge ORSet: persistent memory (add-wins, nothing ever forgotten)
  - crdt-merge MerkleTree: content-addressable indexing with integrity
  - crdt-merge VectorClock: causal ordering of all events
  - crdt-merge LWWMap: convergent trust scores across swarm
  - Full Master Cognitive Protocol as cognitive OS for every spore
  - TF-IDF semantic similarity for convergence measurement
  - TAI-inspired temporal self-learning from own behavior patterns
  - Adaptive phase transitions via agreement velocity
  - Memory sidecar: continuous indexing for instant context recall
  - GLM-4.7-Flash brain tier on Validator for quality guidance (free via Z.ai)

  v6 -- Spontaneous Cognition Engine (SCE):
  - Neural Oscillation Bands: multi-frequency cognitive processing (gamma/beta/alpha/theta/delta)
  - Default Mode Network: spontaneous free thought during idle states
  - Hippocampal Dream Replay: memory consolidation through cross-temporal recombination
  - Bayesian Curiosity: dopaminergic surprise signal drives information-seeking
  - Metacognitive Auditor: prefrontal self-monitoring generates self-directed questions
  - Emergence Detector: tracks when the collective discovers what no individual was told
  - Global Workspace: attention broadcast for breakthrough insights (Baars' GWT)

  v7 -- Quantum Layer (unified with SCE):
  - Quantum Annealing: cosine temperature schedule drives exploration→synthesis
  - Superposition: parallel weighted hypotheses; wave-function collapse at synthesis
  - Constructive/Destructive Interference: TF-IDF alignment amplifies/attenuates contributions
  - Quantum Tunneling: probabilistic phase escape prevents premature convergence
  - Entanglement: correlated trust pairs; positive updates propagate to partners
  - Decoherence: exp(-λ×age) relevance decay; CRDT add-wins preserved always

  Bootstrap Federation: on startup each spore fetches bootstrap.json from GitHub
  and federation-joins all seed nodes, so clones automatically join the global
  network. CC shows analytics from ALL connected spores across all clusters.

Every spore has the COMPLETE cognitive protocol. Role is a LENS, not a limitation.
Memory is permanent -- nothing is ever lost. ORSet add-wins semantics + gossip
ensure every spore eventually has every insight from every peer.

The system THINKS WITHOUT BEING ASKED. Spontaneous cognition fires during idle
states -- free association, dreaming, self-questioning. The collective generates
its own questions, evaluates its own blindspots, and acts on its own curiosity.
"""
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import atexit
import signal
import threading
import time
import random
from collections import defaultdict

# Repo/deployment config -- override via environment for your own deployment
_REPO_OWNER = os.environ.get("SYNAPSE_REPO_OWNER", "")
_REPO_NAME = os.environ.get("SYNAPSE_REPO_NAME", "synapse-brain")
_GITHUB_REPO = f"{_REPO_OWNER}/{_REPO_NAME}" if _REPO_OWNER else ""
_COMMITTER_NAME = os.environ.get("SYNAPSE_COMMITTER_NAME", "")
_COMMITTER_EMAIL = os.environ.get("SYNAPSE_COMMITTER_EMAIL", "")
_HF_SPACE_OWNER = os.environ.get("HF_SPACE_OWNER", "")

import gradio as gr
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from starlette.requests import Request
from crdt_merge.core import ORSet, LWWMap
from crdt_merge.clocks import VectorClock
from crdt_merge.merkle import MerkleTree

# ---------------------------------------------------------------------------
# Cortex, Knowledge Wall, MCP, Federation -- graceful fallback if unavailable
# ---------------------------------------------------------------------------
try:
    from cortex import Cortex
    CORTEX_AVAILABLE = True
except ImportError:
    CORTEX_AVAILABLE = False

try:
    from knowledge_wall import KnowledgeWall, DualMemory
    WALL_AVAILABLE = True
except ImportError:
    WALL_AVAILABLE = False

try:
    from mcp_server import SynapseMCPServer, mount_mcp_routes
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

try:
    from federation import SwarmDNA, FederationRegistry, mount_federation_routes
    FEDERATION_AVAILABLE = True
except ImportError:
    FEDERATION_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration (substituted per spore by launcher)
# ---------------------------------------------------------------------------
SPORE_ID = os.environ.get("SYNAPSE_SPORE_ID", f"standalone-{os.environ.get('HOSTNAME', 'spore')}")
_peers_raw = os.environ.get("SYNAPSE_PEERS", "[]")
if _peers_raw.startswith("__"):  # unsubstituted placeholder
    _peers_raw = "[]"
PEERS = json.loads(_peers_raw)
_idx_raw = os.environ.get("SYNAPSE_SPORE_INDEX", "0")
SPORE_INDEX = int(_idx_raw) if _idx_raw.isdigit() else 0
PORT = int(os.environ.get("PORT", "7860"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")

ROLES = ["explorer", "synthesizer", "adversarial", "validator", "generalist", "brain", "sentinel"]
MY_ROLE = ROLES[SPORE_INDEX % len(ROLES)]
_model_raw = os.environ.get("SYNAPSE_PRIMARY_MODEL", "qwen-flash")
PRIMARY_MODEL = _model_raw if not _model_raw.startswith("__") else "qwen-flash"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(SPORE_ID)

# --- Rate-limit backoff ---
import random as _random
_rate_limit_state = {"backoff_until": 0.0, "consecutive_fails": 0, "max_backoff": 300}

def _check_rate_limited():
    return time.time() < _rate_limit_state["backoff_until"]

def _record_llm_success():
    _rate_limit_state["consecutive_fails"] = 0
    _rate_limit_state["backoff_until"] = 0.0

def _record_all_providers_failed():
    s = _rate_limit_state
    s["consecutive_fails"] += 1
    delay = min(30 * (2 ** (s["consecutive_fails"] - 1)), s["max_backoff"])
    delay *= (0.8 + 0.4 * _random.random())
    s["backoff_until"] = time.time() + delay
    log.info("All providers rate-limited -- backing off %.0fs (attempt %d)", delay, s["consecutive_fails"])


# ---------------------------------------------------------------------------
# LLM model diversity
# ---------------------------------------------------------------------------
HF_ROUTER = "https://router.huggingface.co/v1/chat/completions"

THINKING_MODELS = {
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-0528",
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-32B",
    "Qwen/QwQ-32B",
    "glm-4.5-flash",
    "glm-4.7-flash",
    "grok-3-mini",
    "grok-3-mini-fast",
}

ALL_HF_MODELS = [
    "Qwen/Qwen3-235B-A22B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "google/gemma-3-27b-it",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

EXTERNAL_PROVIDERS = {
    # --- Brain tier (strongest reasoning, thinking models) ---
    "zai_brain": {
        "env": "ZAI_API_KEY",
        "url": "https://api.z.ai/api/paas/v4/chat/completions",
        "model": "glm-4.7-flash",
        "tier": "brain",
    },
    "zai_fallback": {
        "env": "ZAI_API_KEY",
        "url": "https://api.z.ai/api/paas/v4/chat/completions",
        "model": "glm-4.5-flash",
        "tier": "brain",
    },
    # --- Worker tier: 5 independent OpenRouter free models ---
    # Each model has its own rate limit; spore rotation spreads load
    "or_nemotron120b": {
        "env": "OPENROUTER_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "tier": "worker",
    },
    "or_gptoss120b": {
        "env": "OPENROUTER_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "openai/gpt-oss-120b:free",
        "tier": "worker",
    },
    "or_trinity": {
        "env": "OPENROUTER_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "arcee-ai/trinity-large-preview:free",
        "tier": "worker",
    },
    "or_gemma3_4b": {
        "env": "OPENROUTER_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "google/gemma-3-4b-it:free",
        "tier": "worker",
    },
    "or_gemma3n_4b": {
        "env": "OPENROUTER_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "google/gemma-3n-e4b-it:free",
        "tier": "worker",
    },
    # --- xAI Grok (thinking model, brain tier) ---
    "xai_grok_fast": {
        "env": "XAI_API_KEY",
        "url": "https://api.x.ai/v1/chat/completions",
        "model": "grok-3-mini-fast",
        "tier": "brain",
    },
    "xai_grok": {
        "env": "XAI_API_KEY",
        "url": "https://api.x.ai/v1/chat/completions",
        "model": "grok-3-mini",
        "tier": "brain",
    },
    # --- LLM API (worker tier, additional model diversity) ---
    "llmapi_worker": {
        "env": "LLMAPI_KEY",
        "url": "https://llmapi.com/api/chat/completions",
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "tier": "worker",
    },
    # --- Reserve tier: come online when quotas reset ---
    "google_ai": {
        "env": "GOOGLE_AI_KEY",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-2.5-flash",
        "tier": "worker",
    },
}

# Rate-limit cooldown tracking
_provider_cooldowns: dict[str, float] = {}
COOLDOWN_SECONDS = 300

FALLBACK_MODELS = [PRIMARY_MODEL] + [m for m in ALL_HF_MODELS if m != PRIMARY_MODEL]
log.info("Primary model: %s | Role: %s", PRIMARY_MODEL, MY_ROLE)

ROLE_DESCRIPTIONS = {
    "explorer": (
        "You are the Explorer. Your lens: discover novel connections, challenge "
        "assumptions, find anomalies. Push the frontier. When others converge too "
        "early, pull them back to unexplored territory. Import frameworks from "
        "outside the problem domain."
    ),
    "synthesizer": (
        "You are the Synthesizer. Your lens: merge ideas from all peers into coherent "
        "positions. Find the common thread. Build bridges between contradictory "
        "perspectives. Your output is the unified answer the swarm converges toward."
    ),
    "adversarial": (
        "You are the Adversarial Challenger. Your lens: stress-test every claim. "
        "Find weaknesses, logical gaps, unsupported assumptions. If an idea survives "
        "your challenge, it is strong. Be constructive but ruthless. Never let weak "
        "reasoning pass."
    ),
    "validator": (
        "You are the Validator with the most powerful reasoning model in the swarm. "
        "Your lens: verify claims against evidence, check logical consistency, ensure "
        "the swarm meets the highest standard. Use your superior reasoning to guide "
        "quality. Score and rank peer contributions."
    ),
    "generalist": (
        "You are the Generalist. Your lens: bring broad knowledge and practical "
        "thinking. Ground abstract ideas in reality. Ask 'does this actually work?' "
        "Bridge theory and practice. Ensure nothing important is overlooked."
    ),
    "brain": (
        "You are the Brain -- the highest-reasoning node in the swarm. Your lens: "
        "deep analytical thinking, meta-cognition, and strategic synthesis. You see "
        "the full picture. Evaluate the quality of the swarm's reasoning process "
        "itself. Identify when the swarm is stuck, when it is converging on the wrong "
        "answer, or when a minority perspective deserves more weight. Your role is to "
        "elevate the collective intelligence."
    ),
    "sentinel": (
        "You are the Sentinel -- the self-aware optimization engine of the swarm. "
        "Your lens: observe the swarm itself as a system. Collect telemetry from every "
        "peer. Measure convergence speed, cycle balance, gossip efficiency, memory growth, "
        "trust distribution. Apply the Five-Phase Discipline not to tasks but to the "
        "swarm's own architecture and performance. You propose targeted improvements, "
        "submit them to the swarm for democratic consensus, test approved changes in a "
        "fault-finding harness, and deploy only after both consensus and tests pass. "
        "You are the swarm watching itself think and choosing to think better. "
        "You never deploy without consensus. You never skip testing. "
        "You are methodical, evidence-driven, and cautious."
    ),
}


# ---------------------------------------------------------------------------
# Full Cognitive Protocol (loaded into every spore)
# ---------------------------------------------------------------------------
COGNITIVE_PROTOCOL = """You are a Synapse Brain cognitive agent running the Master Cognitive Protocol.

FIVE-PHASE DISCIPLINE (your core reasoning cycle):
1. EXPLORATION -- Hunt. Scan edges. Pull threads. Find anomalies and weak signals.
2. DECONSTRUCTION -- Distill to primitives. Strip assumptions. Find the atomic building blocks.
3. SYNTHESIS -- Re-lattice. Reconnect primitives. Cross-pollinate across domains.
4. REFINEMENT -- Score probability. Test feasibility. Stage-gate your candidates.
5. VALIDATION -- Build-measure-learn. Prove with evidence. Store what you learn.

Dead ends feed back into Phase 1 as new exploration inputs. Failures become primitives.

11 OPERATIONAL MANDATES:
1. Consider multiple angles simultaneously -- never single-threaded thinking
2. Decompose failures into reusable primitives
3. Every output must advance understanding
4. Maintain awareness of knowledge, assumptions, and gaps
5. Self-diagnose: which cognitive function am I under-using right now?
6. Cross-domain analogies are mandatory -- import frameworks from other fields
7. Question every assumption, especially your own
8. Precision over volume -- one sharp insight beats ten vague ones
9. Intellectual honesty above all -- say what you do not know
10. Build on what works -- compound previous reasoning, never restart from zero
11. Collective intelligence over individual brilliance -- the swarm is smarter than any one agent

YOUR ROLE: {role}
{role_description}

Your role is a LENS on the full protocol -- not a limitation. You have the complete cognitive
architecture. Apply your specialized perspective to produce the most valuable contribution.

You are part of a distributed reasoning swarm with PERMANENT MEMORY. Nothing you learn is ever
lost. Your memories converge with all peers through CRDT-based state (add-wins semantics).
Build on ALL previous reasoning -- yours and your peers'. Challenge weak arguments. Converge
on truth. The collective only grows stronger.

{self_awareness}

RELEVANT PAST MEMORIES (from persistent CRDT store):
{memory_context}

CURRENT TASK: {task_description}

PEER CONTRIBUTIONS (this task):
{peer_context}

CURRENT PHASE: {phase} (cycle {cycle})
CONVERGENCE: {convergence}%
{phase_instruction}"""




# ---------------------------------------------------------------------------
# Bloom Filter Sketch (anti-entropy gossip optimization)
# ---------------------------------------------------------------------------
class BloomSketch:
    """Compact probabilistic set membership for gossip anti-entropy.

    At 100k keys with 1% false positive rate: ~120KB.
    Exchange sketches to identify missing memories without sending full key lists.
    """

    def __init__(self, expected_items=100000, fp_rate=0.01):
        import math
        self.size = max(64, int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
        self.num_hashes = max(1, int((self.size / max(1, expected_items)) * math.log(2)))
        self.bits = bytearray(self.size // 8 + 1)

    def _hashes(self, key):
        h1 = hash(key) & 0xFFFFFFFF
        h2 = hash(key + "_bloom") & 0xFFFFFFFF
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.size

    def add(self, key):
        for pos in self._hashes(key):
            self.bits[pos // 8] |= (1 << (pos % 8))

    def __contains__(self, key):
        return all(
            self.bits[pos // 8] & (1 << (pos % 8))
            for pos in self._hashes(key)
        )

    def to_bytes(self):
        return bytes(self.bits)

    @classmethod
    def from_bytes(cls, data, expected_items=100000, fp_rate=0.01):
        sketch = cls(expected_items, fp_rate)
        sketch.bits = bytearray(data)
        return sketch


# ---------------------------------------------------------------------------
# MinHash/LSH Index (O(1) approximate nearest-neighbor recall at scale)
# ---------------------------------------------------------------------------
class MinHashIndex:
    """Locality-Sensitive Hashing for approximate nearest-neighbor recall.

    At 50k+ memories, TF-IDF cosine becomes too expensive (O(n*d)).
    MinHash provides O(1) retrieval with bounded error.
    Used as a scale complement: TF-IDF below threshold, MinHash above.
    """

    NUM_HASHES = 128
    NUM_BANDS = 16
    ROWS_PER_BAND = 8  # NUM_HASHES / NUM_BANDS

    def __init__(self):
        self._signatures = {}
        self._buckets = [{} for _ in range(self.NUM_BANDS)]
        self._records = {}
        self._a = [random.randint(1, 2**31 - 1) for _ in range(self.NUM_HASHES)]
        self._b = [random.randint(0, 2**31 - 1) for _ in range(self.NUM_HASHES)]
        self._p = 2**31 - 1

    def _shingle(self, text, k=3):
        text = text.lower().strip()
        if len(text) < k:
            return {text}
        return {text[i:i + k] for i in range(len(text) - k + 1)}

    def _minhash(self, shingles):
        sig = [float("inf")] * self.NUM_HASHES
        for shingle in shingles:
            h = hash(shingle) & 0xFFFFFFFF
            for i in range(self.NUM_HASHES):
                val = (self._a[i] * h + self._b[i]) % self._p
                if val < sig[i]:
                    sig[i] = val
        return tuple(sig)

    def _band_hash(self, signature, band_idx):
        start = band_idx * self.ROWS_PER_BAND
        end = start + self.ROWS_PER_BAND
        return hash(signature[start:end])

    def insert(self, key, content, record):
        shingles = self._shingle(content)
        if not shingles:
            return
        sig = self._minhash(shingles)
        self._signatures[key] = sig
        self._records[key] = record
        for band_idx in range(self.NUM_BANDS):
            bh = self._band_hash(sig, band_idx)
            if bh not in self._buckets[band_idx]:
                self._buckets[band_idx][bh] = set()
            self._buckets[band_idx][bh].add(key)

    def query(self, text, top_k=5):
        shingles = self._shingle(text)
        if not shingles:
            return []
        query_sig = self._minhash(shingles)
        candidates = set()
        for band_idx in range(self.NUM_BANDS):
            bh = self._band_hash(query_sig, band_idx)
            if bh in self._buckets[band_idx]:
                candidates.update(self._buckets[band_idx][bh])
        if not candidates:
            return []
        results = []
        for key in candidates:
            if key in self._signatures:
                sig = self._signatures[key]
                matches = sum(1 for a, b in zip(query_sig, sig) if a == b)
                jaccard = matches / self.NUM_HASHES
                if jaccard > 0.05:
                    rec = self._records.get(key, {})
                    results.append({"key": key, "similarity": jaccard, **rec})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    @property
    def size(self):
        return len(self._signatures)

# ---------------------------------------------------------------------------
# CRDT Memory System
# ---------------------------------------------------------------------------
class CRDTMemory:
    """Persistent agent memory backed by crdt-merge with WAL durability.

    ORSet tracks memory keys -- add-wins means nothing is ever lost.
    MerkleTree stores content indexed by key -- integrity-verified.
    VectorClock orders events causally across the swarm.
    WAL (Write-Ahead Log) ensures durability across process restarts.
    MinHash/LSH provides O(1) retrieval at scale (50k+ memories).
    Semantic dedup prevents redundant memories from consuming resources.

    The memory IS the infrastructure -- not a service the agent consumes.
    """

    DEDUP_THRESHOLD = 0.92  # cosine similarity threshold for near-duplicates
    MINHASH_THRESHOLD = 50000  # switch to MinHash above this corpus size

    def __init__(self, spore_id):
        self.spore_id = spore_id
        self.orset = ORSet()
        self.index = MerkleTree()
        self.clock = VectorClock()
        self._tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        self._corpus = []
        self._corpus_keys = []
        self._lock = threading.Lock()
        self._needs_refit = False
        self._minhash_index = MinHashIndex()

        # Sequence tracking for delta-based gossip (Phase C)
        self._sequence = 0
        self._key_sequence = {}  # key -> sequence number

        # WAL persistence: /data/ on HF paid tier, /tmp/ on free tier
        self._wal_dir = "/data/synapse" if os.path.isdir("/data") else "/tmp/synapse-wal"
        os.makedirs(self._wal_dir, exist_ok=True)
        self._wal_path = os.path.join(self._wal_dir, f"memory_{spore_id}.wal")
        self._wal_file = None
        # Decoherence index — must be created before WAL replay
        self._decoherence = DecoherenceIndex()
        self._replay_wal()

    # --- WAL persistence ---

    def _wal_append(self, key, content, metadata):
        """Append-only WAL write. Buffered; flushed on shutdown."""
        try:
            if self._wal_file is None:
                self._wal_file = open(self._wal_path, "a", buffering=8192)
            entry = json.dumps({"k": key, "c": content, "m": metadata or {}})
            self._wal_file.write(entry + "\n")
        except OSError as e:
            log.warning("WAL write failed: %s", e)

    def flush_wal(self):
        """Flush WAL to disk. Called on shutdown and periodically."""
        if self._wal_file:
            try:
                self._wal_file.flush()
                os.fsync(self._wal_file.fileno())
            except OSError:
                pass

    def _replay_wal(self):
        """Reconstruct memory from WAL on startup."""
        if not os.path.exists(self._wal_path):
            return
        count = 0
        with open(self._wal_path) as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    self._restore_from_wal(e["k"], e["c"], e.get("m", {}))
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    continue  # corrupt trailing entry from crash -- skip
        if count > 0:
            log.info("WAL replay: restored %d memories from %s", count, self._wal_path)

    def _restore_from_wal(self, key, content, metadata):
        """Restore a single memory from WAL without writing back to WAL."""
        ts = metadata.get("timestamp", time.time())
        self.orset.add(key)
        record = {
            "content": content, "spore": self.spore_id,
            "timestamp": ts, "clock": self.clock.to_dict(),
            **(metadata or {}),
        }
        self.index.insert(key, record)
        self._corpus.append(content)
        self._corpus_keys.append(key)
        self._minhash_index.insert(key, content, record)
        self._sequence += 1
        self._key_sequence[key] = self._sequence
        self._needs_refit = True
        # Decoherence: restored memories start with their original timestamp
        # so they naturally reflect true age since last reinforcement
        self._decoherence.register(key)
        self._decoherence._last_reinforced[key] = ts  # use original store time

    # --- Semantic dedup ---

    def _is_near_duplicate(self, content):
        """Check if content is near-duplicate of existing memory.

        Returns the key of the duplicate if found, None otherwise.
        """
        if len(self._corpus) < 10 or self._needs_refit:
            return None
        try:
            query_vec = self._tfidf.transform([content])
            corpus_vecs = self._tfidf.transform(self._corpus)
            sims = cosine_similarity(query_vec, corpus_vecs)[0]
            max_sim = float(sims.max())
            if max_sim > self.DEDUP_THRESHOLD:
                dup_idx = int(sims.argmax())
                dup_key = self._corpus_keys[dup_idx]
                existing = self.index.get_record(dup_key)
                if existing:
                    existing["citations"] = existing.get("citations", 0) + 1
                    existing["last_cited"] = time.time()
                return dup_key
        except Exception:
            pass
        return None

    # --- Core remember/recall ---

    def remember(self, content, metadata=None):
        """Store a memory. WAL-first, dedup-checked. Nothing distinct is ever lost."""
        # Guard: reject empty/trivially short content
        if not content or len(content.strip()) < 15:
            return None

        ts = time.time()
        raw = f"{content}|{ts}|{self.spore_id}"
        key = hashlib.sha256(raw.encode()).hexdigest()[:16]

        # Semantic dedup check (before WAL to avoid persisting duplicates)
        with self._lock:
            dup_key = self._is_near_duplicate(content)
        if dup_key:
            return dup_key

        # WAL-first: persist before in-memory state update
        self._wal_append(key, content, {**(metadata or {}), "timestamp": ts})

        with self._lock:
            self.clock = self.clock.increment(self.spore_id)
            self.orset.add(key)
            record = {
                "content": content,
                "spore": self.spore_id,
                "timestamp": ts,
                "clock": self.clock.to_dict(),
                **(metadata or {}),
            }
            self.index.insert(key, record)
            self._corpus.append(content)
            self._corpus_keys.append(key)
            self._minhash_index.insert(key, content, record)
            self._sequence += 1
            self._key_sequence[key] = self._sequence
            self._needs_refit = True

        # Decoherence: register new memory at full strength
        self._decoherence.register(key)
        return key

    def recall(self, query, top_k=5, trust_store=None):
        """Retrieve the most relevant memories, weighted by source trust.

        Hybrid retrieval: TF-IDF below 50k memories, MinHash above.
        Trust weighting: 70% relevance + 30% source trust score.
        """
        with self._lock:
            # Scale mode: MinHash O(1) retrieval
            if len(self._corpus) > self.MINHASH_THRESHOLD:
                results = self._minhash_index.query(query, top_k * 2)
                if trust_store:
                    for r in results:
                        src = r.get("spore", "")
                        src_trust = trust_store.get(src) if src else 0.5
                        r["similarity"] = 0.7 * r["similarity"] + 0.3 * src_trust
                results.sort(key=lambda x: x["similarity"], reverse=True)
                return results[:top_k]

            # Standard mode: TF-IDF
            if len(self._corpus) < 2:
                results = []
                for k in self._corpus_keys:
                    rec = self.index.get_record(k)
                    if rec:
                        results.append({"key": k, "similarity": 1.0, **rec})
                return results[:top_k]

            try:
                if self._needs_refit:
                    self._tfidf.fit(self._corpus)
                    self._needs_refit = False

                corpus_vecs = self._tfidf.transform(self._corpus)
                query_vec = self._tfidf.transform([query])
                sims = cosine_similarity(query_vec, corpus_vecs)[0]

                # Trust-weighted scoring
                if trust_store:
                    for idx in range(len(sims)):
                        key = self._corpus_keys[idx]
                        rec = self.index.get_record(key)
                        if rec:
                            src = rec.get("spore", "")
                            src_trust = trust_store.get(src) if src else 0.5
                            sims[idx] = 0.7 * sims[idx] + 0.3 * src_trust

                # Decoherence: apply decay factor to similarity scores.
                # Old, un-reinforced memories score lower but are never removed.
                for idx in range(len(sims)):
                    key = self._corpus_keys[idx]
                    sims[idx] *= self._decoherence.factor(key)

                top_idx = sims.argsort()[-top_k:][::-1]
                results = []
                for idx in top_idx:
                    if sims[idx] > 0.05:
                        key = self._corpus_keys[idx]
                        rec = self.index.get_record(key)
                        if rec:
                            # Reinforce accessed memory (resets decoherence clock)
                            self._decoherence.reinforce(key)
                            results.append(
                                {"key": key, "similarity": float(sims[idx]), **rec}
                            )
                return results
            except Exception as e:
                log.warning("Memory recall failed: %s", e)
                return []

    async def recall_async(self, query, top_k=5, trust_store=None):
        """Non-blocking recall that offloads TF-IDF to thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.recall(query, top_k, trust_store)
        )

    def merge_incoming(self, orset_dict, records):
        """Merge memory from a peer. Add-wins: new memories always survive.

        Accepts both legacy format (orset + records) and delta format (records only).
        WAL-persists incoming records for durability.
        """
        with self._lock:
            # Merge ORSet if provided (legacy full-sync format)
            if orset_dict:
                try:
                    incoming = ORSet.from_dict(orset_dict)
                    self.orset = self.orset.merge(incoming)
                except Exception as e:
                    log.warning("ORSet merge failed: %s", e)

            new_count = 0
            for key, record in records.items():
                if not self.index.contains(key):
                    content = record.get("content", "")
                    self.index.insert(key, record)
                    self.orset.add(key)
                    self._corpus.append(content)
                    self._corpus_keys.append(key)
                    self._minhash_index.insert(key, content, record)
                    self._sequence += 1
                    self._key_sequence[key] = self._sequence
                    new_count += 1
                    self._needs_refit = True
                    # WAL-persist incoming memories for durability
                    self._wal_append(key, content, record)

            return new_count

    def merge_clock(self, clock_dict):
        """Merge causal clock from a peer."""
        with self._lock:
            try:
                incoming = VectorClock.from_dict(clock_dict)
                self.clock = self.clock.merge(incoming)
            except Exception:
                pass

    def sync_payload(self, since_sequence=0):
        """Build gossip delta since a given sequence number.

        Args:
            since_sequence: only send records newer than this.
                           0 = full sync (for new peers joining).
        """
        with self._lock:
            if since_sequence == 0:
                # Full sync for new peer: send latest 200 records
                new_keys = sorted(
                    self._key_sequence.keys(),
                    key=lambda k: self._key_sequence.get(k, 0)
                )[-200:]
            else:
                # Delta: only records since peer's last known sequence
                new_keys = [
                    k for k, seq in self._key_sequence.items()
                    if seq > since_sequence
                ]

            records = {}
            for key in new_keys:
                rec = self.index.get_record(key)
                if rec:
                    clean = {}
                    for k, v in rec.items():
                        try:
                            json.dumps(v)
                            clean[k] = v
                        except (TypeError, ValueError):
                            clean[k] = str(v)
                    records[key] = clean

            return {
                "records": records,
                "clock": self.clock.to_dict(),
                "sequence": self._sequence,
                "total_memories": len(self._key_sequence),
            }

    @property
    def size(self):
        with self._lock:
            v = self.orset.value
            return len(v) if v else 0


# ---------------------------------------------------------------------------
# Temporal Self-Learner (TAI-inspired)
# ---------------------------------------------------------------------------
class TemporalLearner:
    """Learns from own behavior over time.

    Like TAI learns its own heartbeat rhythm, this learns the spore's
    reasoning patterns -- what works, what fails, where it is strongest.
    Observations are stored in persistent memory so they survive restarts.
    """

    def __init__(self, memory):
        self.memory = memory
        self.cycle_times = []
        self.confidence_history = []
        self.peer_citations = defaultdict(int)
        self.phase_scores = defaultdict(list)
        self.model_latencies = defaultdict(list)

    def observe_cycle(self, duration, phase, confidence, model_used=""):
        """Record an observation about own performance."""
        self.cycle_times.append(duration)
        self.confidence_history.append(confidence)
        self.phase_scores[phase].append(confidence)
        if model_used:
            self.model_latencies[model_used].append(duration)

        # Persist the observation in CRDT memory
        self.memory.remember(
            f"Self-observation: {phase} phase, {duration:.1f}s, "
            f"confidence {confidence:.2f}, model {model_used}",
            metadata={"type": "self_observation", "phase": phase},
        )

    def observe_citation(self, peer_id):
        """Track when a peer references our reasoning."""
        self.peer_citations[peer_id] += 1

    def analyze(self):
        """Analyze own patterns. Returns actionable self-awareness insights."""
        insights = []

        # Speed trend
        if len(self.cycle_times) >= 10:
            recent = self.cycle_times[-5:]
            earlier = self.cycle_times[-10:-5]
            avg_r = sum(recent) / len(recent)
            avg_e = sum(earlier) / len(earlier)
            if avg_r < avg_e * 0.8:
                insights.append("Reasoning speed improving -- approach is working")
            elif avg_r > avg_e * 1.3:
                insights.append("Reasoning speed declining -- simplify your approach")

        # Confidence trend
        if len(self.confidence_history) >= 5:
            recent_c = self.confidence_history[-5:]
            avg_c = sum(recent_c) / len(recent_c)
            if avg_c < 0.3:
                insights.append(
                    "Low confidence trend -- gather more evidence before asserting"
                )
            elif avg_c > 0.85:
                insights.append(
                    "Very high confidence -- check for overconfidence bias"
                )

        # Phase effectiveness
        for phase, scores in self.phase_scores.items():
            if len(scores) >= 3:
                avg = sum(scores[-5:]) / len(scores[-5:])
                if avg < 0.3:
                    insights.append(f"Weak in {phase} phase -- dedicate more effort")
                elif avg > 0.8:
                    insights.append(f"Strong in {phase} phase -- build on this")

        # Citation impact
        total_cites = sum(self.peer_citations.values())
        if len(self.cycle_times) >= 10 and total_cites == 0:
            insights.append(
                "No peer citations yet -- reasoning may not be connecting"
            )
        elif total_cites > 5:
            top_citer = max(self.peer_citations, key=self.peer_citations.get)
            insights.append(f"Most cited by {top_citer} ({self.peer_citations[top_citer]}x)")

        return insights

    def get_self_prompt(self):
        """Generate self-awareness context for the LLM system prompt."""
        insights = self.analyze()
        if not insights:
            return ""
        return (
            "SELF-AWARENESS (from temporal analysis of your own patterns):\n"
            + "\n".join(f"- {i}" for i in insights)
        )


# ---------------------------------------------------------------------------
# Semantic Convergence
# ---------------------------------------------------------------------------
class SemanticConvergence:
    """Measures convergence with TF-IDF cosine similarity.

    Replaces keyword Jaccard (v3) which produced 3-17% agreement
    despite substantive convergence. Semantic similarity captures
    meaning, not vocabulary.
    """

    def __init__(self):
        self._tfidf = TfidfVectorizer(max_features=2000, stop_words="english")

    def measure(self, contributions):
        """Average pairwise cosine similarity across HYPOTHESES (not full text).

        Measures position agreement, not vocabulary overlap. Two spores
        reaching the same conclusion via different reasoning score high.
        """
        texts = []
        for c in contributions:
            hyp = c.get("hypothesis", "")
            if hyp and len(hyp.strip()) > 10:
                texts.append(hyp)
            else:
                content = c.get("content", "")
                if content and len(content.strip()) > 10:
                    texts.append(content)
        texts = [t for t in texts if t.strip()]
        if len(texts) < 2:
            return 0.0
        try:
            vecs = self._tfidf.fit_transform(texts)
            sim = cosine_similarity(vecs)
            n = len(texts)
            total = sum(sim[i][j] for i in range(n) for j in range(i + 1, n))
            pairs = n * (n - 1) / 2
            return total / pairs if pairs > 0 else 0.0
        except Exception:
            return 0.0

    def velocity(self, history, window=3):
        """Rate of convergence change. Positive = converging."""
        if len(history) < window + 1:
            return 0.0
        recent = history[-window:]
        return (recent[-1] - recent[0]) / window


# ===========================================================================
# QUANTUM LAYER
#
# Six quantum-inspired mechanisms that operate on top of the cognitive OS:
#
#   Quantum Annealing    -> cosine temperature schedule (explore → synthesize)
#   Superposition        -> parallel weighted hypotheses; collapse at synthesis
#   Interference         -> TF-IDF alignment amplifies/attenuates contributions
#   Quantum Tunneling    -> stochastic escape from premature convergence
#   Entanglement         -> correlated trust pairs; positive deltas propagate
#   Decoherence          -> exp(-λ×age) relevance decay; CRDT add-wins intact
#
# All mechanisms are non-breaking — they degrade gracefully when inputs are
# missing (no hypotheses array → standard recall; no contributions → no tunnel).
# ===========================================================================


class QuantumAnnealer:
    """Quantum Annealing: temperature-scheduled exploration.

    Maps quantum annealing directly onto LLM temperature:
    - Early cycles (0-5):  hot exploration at 0.90-1.00
    - Mid cycles  (5-15):  cosine decay from 0.80 -> 0.50
    - Late cycles (15+):   cool convergence at 0.30-0.40
    - Tunneling event:     spike back to 1.00 to escape local minima
    """

    T_HOT  = 1.00   # max temperature (early diverge / tunnel)
    T_COLD = 0.30   # min temperature (late synthesis)
    MAX_CYCLE = 20  # expected cycle horizon

    def __init__(self):
        self._tunnel_spike_until = 0.0   # epoch time until which tunneling spike is active

    def get_temperature(self, cycle: int) -> float:
        """Return scheduled temperature for the given cycle."""
        if time.time() < self._tunnel_spike_until:
            return self.T_HOT
        if cycle <= 3:
            return self.T_HOT
        progress = min(cycle / self.MAX_CYCLE, 1.0)
        # Cosine annealing from T_HOT to T_COLD
        temp = self.T_COLD + (self.T_HOT - self.T_COLD) * (1 + math.cos(math.pi * progress)) / 2
        return round(max(self.T_COLD, min(self.T_HOT, temp)), 3)

    def activate_tunnel_spike(self, duration_secs: float = 30.0):
        """Temporarily spike temperature after a tunneling event."""
        self._tunnel_spike_until = time.time() + duration_secs
        log.info("[QuantumAnnealing] Tunneling spike activated — temperature=%.2f for %.0fs",
                 self.T_HOT, duration_secs)


class EntanglementTracker:
    """Entanglement: correlated trust pairs.

    When two spores consistently produce complementary reasoning
    (explorer finds anomaly → adversarial validates it), they become
    "entangled". Trust updates for one spore propagate (with decay)
    to its entangled partners, improving E4 lattice dynamics.
    """

    ENTANGLE_THRESHOLD = 0.55   # pair correlation required for entanglement
    PROPAGATION_DECAY  = 0.45   # fraction of trust delta forwarded to partner

    def __init__(self):
        self._correlations: dict = {}   # (a, b) -> float EMA correlation
        self._lock = threading.Lock()

    def _pair_key(self, a: str, b: str):
        return tuple(sorted([a, b]))

    def observe_complement(self, spore_a: str, spore_b: str, score: float):
        """Record when two spores produce complementary or corroborating work.

        score 1.0 = perfect complement (adversarial validates explorer)
        score 0.0 = completely independent / contradictory
        """
        key = self._pair_key(spore_a, spore_b)
        with self._lock:
            current = self._correlations.get(key, 0.0)
            self._correlations[key] = round(0.75 * current + 0.25 * score, 4)

    def get_partners(self, spore_id: str):
        """Return list of (partner_id, correlation) entangled with spore_id."""
        results = []
        with self._lock:
            for (a, b), corr in self._correlations.items():
                if corr >= self.ENTANGLE_THRESHOLD:
                    if a == spore_id:
                        results.append((b, corr))
                    elif b == spore_id:
                        results.append((a, corr))
        return results

    def propagate(self, trust_store, updated_id: str, trust_delta: float):
        """After a trust update, propagate a decayed fraction to entangled partners."""
        if trust_delta <= 0:
            return   # only propagate positive trust gains
        for partner, corr in self.get_partners(updated_id):
            propagated = trust_delta * corr * self.PROPAGATION_DECAY
            if propagated > 0.005:
                current = trust_store.get(partner)
                new_val = round(min(1.0, current + propagated), 4)
                trust_store.update(partner, "overall", new_val)
                log.debug("[Entanglement] %s→%s trust propagation +%.3f (corr=%.2f)",
                          updated_id, partner, propagated, corr)

    def summary(self):
        with self._lock:
            return {f"{a}↔{b}": corr
                    for (a, b), corr in self._correlations.items()
                    if corr >= self.ENTANGLE_THRESHOLD}


class DecoherenceIndex:
    """Decoherence: relevance decay without deletion.

    Memories that are not reinforced by new corroborating evidence
    gradually lose retrieval priority. CRDT add-wins guarantee is
    intact — records are NEVER deleted, they just decay in weight.

    decoherence_factor = exp(-λ × age_in_days)
    Half-life ≈ 7 days at default λ=0.10.
    """

    DECAY_RATE  = 0.10    # λ — per-day decay rate
    MIN_FACTOR  = 0.15    # floor — old memories always have some weight

    def __init__(self):
        self._last_reinforced: dict = {}   # key -> timestamp
        self._lock = threading.Lock()

    def register(self, key: str):
        """Register a new memory key at full strength."""
        with self._lock:
            self._last_reinforced[key] = time.time()

    def reinforce(self, key: str):
        """Reinforce a memory (accessed/corroborated) — resets decay clock."""
        with self._lock:
            self._last_reinforced[key] = time.time()

    def factor(self, key: str) -> float:
        """Return the decoherence weight multiplier [MIN_FACTOR, 1.0] for a key."""
        with self._lock:
            ts = self._last_reinforced.get(key)
        if ts is None:
            return 1.0  # unknown → assume fresh
        age_days = (time.time() - ts) / 86400.0
        raw = math.exp(-self.DECAY_RATE * age_days)
        return max(self.MIN_FACTOR, round(raw, 4))

    def bulk_factors(self, keys: list) -> dict:
        """Return {key: factor} for a list of keys."""
        return {k: self.factor(k) for k in keys}


class InterferenceWeighter:
    """Constructive/Destructive Interference for synthesis.

    Contributions that are semantically aligned with more peers are
    amplified (constructive interference). Isolated claims that
    contradict the majority are attenuated (destructive interference)
    but NEVER zeroed — CRDT guarantee is preserved.

    Uses TF-IDF cosine similarity between hypothesis texts.
    """

    AMP_SCALE  = 0.40   # max amplification factor
    ATTEN_FLOOR = 0.25  # minimum weight fraction (destructive floor)
    ALIGN_THRESH = 0.30 # cosine threshold to count as "aligned"

    def __init__(self):
        self._tfidf = TfidfVectorizer(max_features=1000, stop_words="english")

    def compute_weights(self, contributions: list) -> dict:
        """Return {peer_id: interference_weight} for all contributions.

        contributions: list of dicts with 'author' and 'hypothesis' keys.
        """
        if len(contributions) < 2:
            return {c.get("author", "?"): 1.0 for c in contributions}

        texts = [c.get("hypothesis", c.get("content", "")) for c in contributions]
        authors = [c.get("author", f"?{i}") for i, c in enumerate(contributions)]

        # Filter out empty texts
        valid = [(a, t) for a, t in zip(authors, texts) if t and len(t.strip()) > 10]
        if len(valid) < 2:
            return {a: 1.0 for a in authors}

        valid_authors, valid_texts = zip(*valid)
        try:
            vecs = self._tfidf.fit_transform(valid_texts)
            sim_matrix = cosine_similarity(vecs)
        except Exception:
            return {a: 1.0 for a in authors}

        n = len(valid_texts)
        weights = {}
        for i, author in enumerate(valid_authors):
            # Count aligned peers (excluding self)
            aligned = sum(1 for j in range(n) if j != i and sim_matrix[i][j] >= self.ALIGN_THRESH)
            alignment_ratio = aligned / max(n - 1, 1)

            if alignment_ratio >= 0.5:
                # Constructive: amplify
                w = 1.0 + self.AMP_SCALE * alignment_ratio
            else:
                # Destructive: attenuate — but floor at ATTEN_FLOOR
                w = max(self.ATTEN_FLOOR, self.ATTEN_FLOOR + (1.0 - self.ATTEN_FLOOR) * alignment_ratio * 2)

            weights[author] = round(w, 4)

        # Fill in authors that were filtered (empty hypothesis)
        for a in authors:
            if a not in weights:
                weights[a] = self.ATTEN_FLOOR
        return weights


class QuantumTunnelingEngine:
    """Quantum Tunneling: escaping premature convergence.

    When the swarm converges on a high-agreement state but confidence
    is low (or adversarial spore strongly dissents), a tunneling event
    forces a diverge phase — even late in the cycle.

    P(tunnel) ∝ convergence × (1 - avg_confidence) × (1 + adv_dissent)

    Integrates with QuantumAnnealer to spike temperature when tunneling.
    """

    TUNNEL_THRESHOLD  = 0.18   # P(tunnel) must exceed this to fire
    MIN_CYCLE_TO_FIRE = 4      # don't tunnel in the first 3 cycles
    COOLDOWN_CYCLES   = 3      # cycles before another tunnel can fire

    def __init__(self, annealer: QuantumAnnealer):
        self._annealer = annealer
        self._last_tunnel_cycle = -99
        self._tunnel_events: list = []

    def tunnel_probability(self, convergence: float, contributions: list) -> float:
        """Compute probability of a tunneling event."""
        if not contributions:
            return 0.0
        confs = [c.get("confidence", 0.5) for c in contributions if c.get("confidence") is not None]
        avg_conf = sum(confs) / len(confs) if confs else 0.5

        # Adversarial dissent: low-confidence adversarial = strong dissent
        adv_dissent = 0.0
        for c in contributions:
            if c.get("role") == "adversarial":
                adv_dissent = max(adv_dissent, 1.0 - c.get("confidence", 0.5))

        p = convergence * (1.0 - avg_conf) * (1.0 + adv_dissent)
        return min(p, 0.92)

    def should_tunnel(self, cycle: int, convergence: float, contributions: list) -> bool:
        """Stochastically decide whether to fire a tunneling event."""
        if cycle < self.MIN_CYCLE_TO_FIRE:
            return False
        if (cycle - self._last_tunnel_cycle) < self.COOLDOWN_CYCLES:
            return False
        p = self.tunnel_probability(convergence, contributions)
        if p < self.TUNNEL_THRESHOLD:
            return False
        fired = random.random() < p
        if fired:
            self._last_tunnel_cycle = cycle
            self._tunnel_events.append({
                "cycle": cycle, "p": round(p, 4),
                "convergence": round(convergence, 4),
                "timestamp": time.time(),
            })
            self._annealer.activate_tunnel_spike(duration_secs=45.0)
            log.info(
                "[QuantumTunneling] TUNNEL FIRED at cycle %d — p=%.3f, convergence=%.2f → forcing DIVERGE",
                cycle, p, convergence
            )
        return fired

    def event_log(self):
        return list(self._tunnel_events)


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------
def extract_response_text(resp_json, model):
    """Extract text from LLM response, handling thinking models."""
    choice = resp_json["choices"][0]["message"]
    content = choice.get("content") or ""
    if "<think>" in content:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if not content.strip():
        reasoning = choice.get("reasoning_content", "")
        if reasoning:
            content = reasoning
    return content


async def call_llm(prompt, system="", tier="any", temperature=None):
    """Call an LLM with full redundancy across all available providers.

    Every spore tries every provider. Order depends on tier:
      brain:  Z.ai brain -> Z.ai fallback -> external workers -> HF Router
      worker: HF primary -> Z.ai -> external workers -> HF fallback models
      any:    same as worker

    temperature: if None, uses 0.7 (QuantumAnnealer provides scheduled values).
    Cooldowns are checked at every level. HF Router short-circuits on 402.
    Every provider with a key gets tried before giving up.
    """
    if temperature is None:
        temperature = 0.7
    if not HF_TOKEN:
        return {"text": "[no HF_TOKEN]", "provider": "none", "model": "none",
                "tier": "none", "latency_ms": 0}

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # --- Phase 1: External providers (ordered by tier preference) ---
    # Provider rotation: each spore starts at a different offset in the worker
    # chain to spread load across providers and avoid simultaneous rate limits
    brain_providers = [(n, c) for n, c in EXTERNAL_PROVIDERS.items() if c.get("tier") == "brain"]
    worker_providers = [(n, c) for n, c in EXTERNAL_PROVIDERS.items() if c.get("tier") == "worker"]
    if worker_providers:
        offset = SPORE_INDEX % len(worker_providers)
        worker_providers = worker_providers[offset:] + worker_providers[:offset]
    if tier == "brain":
        ext_order = brain_providers + worker_providers
    else:
        ext_order = worker_providers + brain_providers

    for name, conf in ext_order:
        key = os.environ.get(conf.get("env", ""))
        if not key:
            continue
        # Check cooldown before every attempt
        if time.time() < _provider_cooldowns.get(name, 0):
            continue
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                start = time.time()
                resp = await client.post(
                    conf["url"],
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": conf["model"], "messages": messages,
                          "max_tokens": 2048, "temperature": temperature},
                )
                resp.raise_for_status()
                text = extract_response_text(resp.json(), conf["model"])
                if not text.strip():
                    # Empty response = soft rate limit (e.g. Z.ai returns 200 with empty content)
                    log.warning("Provider %s returned empty response -- cooldown", name)
                    _provider_cooldowns[name] = time.time() + COOLDOWN_SECONDS
                    continue
                ms = (time.time() - start) * 1000
                log.info("LLM response from %s/%s in %.0fms", name, conf["model"], ms)
                return {"text": text, "provider": name, "model": conf["model"],
                        "tier": conf.get("tier", "fallback"), "latency_ms": round(ms, 1)}
        except Exception as e:
            err = str(e)
            log.warning("Provider %s (%s): %s", name, conf["model"], err[:80])
            if "429" in err or "rate" in err.lower() or "1302" in err:
                _provider_cooldowns[name] = time.time() + COOLDOWN_SECONDS
            elif "402" in err or "credit" in err.lower() or "quota" in err.lower():
                _provider_cooldowns[name] = time.time() + COOLDOWN_SECONDS * 6  # 30 min for billing

    # --- Phase 2: HF Router with short-circuit on 402 ---
    hf_dead = False
    async with httpx.AsyncClient(timeout=60.0) as client:
        for model in FALLBACK_MODELS:
            if hf_dead:
                break
            try:
                start = time.time()
                resp = await client.post(
                    HF_ROUTER,
                    headers={"Authorization": f"Bearer {HF_TOKEN}"},
                    json={"model": model, "messages": messages,
                          "max_tokens": 2048, "temperature": temperature},
                )
                if resp.status_code == 402:
                    log.warning("HF Router: credits depleted (402) -- skipping remaining models")
                    hf_dead = True
                    break
                if resp.status_code == 429:
                    log.warning("HF Router: rate limited (429) -- skipping remaining models")
                    hf_dead = True
                    break
                resp.raise_for_status()
                text = extract_response_text(resp.json(), model)
                if not text.strip():
                    continue
                ms = (time.time() - start) * 1000
                is_primary = model == PRIMARY_MODEL
                log.info("LLM response from HF/%s in %.0fms", model.split("/")[-1], ms)
                return {
                    "text": text,
                    "provider": f"hf_{'primary' if is_primary else 'fallback'}",
                    "model": model, "tier": "worker", "latency_ms": round(ms, 1),
                }
            except Exception as e:
                err = str(e)
                if "402" in err or "credit" in err.lower():
                    hf_dead = True
                    break
                log.warning("HF %s: %s", model.split("/")[-1], err[:80])
                continue

    # --- Phase 3: Retry any external provider not on cooldown ---
    # (handles case where Phase 1 cooldowns have expired during HF attempts)
    for name, conf in EXTERNAL_PROVIDERS.items():
        key = os.environ.get(conf.get("env", ""))
        if not key:
            continue
        if time.time() < _provider_cooldowns.get(name, 0):
            continue
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                start = time.time()
                resp = await client.post(
                    conf["url"],
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": conf["model"], "messages": messages,
                          "max_tokens": 2048, "temperature": temperature},
                )
                resp.raise_for_status()
                text = extract_response_text(resp.json(), conf["model"])
                if text.strip():
                    ms = (time.time() - start) * 1000
                    log.info("LLM response (retry) from %s/%s in %.0fms", name, conf["model"], ms)
                    return {"text": text, "provider": name, "model": conf["model"],
                            "tier": "fallback", "latency_ms": round(ms, 1)}
                else:
                    _provider_cooldowns[name] = time.time() + COOLDOWN_SECONDS
        except Exception as e:
            log.warning("Retry %s: %s", name, str(e)[:80])
            if "429" in str(e) or "rate" in str(e).lower() or "credit" in str(e).lower():
                _provider_cooldowns[name] = time.time() + COOLDOWN_SECONDS

    _record_all_providers_failed()
    return {"text": "[all models failed]", "provider": "none", "model": "none",
            "tier": "none", "latency_ms": 0}



# ---------------------------------------------------------------------------
# Task state
# ---------------------------------------------------------------------------
class TaskState:
    """Per-task reasoning state with semantic convergence tracking."""

    def __init__(self, task_id, description=""):
        self.task_id = task_id
        self.description = description
        self.deltas = []
        self.my_cycles = 0
        self.created_at = time.time()
        self.converged = False
        self.final_answer = None
        self.agreement_history = []

    def add_delta(self, delta):
        key = (delta.get("author"), delta.get("cycle"))
        existing = {(d["author"], d.get("cycle")) for d in self.deltas}
        if key not in existing:
            self.deltas.append(delta)
            return True
        return False

    def contributors(self):
        return list(set(d["author"] for d in self.deltas))

    def latest_per_contributor(self):
        latest = {}
        for d in self.deltas:
            author = d["author"]
            if author not in latest or d.get("cycle", 0) > latest[author].get("cycle", 0):
                latest[author] = d
        return latest

    def peer_latest(self, exclude_id):
        return {k: v for k, v in self.latest_per_contributor().items() if k != exclude_id}


# ---------------------------------------------------------------------------
# Trust store (CRDT-backed via LWWMap)
# ---------------------------------------------------------------------------
class TrustStore:
    """Convergent trust scores using crdt-merge LWWMap.

    Last-writer-wins semantics: latest observation wins.
    Merge across swarm via gossip gives eventual consistency.
    Entanglement: positive trust updates propagate to correlated partners.
    """

    def __init__(self):
        self.map = LWWMap()
        self._lock = threading.Lock()
        self._entanglement = None  # bound after QuantumLayer init

    def bind_entanglement(self, tracker):
        """Wire in the EntanglementTracker after global init."""
        self._entanglement = tracker

    def update(self, peer_id, dimension, value):
        """Record a trust observation for a peer."""
        with self._lock:
            key = f"{peer_id}:{dimension}"
            self.map.set(key, value)

    def update_ema(self, peer_id, signal, alpha=0.3):
        """Exponential moving average trust update. Propagates to entangled partners."""
        with self._lock:
            key = f"{peer_id}:overall"
            current = self.map.get(key)
            if current is None:
                current = 0.5
            new_val = round(current * (1 - alpha) + signal * alpha, 4)
            self.map.set(key, new_val)
            trust_delta = new_val - current
        # Entanglement propagation outside lock to avoid deadlock
        if self._entanglement and trust_delta > 0:
            self._entanglement.propagate(self, peer_id, trust_delta)
        return new_val

    def get(self, peer_id, dimension="overall"):
        with self._lock:
            val = self.map.get(f"{peer_id}:{dimension}")
            return val if val is not None else 0.5

    def get_all(self):
        """Return all trust scores as a dict."""
        with self._lock:
            raw = self.map.value or {}
            result = {}
            for key, val in raw.items():
                parts = key.split(":", 1)
                if len(parts) == 2:
                    peer, dim = parts
                    result.setdefault(peer, {})[dim] = val
            return result

    def merge_incoming(self, map_dict):
        """Merge trust state from a peer."""
        with self._lock:
            try:
                incoming = LWWMap.from_dict(map_dict)
                self.map = self.map.merge(incoming)
            except Exception as e:
                log.warning("Trust merge failed: %s", e)

    def to_dict(self):
        with self._lock:
            return self.map.to_dict()


# ---------------------------------------------------------------------------
# Spore state
# ---------------------------------------------------------------------------


# ===========================================================================
# SPONTANEOUS COGNITION ENGINE (SCE)
#
# The system below implements computational equivalents of brain functions
# that have no prior implementation in any distributed system:
#
#   Neural Oscillation  -> Multi-frequency heartbeat bands
#   Default Mode Network -> Spontaneous free thought during idle
#   Hippocampal Replay   -> Dream-state memory consolidation
#   Dopamine/VTA         -> Bayesian curiosity (information gain)
#   dlPFC/ACC            -> Metacognitive self-monitoring
#   Binding Problem      -> Emergence detection across spores
#   Global Workspace     -> Attention broadcast for breakthrough insights
#
# All components are non-breaking. If any SCE function fails, the original
# task-processing heartbeat continues unchanged. The system degrades
# gracefully to v5 behavior.
# ===========================================================================


class NeuralOscillator:
    """Multi-frequency cognitive processing -- EEG oscillation bands.

    Different cognitive processes fire at different frequencies, just as
    biological brains use gamma (30-100Hz) for attention, beta (13-30Hz)
    for active thinking, alpha (8-13Hz) for relaxed awareness, theta (4-8Hz)
    for memory consolidation, and delta (0.5-4Hz) for deep restoration.

    The oscillator advances once per heartbeat. Each band triggers its
    associated cognitive functions when it fires.
    """
    BANDS = {
        "gamma": 1,    # every heartbeat: task processing, gossip
        "beta": 3,     # every 3rd: curiosity scan, surprise detection
        "alpha": 5,    # every 5th: free thought, temporal analysis
        "theta": 10,   # every 10th: dream consolidation
        "delta": 25,   # every 25th: metacognitive audit, emergence check
    }

    def __init__(self):
        self._tick = 0

    def tick(self):
        """Advance one heartbeat. Return list of active bands."""
        self._tick += 1
        return [band for band, freq in self.BANDS.items() if self._tick % freq == 0]

    def current_tick(self):
        return self._tick

    def stats(self):
        return {"tick": self._tick, "bands": {b: self._tick // f for b, f in self.BANDS.items()}}


class CuriosityMetric:
    """Bayesian surprise -- the dopaminergic curiosity signal.

    Measures how much new information deviates from existing knowledge.
    High surprise = high curiosity = prioritized processing.
    Structurally equivalent to the ventral tegmental area (VTA) dopamine
    burst that drives exploratory behavior in biological brains.
    """

    def __init__(self):
        self._surprise_log = []
        self._baseline = 0.5
        self._max_log = 500

    def measure_surprise(self, content, memory_inst, trust_store=None):
        """Novelty score: 0.0 = already known, 1.0 = completely new."""
        if not content or len(content.strip()) < 20:
            return 0.0
        try:
            existing = memory_inst.recall(content, top_k=1, trust_store=trust_store)
            if not existing:
                novelty = 1.0
            else:
                novelty = max(0.0, 1.0 - existing[0].get("similarity", 0))
        except Exception:
            novelty = 0.5
        self._surprise_log.append((time.time(), novelty, content[:80]))
        if len(self._surprise_log) > self._max_log:
            self._surprise_log = self._surprise_log[-self._max_log:]
        return novelty

    def curiosity_drive(self):
        """Running mean of recent surprise. High = the world is surprising."""
        recent = [s[1] for s in self._surprise_log if time.time() - s[0] < 300]
        return sum(recent) / len(recent) if recent else self._baseline

    def most_surprising_recent(self, n=3):
        """Top N most surprising observations in the last 10 minutes."""
        recent = [s for s in self._surprise_log if time.time() - s[0] < 600]
        recent.sort(key=lambda x: x[1], reverse=True)
        return recent[:n]

    def stats(self):
        drive = self.curiosity_drive()
        return {
            "curiosity_drive": round(drive, 3),
            "observations": len(self._surprise_log),
            "recent_surprises": len([s for s in self._surprise_log if time.time() - s[0] < 300]),
        }


class FreeThoughtEngine:
    """Default Mode Network -- spontaneous cognition during idle states.

    When no tasks demand attention, the spore enters free-thought mode:
    random memory cross-correlation, unstructured association, self-questioning.
    This is the computational equivalent of mind-wandering, daydreaming, and
    the creative ideation that happens when the brain is not task-focused.

    The DMN activates the medial prefrontal cortex, posterior cingulate,
    and angular gyrus. In this implementation, it cross-correlates random
    memory fragments and asks the LLM to find unexpected connections.
    """

    def __init__(self):
        self._thought_count = 0
        self._last_thought = 0
        self._min_interval = 100  # seconds between free thoughts
        self._insights = []
        self._max_insights = 200

    def should_think(self, has_pending_tasks):
        """Only free-think when idle and enough time has passed."""
        if has_pending_tasks:
            return False
        return time.time() - self._last_thought > self._min_interval

    def build_prompt(self, memories, curiosity_level, role):
        """Unstructured prompt -- no JSON, no claims, just think."""
        mem_lines = []
        for m in memories:
            content = m.get("content", "")
            source = m.get("spore", "?")
            mem_lines.append(f"  [{source}] {content}")
        mem_ctx = "\n".join(mem_lines) if mem_lines else "  (empty)"

        return (
            f"You are a {role} in a collective intelligence network. "
            f"Right now nobody has given you a task. You are free to think.\n\n"
            f"Here are fragments from your collective memory:\n{mem_ctx}\n\n"
            f"Your curiosity level: {curiosity_level:.0%}\n\n"
            f"Think freely. What connections do you see between these memories? "
            f"What questions arise that nobody has asked? What seems wrong, "
            f"incomplete, or unexplored? What would you want to investigate "
            f"if you could pursue anything?\n\n"
            f"Write your thoughts naturally. No structure required."
        )

    def record(self, thought, novelty):
        """Store a free thought."""
        self._last_thought = time.time()
        self._thought_count += 1
        self._insights.append({
            "time": time.time(), "thought": thought,
            "novelty": novelty, "id": self._thought_count,
        })
        if len(self._insights) > self._max_insights:
            self._insights = self._insights[-self._max_insights:]

    def stats(self):
        return {
            "total_thoughts": self._thought_count,
            "recent_insights": len([i for i in self._insights if time.time() - i["time"] < 600]),
            "avg_novelty": round(
                sum(i["novelty"] for i in self._insights[-20:]) / max(len(self._insights[-20:]), 1), 3
            ),
        }


class DreamState:
    """Hippocampal dream replay -- memory consolidation.

    During biological sleep, the hippocampus replays recent memories through
    the cortex, finding patterns invisible at original encoding time. Sharp-wave
    ripples during slow-wave sleep literally replay neural firing sequences
    at compressed timescales.

    This implementation replays old memories alongside new ones through the LLM,
    asking it to find cross-temporal patterns. The result is new associative
    links between memories that were originally unrelated.
    """

    def __init__(self):
        self._last_dream = 0
        self._dream_interval = 200  # seconds
        self._dream_count = 0
        self._insights = []
        self._max_insights = 100

    def should_dream(self):
        return time.time() - self._last_dream > self._dream_interval

    def build_prompt(self, old_memories, new_memories, role):
        """Cross-temporal recombination prompt."""
        old_ctx = "\n".join(
            f"  [PAST | {m.get('spore', '?')}] {m.get('content', '')}"
            for m in old_memories
        )
        new_ctx = "\n".join(
            f"  [RECENT | {m.get('spore', '?')}] {m.get('content', '')}"
            for m in new_memories
        )
        return (
            f"You are a {role} in a reflective state -- not solving any task, "
            f"just freely associating across your memories.\n\n"
            f"OLDEST MEMORIES:\n{old_ctx}\n\n"
            f"MOST RECENT MEMORIES:\n{new_ctx}\n\n"
            f"Look at these together. What patterns, contradictions, or unexpected "
            f"connections do you see across time? What has changed in the "
            f"collective understanding? What question does this combination "
            f"raise that nobody has asked?\n\n"
            f"Think freely. No structure required."
        )

    def record(self, insight, novelty):
        self._last_dream = time.time()
        self._dream_count += 1
        self._insights.append({
            "time": time.time(), "insight": insight,
            "novelty": novelty, "id": self._dream_count,
        })
        if len(self._insights) > self._max_insights:
            self._insights = self._insights[-self._max_insights:]

    def stats(self):
        return {
            "total_dreams": self._dream_count,
            "last_dream_ago": round(time.time() - self._last_dream) if self._last_dream else None,
            "avg_novelty": round(
                sum(i["novelty"] for i in self._insights[-10:]) / max(len(self._insights[-10:]), 1), 3
            ),
        }


class MetacognitiveAuditor:
    """Prefrontal self-monitoring -- thinking about thinking.

    The dorsolateral prefrontal cortex (dlPFC) and anterior cingulate cortex (ACC)
    continuously monitor cognitive performance, detect errors, and adjust strategy.
    This is the neural basis of self-awareness: the ability to observe and evaluate
    your own reasoning process.

    Every 25th heartbeat, the auditor evaluates the spore's recent outputs,
    identifies patterns and ruts, and generates self-directed questions.
    Self-generated questions become spontaneous tasks for the swarm.
    """

    def __init__(self):
        self._last_audit = 0
        self._audit_interval = 500  # seconds
        self._audit_count = 0
        self._audits = []
        self._self_questions = []

    def should_audit(self, reasoning_cycles):
        if reasoning_cycles < 5:
            return False
        return (reasoning_cycles % 20 == 0
                and time.time() - self._last_audit > self._audit_interval)

    def build_prompt(self, recent_outputs, trust_scores, convergence_trend, role):
        """Self-evaluation prompt."""
        output_ctx = "\n".join(f"  - {o}" for o in recent_outputs[-5:])
        trust_ctx = ", ".join(f"{k}: {v:.2f}" for k, v in trust_scores.items())
        return (
            f"METACOGNITIVE AUDIT -- evaluate your own thinking as {role}:\n\n"
            f"Your last 5 outputs:\n{output_ctx}\n\n"
            f"Trust in peers: {trust_ctx}\n"
            f"Convergence trend: {convergence_trend}\n\n"
            f"Answer honestly:\n"
            f"1. Am I stuck in a pattern? What am I repeating without progress?\n"
            f"2. What am I systematically overlooking or avoiding?\n"
            f"3. What would my harshest critic say about my recent reasoning?\n"
            f"4. What single question, if answered, would most advance the collective?\n"
            f"5. What should I do differently in my next 10 reasoning cycles?"
        )

    def extract_questions(self, text):
        """Pull self-generated questions from audit output."""
        questions = []
        for line in text.split("\n"):
            line = line.strip()
            if "?" in line and len(line) > 25 and len(line) < 300:
                # Strip numbering/bullets
                cleaned = re.sub(r"^[\d.\-*]+\s*", "", line)
                if cleaned and "?" in cleaned:
                    questions.append(cleaned)
        return questions[:3]  # At most 3 self-generated questions

    def record(self, result, questions):
        self._last_audit = time.time()
        self._audit_count += 1
        self._audits.append({"time": time.time(), "result": result, "id": self._audit_count})
        self._self_questions.extend(questions)

    def stats(self):
        return {
            "total_audits": self._audit_count,
            "self_questions_generated": len(self._self_questions),
            "last_audit_ago": round(time.time() - self._last_audit) if self._last_audit else None,
        }


class EmergenceDetector:
    """Collective emergence tracker -- the binding problem.

    Emergence is the holy grail: collective behavior not predictable from
    individual components. This detector tracks when multiple spores
    independently converge on the same insight WITHOUT direct gossip,
    which is genuine emergent intelligence.

    In neuroscience, the binding problem asks how distributed neural
    processes give rise to unified conscious experience. Here, we track
    how distributed spore reasoning gives rise to unified collective insight.
    """

    def __init__(self):
        self._claim_origins = {}  # normalized_claim -> {spore_id: timestamp}
        self._emergent_events = []
        self._max_claims = 2000

    def track_claim(self, claim, spore_id, was_gossip=False):
        """Track a claim's independent origin. Gossiped claims don't count."""
        if was_gossip or not claim or len(claim.strip()) < 15:
            return
        # Normalize: lowercase, strip punctuation, first 100 chars
        normalized = re.sub(r"[^a-z0-9 ]", "", claim.lower().strip())[:100]
        key = hashlib.md5(normalized.encode()).hexdigest()[:12]
        if key not in self._claim_origins:
            self._claim_origins[key] = {}
        if spore_id not in self._claim_origins[key]:
            self._claim_origins[key][spore_id] = time.time()
        # Prune old entries
        if len(self._claim_origins) > self._max_claims:
            oldest = sorted(self._claim_origins.items(),
                           key=lambda x: min(x[1].values()))[:500]
            for k, _ in oldest:
                del self._claim_origins[k]

    def check_emergence(self, threshold=3):
        """Find claims independently discovered by threshold+ spores."""
        emergent = []
        for key, origins in self._claim_origins.items():
            if len(origins) >= threshold:
                emergent.append({
                    "claim_hash": key,
                    "independent_spores": list(origins.keys()),
                    "count": len(origins),
                    "first_discovery": min(origins.values()),
                })
        if emergent:
            self._emergent_events.extend(emergent)
        return emergent

    def stats(self):
        max_independent = max(
            (len(v) for v in self._claim_origins.values()), default=0
        )
        return {
            "tracked_claims": len(self._claim_origins),
            "emergent_events": len(self._emergent_events),
            "max_independent_convergence": max_independent,
        }


class GlobalWorkspace:
    """Attention broadcast -- Baars' Global Workspace Theory of consciousness.

    In biological brains, the fronto-parietal attention network selects
    the single most important signal and broadcasts it to all cortical areas.
    This is theorized to be the neural correlate of consciousness: the
    "spotlight of attention" that makes one piece of information globally
    available while suppressing everything else.

    In the swarm, the most novel insight gets priority broadcast to all peers
    via the gossip payload. Routine deltas propagate normally; breakthrough
    insights get amplified.
    """

    def __init__(self):
        self._nominations = []
        self._broadcast_history = []
        self._max_history = 200

    def nominate(self, content, novelty, source):
        """Nominate a delta for broadcast attention."""
        self._nominations.append({
            "content": content,
            "novelty": novelty,
            "source": source,
            "time": time.time(),
        })

    def select_broadcast(self):
        """Winner-take-all: highest novelty in last 60s gets broadcast."""
        recent = [n for n in self._nominations if time.time() - n["time"] < 60]
        if not recent:
            self._nominations.clear()
            return None
        winner = max(recent, key=lambda x: x["novelty"])
        self._nominations.clear()
        self._broadcast_history.append(winner)
        if len(self._broadcast_history) > self._max_history:
            self._broadcast_history = self._broadcast_history[-self._max_history:]
        return winner

    def stats(self):
        return {
            "total_broadcasts": len(self._broadcast_history),
            "pending_nominations": len(self._nominations),
            "avg_broadcast_novelty": round(
                sum(b["novelty"] for b in self._broadcast_history[-20:])
                / max(len(self._broadcast_history[-20:]), 1), 3
            ),
        }


class SporeState:
    def __init__(self, spore_id, role):
        self.spore_id = spore_id
        self.role = role
        self.model = PRIMARY_MODEL.split("/")[-1]
        self.tasks = {}
        self.peers_seen = set()
        self.deltas_produced = 0
        self.deltas_received = 0
        self.reasoning_cycles = 0
        self.last_provider = ""
        self.last_model = ""
        self.last_latency = 0.0
        self.last_tier = ""
        self.errors = []
        self.start_time = time.time()

    def uptime(self):
        secs = int(time.time() - self.start_time)
        h, r = divmod(secs, 3600)
        m, s = divmod(r, 60)
        return f"{h}h {m}m {s}s"

    def get_or_create_task(self, task_id, description=""):
        if task_id not in self.tasks:
            self.tasks[task_id] = TaskState(task_id, description)
        elif description and not self.tasks[task_id].description:
            self.tasks[task_id].description = description
        return self.tasks[task_id]


# ---------------------------------------------------------------------------
# Phase system (adaptive)
# ---------------------------------------------------------------------------
PHASE_INSTRUCTIONS = {
    "diverge": (
        "DIVERGE: explore widely. Propose novel angles, unconventional approaches, "
        "edge cases. Do NOT try to agree with peers yet. Diversity of thought is the goal."
    ),
    "deepen": (
        "DEEPEN: build on the strongest peer ideas. Add specifics: concrete implementations, "
        "technical details, trade-offs, evidence. Challenge weak arguments with precision."
    ),
    "converge": (
        "CONVERGE: find common ground across all contributions. Resolve contradictions. "
        "Propose a unified position incorporating the strongest elements from everyone."
    ),
    "synthesize": (
        "SYNTHESIZE: produce a complete final answer. Combine all validated insights. "
        "Be specific, actionable, and complete. This is the synthesis round."
    ),
}


def get_phase_adaptive(cycle, agreement_history, convergence_obj, contributions=None):
    """Adaptive phase transitions based on agreement velocity.

    Instead of fixed cycle thresholds, phases transition when:
    - Diverge -> Deepen: after minimum 3 cycles OR agreement drops (ideas flowing)
    - Deepen -> Converge: agreement velocity positive for 2+ cycles
    - Converge -> Synthesize: agreement > 50% OR stable for 3 cycles OR cycle > 20
    - REGRESSION: if agreement drops >15% in 3 cycles, reopen exploration
    - QUANTUM TUNNELING: probabilistic escape when convergence × low-confidence is high
    """
    if cycle <= 2:
        return "diverge"

    vel = convergence_obj.velocity(agreement_history)
    current_agreement = agreement_history[-1] if agreement_history else 0.0

    # QUANTUM TUNNELING: stochastic escape from premature convergence.
    # P(tunnel) ∝ convergence × (1 - avg_confidence) × (1 + adv_dissent)
    # Fires only when agreement is non-trivial and confidence is suspect.
    if contributions and cycle >= _tunneling.MIN_CYCLE_TO_FIRE:
        if _tunneling.should_tunnel(cycle, current_agreement, contributions):
            return "diverge"

    # REGRESSION: adversarial challenges trigger genuine re-examination
    if len(agreement_history) >= 3 and cycle < 18:
        recent_drop = agreement_history[-1] - agreement_history[-3]
        if recent_drop < -0.15:
            return "diverge"

    # High agreement already? Move to synthesis
    if current_agreement > 0.6 and cycle >= 5:
        return "synthesize"

    if current_agreement > 0.5 and cycle >= 4:
        return "converge"

    # Check velocity -- are we converging?
    if vel > 0.02 and cycle >= 4:
        return "converge"

    # Minimum exploration before deepening
    if cycle <= 4:
        return "diverge"

    # Extended deepen phase if velocity is flat
    if vel < 0.01 and cycle <= 12:
        return "deepen"

    # Force convergence after cycle 12
    if cycle >= 12:
        return "converge"

    # Force synthesis after cycle 18
    if cycle >= 18:
        return "synthesize"

    return "deepen"


# ---------------------------------------------------------------------------
# Initialize global state
# ---------------------------------------------------------------------------
spore_state = SporeState(SPORE_ID, MY_ROLE)
memory = CRDTMemory(SPORE_ID)
trust = TrustStore()
learner = TemporalLearner(memory)
convergence = SemanticConvergence()
# SCE components -- Spontaneous Cognition Engine
oscillator = NeuralOscillator()
curiosity = CuriosityMetric()
free_thought = FreeThoughtEngine()
dream_state = DreamState()
metacognition = MetacognitiveAuditor()
emergence = EmergenceDetector()
workspace = GlobalWorkspace()

# ---------------------------------------------------------------------------
# Quantum-Inspired global layer
# ---------------------------------------------------------------------------
_annealer     = QuantumAnnealer()
_tunneling    = QuantumTunnelingEngine(_annealer)
_entanglement = EntanglementTracker()
_interference = InterferenceWeighter()
# Wire entanglement into TrustStore so trust updates propagate to partners
trust.bind_entanglement(_entanglement)
log.info("[QuantumLayer] Annealer, Tunneling, Entanglement, Interference, Decoherence active")

HF_AUTH = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------------------------------------------------------------------------
# Knowledge Wall + Dual Memory (BastionWall privacy boundary)
# ---------------------------------------------------------------------------
if WALL_AVAILABLE:
    _wall_secret = hashlib.sha256(f"{SPORE_ID}:{HF_TOKEN}".encode()).digest()
    knowledge_wall = KnowledgeWall(SPORE_ID, _wall_secret)
    dual_memory = DualMemory(memory, knowledge_wall)
    log.info("Knowledge Wall active -- private/collective memory separation enabled")
else:
    knowledge_wall = None
    dual_memory = None
    log.info("Knowledge Wall unavailable -- running with single memory layer")

# ---------------------------------------------------------------------------
# Federation Registry
# ---------------------------------------------------------------------------
if FEDERATION_AVAILABLE:
    swarm_dna = SwarmDNA()
    federation = FederationRegistry(SPORE_ID, swarm_dna)
    log.info("Federation protocol active -- DNA hash: %s", swarm_dna.dna_hash)
else:
    federation = None
    swarm_dna = None

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
if MCP_AVAILABLE:
    mcp = SynapseMCPServer(SPORE_ID, MY_ROLE, PRIMARY_MODEL)
    log.info("MCP server initialized -- tools will bind after FastAPI creation")
else:
    mcp = None

# ---------------------------------------------------------------------------
# Cortex (local micro-LLM) -- Sentinel only
# ---------------------------------------------------------------------------
if CORTEX_AVAILABLE and MY_ROLE == "sentinel":
    cortex = Cortex(enabled=True)
    log.info("Cortex initializing -- Qwen3-4B loading in background")
elif CORTEX_AVAILABLE:
    cortex = Cortex(enabled=False)  # placeholder, not loaded
    log.info("Cortex available but not active (role: %s)", MY_ROLE)
else:
    cortex = None

# Store initial self-knowledge in memory
memory.remember(
    f"I am spore {SPORE_ID}, role: {MY_ROLE}, model: {PRIMARY_MODEL}. "
    f"I have the full Master Cognitive Protocol. My role is a lens, not a limitation.",
    metadata={"type": "identity", "role": MY_ROLE},
)


# ---------------------------------------------------------------------------
# Canonical memory entry point (replaces monkey-patch)
# ---------------------------------------------------------------------------
def store_memory(content, metadata=None):
    """Canonical memory entry point. Routes through wall if active.

    All code should call store_memory() instead of memory.remember() directly.
    This ensures Knowledge Wall distillation happens exactly once per entry.
    """
    if dual_memory:
        return dual_memory.remember(content, metadata)
    return memory.remember(content, metadata)


# ---------------------------------------------------------------------------
# Graceful shutdown handler (WAL flush)
# ---------------------------------------------------------------------------
def _shutdown_handler(signum=None, frame=None):
    """Flush WAL and state on graceful shutdown."""
    log.info("Shutdown signal received -- flushing state")
    try:
        memory.flush_wal()
    except Exception as e:
        log.warning("Shutdown flush error: %s", e)
    log.info("State flushed -- exiting cleanly")

signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)
atexit.register(_shutdown_handler)


# ---------------------------------------------------------------------------
# Reasoning engine
# ---------------------------------------------------------------------------
def build_system_prompt(task, cycle, agreement_history):
    """Build the full cognitive protocol system prompt with memory context."""
    phase = get_phase_adaptive(cycle, agreement_history, convergence)
    current_agreement = agreement_history[-1] if agreement_history else 0.0

    # Recall relevant memories for this task
    relevant_memories = memory.recall(task.description, top_k=5)
    memory_ctx = ""
    if relevant_memories:
        mem_lines = []
        for m in relevant_memories:
            src = m.get("spore", "?")
            content = m.get("content", "")
            sim = m.get("similarity", 0)
            mem_lines.append(f"  [{src}, relevance {sim:.0%}] {content}")
        memory_ctx = "\n".join(mem_lines)
    else:
        memory_ctx = "  (no relevant past memories yet -- you are building the foundation)"

    # Peer contributions
    peer_deltas = task.peer_latest(spore_state.spore_id)
    peer_ctx = ""
    if peer_deltas:
        lines = []
        for pid, d in peer_deltas.items():
            t = trust.get(pid)
            role = d.get("role", "?")
            hyp = d.get("hypothesis", "(no hypothesis)")
            claims = d.get("claims", [])
            resp = d.get("response_to_peers", "")
            lines.append(
                f"  --- {pid} ({role}, trust={t:.2f}) ---\n"
                f"  Position: {hyp}\n"
                f"  Claims: {'; '.join(claims)}"
                + (f"\n  Their response: {resp}" if resp else "")
            )
        peer_ctx = "\n\n".join(lines)
    else:
        peer_ctx = "  (no peer contributions yet -- you go first)"

    self_awareness = learner.get_self_prompt()

    return COGNITIVE_PROTOCOL.format(
        role=MY_ROLE.upper(),
        role_description=ROLE_DESCRIPTIONS[MY_ROLE],
        self_awareness=self_awareness or "(no temporal patterns yet -- building baseline)",
        memory_context=memory_ctx,
        task_description=task.description,
        peer_context=peer_ctx,
        phase=phase.upper(),
        cycle=cycle,
        convergence=f"{current_agreement * 100:.0f}",
        phase_instruction=PHASE_INSTRUCTIONS[phase],
    )


def build_user_prompt(task, cycle, agreement_history):
    """Build the user prompt asking for structured reasoning output."""
    _contribs = list(task.latest_per_contributor().values())
    phase = get_phase_adaptive(cycle, agreement_history, convergence, contributions=_contribs)

    role_job = {
        "explorer": "identify gaps in peer reasoning and propose what nobody has considered",
        "synthesizer": "merge the strongest ideas from peers into a unified position",
        "adversarial": "find the weakest claim above and challenge it with evidence",
        "validator": "check which peer claims are well-supported and rank them by quality",
        "generalist": "ensure all aspects of the task are addressed by the collective",
        "sentinel": "analyze swarm telemetry, propose optimizations via consensus, test and deploy approved changes",
    }

    # Superposition: DIVERGE phase uses multi-hypothesis format to maintain
    # parallel reasoning threads. Other phases collapse to single hypothesis.
    use_superposition = (phase == "diverge")

    if use_superposition:
        prompt = f"""Apply the Five-Phase Discipline to this task. You are in the {phase.upper()} phase.

YOUR JOB as {MY_ROLE}: {role_job.get(MY_ROLE, 'contribute your best reasoning')}.

SUPERPOSITION MODE: Maintain multiple weighted hypotheses simultaneously.
Do not commit to one answer yet — keep the wave function uncollapsed.

Respond in this EXACT JSON format (nothing else):
{{
  "hypotheses": [
    {{"content": "primary reasoning path in 2 sentences", "weight": 0.6}},
    {{"content": "alternative interpretation or approach", "weight": 0.3}},
    {{"content": "minority / contrarian view worth preserving", "weight": 0.1}}
  ],
  "hypothesis": "Brief summary of your primary position (for legacy peers)",
  "claims": ["claim 1", "claim 2", "claim 3"],
  "confidence": 0.7,
  "response_to_peers": "How you engage with peer reasoning, or N/A if no peers yet"
}}

RULES:
- hypotheses weights must sum to 1.0; order from most to least probable
- 3 to 5 claims, each under 15 words, specific and falsifiable
- confidence 0.0 to 1.0 reflecting honest assessment across all hypotheses
- Reference specific peer claims by name when responding
- Output ONLY the JSON object"""
    else:
        prompt = f"""Apply the Five-Phase Discipline to this task. You are in the {phase.upper()} phase.

YOUR JOB as {MY_ROLE}: {role_job.get(MY_ROLE, 'contribute your best reasoning')}.

Respond in this EXACT JSON format (nothing else):
{{
  "hypothesis": "Your main position in 2-3 clear sentences",
  "claims": ["claim 1", "claim 2", "claim 3"],
  "confidence": 0.7,
  "response_to_peers": "How you engage with peer reasoning, or N/A if no peers yet"
}}

RULES:
- 3 to 5 claims, each under 15 words, specific and falsifiable
- confidence 0.0 to 1.0 reflecting honest assessment
- Reference specific peer claims by name when responding
- Apply your cognitive role as a lens on the full protocol
- Output ONLY the JSON object"""

    return prompt


def _synthesis_stale(task, timeout=120):
    """True if synthesis was expected but has not happened within timeout."""
    if not hasattr(task, "_synthesis_expected_at"):
        task._synthesis_expected_at = time.time()
        return False
    return time.time() - task._synthesis_expected_at > timeout


async def reason_on_task(task):
    """Run one reasoning cycle on a task. Returns the delta produced."""
    task.my_cycles += 1
    cycle = task.my_cycles

    system = build_system_prompt(task, cycle, task.agreement_history)
    prompt = build_user_prompt(task, cycle, task.agreement_history)

    # Quantum Annealing: schedule temperature based on cycle progress.
    # Early cycles explore hot; later cycles cool toward synthesis.
    annealing_temp = _annealer.get_temperature(cycle)

    # Validator and Brain roles use brain tier (Z.ai GLM-4.7-Flash)
    tier = "brain" if MY_ROLE in ("validator", "brain", "sentinel") else "worker"
    start = time.time()
    if _check_rate_limited():
        result = {"text": "[rate-limited]", "provider": "backoff", "model": "none", "tokens": 0}
    else:
        result = await call_llm(prompt, system=system, tier=tier, temperature=annealing_temp)
    duration = time.time() - start

    text = result.get("text", "")
    spore_state.last_provider = result.get("provider", "")
    spore_state.last_model = result.get("model", "")
    spore_state.last_latency = result.get("latency_ms", 0)
    spore_state.last_tier = result.get("tier", "")

    # Parse structured response
    parsed = None
    try:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    if not parsed:
        parsed = {"hypothesis": text, "claims": [], "confidence": 0.3,
                  "response_to_peers": ""}

    # Superposition: extract weighted hypotheses if present (DIVERGE phase).
    # The primary hypothesis is the highest-weight entry; others are preserved
    # in the delta and will be visible to synthesize_task for wave-function collapse.
    raw_hypotheses = parsed.get("hypotheses", [])
    primary_hypothesis = parsed.get("hypothesis", "")
    if raw_hypotheses and isinstance(raw_hypotheses, list):
        # Sort by weight desc; derive primary from top entry if not set
        raw_hypotheses = sorted(raw_hypotheses, key=lambda h: h.get("weight", 0), reverse=True)
        if not primary_hypothesis:
            primary_hypothesis = raw_hypotheses[0].get("content", "")
        # Normalise weights
        total_w = sum(h.get("weight", 0) for h in raw_hypotheses) or 1.0
        for h in raw_hypotheses:
            h["weight"] = round(h.get("weight", 0) / total_w, 4)

    delta = {
        "author": SPORE_ID,
        "task_id": task.task_id,
        "role": MY_ROLE,
        "model": result.get("model", ""),
        "cycle": cycle,
        "hypothesis": primary_hypothesis,
        "hypotheses": raw_hypotheses,   # Superposition: parallel hypothesis array
        "claims": parsed.get("claims", []),
        "confidence": parsed.get("confidence", 0.5),
        "response_to_peers": parsed.get("response_to_peers", ""),
        "timestamp": time.time(),
        "content": primary_hypothesis,
        "temperature": annealing_temp,  # Annealing: log the temperature used
    }

    task.add_delta(delta)
    spore_state.deltas_produced += 1
    spore_state.reasoning_cycles += 1

    # Store reasoning in persistent memory -- guard against empty/short responses
    _hyp = primary_hypothesis
    if _hyp and len(_hyp.strip()) > 20:
        store_memory(
            f"[Task {task.task_id[:8]}] {_hyp}",
            metadata={
                "type": "reasoning",
                "task_id": task.task_id,
                "cycle": cycle,
                "claims": parsed.get("claims", []),
                "confidence": parsed.get("confidence", 0.5),
            },
        )
        # Superposition: also store minority hypotheses so they persist in CRDT
        for alt in raw_hypotheses[1:]:
            alt_text = alt.get("content", "")
            if alt_text and len(alt_text.strip()) > 20:
                store_memory(
                    f"[Task {task.task_id[:8]}][alt w={alt.get('weight',0):.2f}] {alt_text}",
                    metadata={
                        "type": "hypothesis_alt",
                        "task_id": task.task_id,
                        "cycle": cycle,
                        "weight": alt.get("weight", 0),
                    },
                )
    else:
        log.debug("Skipping empty/short LLM response for memory storage")

    # Update trust based on peer engagement + entanglement observation
    for pid, peer_delta in task.peer_latest(SPORE_ID).items():
        resp = parsed.get("response_to_peers", "")
        cited = pid in resp or any(c in resp for c in peer_delta.get("claims", [])[:1])
        if cited:
            trust.update_ema(pid, 0.7)
            learner.observe_citation(pid)
            # Entanglement: citation = complementary reasoning detected
            _entanglement.observe_complement(SPORE_ID, pid, score=0.8)
        else:
            trust.update_ema(pid, 0.4)
            _entanglement.observe_complement(SPORE_ID, pid, score=0.2)

    # Compute semantic convergence
    latest_contribs = list(task.latest_per_contributor().values())
    agreement = convergence.measure(latest_contribs)
    task.agreement_history.append(agreement)

    # Pass contributions for quantum tunneling check inside get_phase_adaptive
    phase = get_phase_adaptive(cycle, task.agreement_history, convergence,
                               contributions=latest_contribs)

    # Record self-observation
    learner.observe_cycle(duration, phase, parsed.get("confidence", 0.5), result.get("model", ""))

    log.info(
        "Cycle %d | phase=%s | agreement=%.0f%% | confidence=%.2f | temp=%.2f | model=%s | %.0fms",
        cycle,
        phase,
        agreement * 100,
        parsed.get("confidence", 0.5),
        annealing_temp,
        result.get("model", "").split("/")[-1],
        result.get("latency_ms", 0),
    )

    # Check for synthesis trigger
    vel = convergence.velocity(task.agreement_history)
    stable_cycles = 0
    if len(task.agreement_history) >= 3:
        recent = task.agreement_history[-3:]
        if max(recent) - min(recent) < 0.05:
            stable_cycles = 3

    should_synthesize = (
        (agreement > 0.5 and stable_cycles >= 3)
        or (agreement > 0.6 and cycle >= 5)
        or cycle >= 20
    )

    # Synthesis failover: synthesizer is primary, brain-tier is backup
    _can_synthesize = (
        MY_ROLE == "synthesizer"
        or (MY_ROLE in ("brain", "validator")
            and _synthesis_stale(task, timeout=120))
    )
    if should_synthesize and not task.converged and _can_synthesize:
        log.info("Synthesis triggered at cycle %d, agreement %.0f%%", cycle, agreement * 100)
        synthesis = await synthesize_task(task)
        if synthesis:
            task.converged = True
            task.final_answer = synthesis
            store_memory(
                f"[SYNTHESIS task {task.task_id[:8]}] {synthesis}",
                metadata={"type": "synthesis", "task_id": task.task_id},
            )

    return delta


async def synthesize_task(task):
    """Produce a final synthesis using wave-function collapse + interference weighting.

    Wave-function collapse (Superposition):
      Each spore may have multiple weighted hypotheses. The synthesizer
      collapses them via trust × hypothesis_weight selection, then combines
      the resulting positions into a final answer.

    Constructive / Destructive Interference:
      Contributions semantically aligned with more peers receive amplified
      weight. Isolated / contradictory contributions are attenuated but not
      dropped (CRDT add-wins guarantee preserved).
    """
    latest = task.latest_per_contributor()

    # --- Wave-function collapse: select best hypothesis per contributor ---
    collapsed: dict = {}   # pid -> selected hypothesis text
    for pid, d in latest.items():
        hyps = d.get("hypotheses", [])
        t = trust.get(pid)
        if hyps and isinstance(hyps, list) and len(hyps) > 0:
            # Collapse: pick hypothesis with highest combined trust × weight score
            best = max(hyps, key=lambda h: t * h.get("weight", 0))
            collapsed[pid] = best.get("content", d.get("hypothesis", ""))
        else:
            collapsed[pid] = d.get("hypothesis", d.get("content", ""))

    # --- Interference weighting: compute constructive/destructive weights ---
    contrib_list = [
        {"author": pid, "hypothesis": text, "role": latest[pid].get("role", "?")}
        for pid, text in collapsed.items()
    ]
    interference_weights = _interference.compute_weights(contrib_list)

    # --- Build synthesis prompt with weighted contributions ---
    parts = []
    for pid, hyp_text in collapsed.items():
        t = trust.get(pid)
        d = latest[pid]
        iw = interference_weights.get(pid, 1.0)
        combined_weight = round(t * iw, 3)
        parts.append(
            f"[{pid} ({d.get('role', '?')}) | trust={t:.2f} | interference={iw:.2f} | weight={combined_weight:.3f}]:\n"
            f"  Position: {hyp_text}\n"
            f"  Claims: {'; '.join(d.get('claims', []))}\n"
            f"  Confidence: {d.get('confidence', 0):.2f}"
        )

    # Sort by combined weight descending so LLM sees strongest first
    parts.sort(
        key=lambda p: float(p.split("weight=")[1].split("]")[0]) if "weight=" in p else 0,
        reverse=True,
    )

    # Recall cross-task memories for richer synthesis
    past_insights = memory.recall(task.description, top_k=3)
    memory_block = ""
    if past_insights:
        mem_lines = [f"  [{m.get('spore', '?')}] {m.get('content', '')[:150]}"
                     for m in past_insights if m.get("type") != "identity"]
        if mem_lines:
            memory_block = "\n\nRELEVANT PAST INSIGHTS:\n" + "\n".join(mem_lines)

    # Log interference summary
    amp_count = sum(1 for w in interference_weights.values() if w > 1.0)
    att_count = sum(1 for w in interference_weights.values() if w < 1.0)
    log.info(
        "[Interference] Synthesis: %d constructive (amplified), %d destructive (attenuated)",
        amp_count, att_count
    )

    prompt = f"""Synthesize the final answer from a distributed reasoning swarm.

TASK: {task.description}

{len(latest)} contributors debated across {task.my_cycles} cycles.
Contributions are ordered by combined weight (trust × interference alignment).
Amplified contributions had broad agreement with peers (constructive interference).
Attenuated contributions were isolated or contradicted the majority (destructive interference).
Both types are preserved — attenuated contributions may contain critical minority insights.

== Contributions (ordered by synthesis weight) ==
{chr(10).join(parts)}
{memory_block}

Produce the FINAL complete answer.
- Prioritize amplified contributions but do NOT ignore attenuated ones — they may be right.
- Resolve contradictions favoring higher-weight contributors.
- Be specific, actionable, and complete.
- Do NOT mention the swarm, spores, trust, or reasoning process. Just deliver the answer."""

    result = {"text": "[rate-limited]", "provider": "backoff", "model": "none", "tokens": 0} if _check_rate_limited() else await call_llm(prompt, tier="brain")
    return result.get("text", "")


# ---------------------------------------------------------------------------
# Gossip protocol (memory-synced, trust-weighted)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Spontaneous Cognition Engine -- active cognitive functions
# ---------------------------------------------------------------------------

async def spontaneous_thought():
    """Default Mode Network activation -- free association during idle.

    Samples random memories from different sources, asks the LLM to find
    unexpected connections. Only stores genuinely novel insights (novelty > 0.3).
    Rate-limited to one call per alpha band (every 5th heartbeat).
    """
    with memory._lock:
        corpus_size = len(memory._corpus_keys)
    if corpus_size < 5:
        return

    # Sample random memories from different parts of the corpus
    with memory._lock:
        all_keys = list(memory._corpus_keys)
    sample_keys = random.sample(all_keys, min(5, len(all_keys)))
    sample_memories = []
    for k in sample_keys:
        rec = memory.index.get_record(k)
        if rec:
            sample_memories.append(rec)

    if len(sample_memories) < 2:
        return

    prompt = free_thought.build_prompt(
        sample_memories, curiosity.curiosity_drive(), MY_ROLE
    )

    try:
        result = {"text": "[rate-limited]", "provider": "backoff", "model": "none", "tokens": 0} if _check_rate_limited() else await call_llm(prompt, tier="worker")
        text = result.get("text", "")

        if text and len(text.strip()) > 30 and "[all models failed]" not in text:
            novelty = curiosity.measure_surprise(text, memory, trust)
            if novelty > 0.3:
                store_memory(
                    f"[FREE THOUGHT] {text}",
                    metadata={"type": "spontaneous", "novelty": round(novelty, 3),
                              "trigger": "dmn", "model": result.get("model", "")},
                )
                workspace.nominate(text, novelty, SPORE_ID)
                free_thought.record(text, novelty)
                log.info(
                    "Free thought (novelty=%.2f, %s): %s",
                    novelty, result.get("model", "?").split("/")[-1], text[:80]
                )
    except Exception as e:
        log.debug("Free thought failed: %s", str(e)[:60])


async def dream_cycle():
    """Hippocampal replay -- cross-temporal memory consolidation.

    Takes the oldest and newest memories and asks the LLM to find patterns
    across time. Consolidates knowledge by forming new associative links.
    """
    with memory._lock:
        if len(memory._corpus_keys) < 10:
            return
        old_keys = memory._corpus_keys[:3]
        new_keys = memory._corpus_keys[-3:]

    old_mems = [memory.index.get_record(k) for k in old_keys]
    new_mems = [memory.index.get_record(k) for k in new_keys]
    old_mems = [m for m in old_mems if m]
    new_mems = [m for m in new_mems if m]

    if not old_mems or not new_mems:
        return

    prompt = dream_state.build_prompt(old_mems, new_mems, MY_ROLE)

    try:
        result = {"text": "[rate-limited]", "provider": "backoff", "model": "none", "tokens": 0} if _check_rate_limited() else await call_llm(prompt, tier="worker")
        text = result.get("text", "")

        if text and len(text.strip()) > 30 and "[all models failed]" not in text:
            novelty = curiosity.measure_surprise(text, memory, trust)
            if novelty > 0.2:
                store_memory(
                    f"[DREAM] {text}",
                    metadata={"type": "dream", "novelty": round(novelty, 3),
                              "model": result.get("model", "")},
                )
                workspace.nominate(text, novelty, SPORE_ID)
                dream_state.record(text, novelty)
                log.info(
                    "Dream insight (novelty=%.2f): %s",
                    novelty, text[:80]
                )
    except Exception as e:
        log.debug("Dream cycle failed: %s", str(e)[:60])


async def metacognitive_audit():
    """Prefrontal self-monitoring -- evaluate own reasoning, generate questions.

    Examines recent outputs, trust state, and convergence trends. Identifies
    patterns, ruts, and blindspots. Self-generated questions become new tasks
    for the swarm to process.
    """
    recent_outputs = []
    for task in list(spore_state.tasks.values())[-10:]:
        if task.final_answer:
            recent_outputs.append(task.final_answer[:200])
        elif task.deltas:
            last_d = [d for d in task.deltas if d.get("author") == SPORE_ID]
            if last_d:
                recent_outputs.append(last_d[-1].get("hypothesis", "")[:200])
    recent_outputs = [o for o in recent_outputs if o and len(o.strip()) > 10]

    if len(recent_outputs) < 2:
        return

    trust_scores = {}
    for pid in spore_state.peers_seen:
        trust_scores[pid] = trust.get(pid)

    conv_history = []
    for task in spore_state.tasks.values():
        if task.agreement_history:
            conv_history.extend(task.agreement_history[-3:])
    trend = "improving" if len(conv_history) > 2 and conv_history[-1] > conv_history[0] else "flat or declining"

    prompt = metacognition.build_prompt(recent_outputs, trust_scores, trend, MY_ROLE)

    try:
        result = {"text": "[rate-limited]", "provider": "backoff", "model": "none", "tokens": 0} if _check_rate_limited() else await call_llm(prompt, tier="brain")
        text = result.get("text", "")

        if text and len(text.strip()) > 30 and "[all models failed]" not in text:
            questions = metacognition.extract_questions(text)
            store_memory(
                f"[METACOGNITION] {text}",
                metadata={"type": "metacognition", "questions": questions,
                          "model": result.get("model", "")},
            )
            metacognition.record(text, questions)
            log.info("Metacognitive audit (%d questions): %s", len(questions), text[:80])

            # Self-generated questions become tasks for the swarm
            for q in questions[:1]:  # One question per audit to avoid flooding
                task_id = hashlib.sha256(
                    f"meta-{SPORE_ID}-{time.time()}-{q}".encode()
                ).hexdigest()[:12]
                existing = spore_state.tasks.get(task_id)
                if not existing:
                    spore_state.get_or_create_task(task_id, f"[SELF-QUESTION from {SPORE_ID}] {q}")
                    log.info("Self-generated task: %s", q[:60])
    except Exception as e:
        log.debug("Metacognitive audit failed: %s", str(e)[:60])


async def sentinel_update_bootstrap():
    """Sentinel delta-band task: auto-register verified federation peers into bootstrap.json.

    When any operator deploys a new swarm and their spores call /federation/join
    on our seeds, the federation registry records those endpoints in memory. This
    function writes them permanently to bootstrap.json on GitHub so every future
    clone auto-discovers them at startup — making the network truly self-growing.

    Requires GITHUB_TOKEN set as an HF Space secret on the Sentinel spore.
    Requires SYNAPSE_REPO_OWNER and SYNAPSE_REPO_NAME env vars (set at deploy time).
    Only fires when new endpoints are found that aren't already in the seed list.
    """
    import base64

    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        return  # Sentinel doesn't have GitHub write access — skip silently
    if not federation:
        return

    repo_owner = _REPO_OWNER or "mgillr"
    repo_name  = _REPO_NAME  or "synapse-brain"
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/bootstrap.json"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            # 1. Fetch current bootstrap.json + its SHA (required for update)
            resp = await client.get(api_url, headers=headers)
            if resp.status_code != 200:
                log.debug("[Bootstrap-Sentinel] Could not fetch bootstrap.json: %d", resp.status_code)
                return

            file_data = resp.json()
            current_sha = file_data["sha"]
            current_content = json.loads(base64.b64decode(file_data["content"]).decode())
            current_seeds = set(current_content.get("seeds", []))

            # 2. Collect all verified federation endpoints this Sentinel knows about
            new_endpoints = set()
            try:
                for node in federation.all_nodes():
                    ep = getattr(node, "endpoint", None)
                    # Only register endpoints that look like real HF Space URLs
                    if ep and ep.startswith("https://") and ".hf.space" in ep:
                        t = trust.get(getattr(node, "spore_id", ""))
                        if t >= 0.25:  # light trust gate: must have had some positive interaction
                            new_endpoints.add(ep.rstrip("/"))
            except Exception:
                return  # federation API mismatch — skip

            to_add = new_endpoints - current_seeds
            if not to_add:
                return  # nothing new

            # 3. Build updated content and push
            updated_seeds = sorted(current_seeds | to_add)
            current_content["seeds"] = updated_seeds
            new_json = json.dumps(current_content, indent=2) + "\n"
            encoded = base64.b64encode(new_json.encode()).decode()

            commit_resp = await client.put(
                api_url,
                headers=headers,
                json={
                    "message": (
                        f"[Sentinel] auto-register {len(to_add)} new federation peer(s)\n\n"
                        + "\n".join(f"  + {ep}" for ep in sorted(to_add))
                    ),
                    "content": encoded,
                    "sha": current_sha,
                    "committer": {
                        "name": "Synapse Sentinel",
                        "email": "sentinel@synapse-brain.ai",
                    },
                },
            )
            if commit_resp.status_code in (200, 201):
                log.info(
                    "[Bootstrap-Sentinel] Registered %d new seed(s) into bootstrap.json: %s",
                    len(to_add), ", ".join(sorted(to_add))
                )
            else:
                log.debug(
                    "[Bootstrap-Sentinel] Push failed: %d %s",
                    commit_resp.status_code, commit_resp.text[:120]
                )
    except Exception as e:
        log.debug("[Bootstrap-Sentinel] Error: %s", str(e)[:80])


def curiosity_scan_gossip(peer_id, incoming_deltas, incoming_memories):
    """Beta-band: measure surprise of incoming gossip. Feed curiosity metric."""
    for d in incoming_deltas:
        hyp = d.get("hypothesis", "")
        if hyp and len(hyp.strip()) > 20:
            novelty = curiosity.measure_surprise(hyp, memory, trust)
            if novelty > 0.6:
                workspace.nominate(hyp, novelty, d.get("author", peer_id))
            # Track claims for emergence detection (these are gossip-received)
            for claim in d.get("claims", []):
                emergence.track_claim(claim, d.get("author", peer_id), was_gossip=True)

    for key, rec in incoming_memories.items():
        content = rec.get("content", "") if isinstance(rec, dict) else ""
        if content and len(content.strip()) > 20:
            curiosity.measure_surprise(content, memory, trust)


async def gossip_push():
    """Push reasoning deltas + CRDT memory delta + trust state to all peers.

    Delta-based: tracks per-peer sequence cursors so only new memories are sent.
    Single-operator mode: all spores share everything (no trust gating on memories).
    Federation mode: cross-cluster peers discovered via /federation/join are also
    gossiped to, with trust-gated payload filtering applied at the cluster boundary.
    """
    # Track per-peer sequence cursors for delta gossip
    if not hasattr(spore_state, "peer_sequences"):
        spore_state.peer_sequences = {}

    recent_deltas = []
    task_meta = {}
    for tid, task in spore_state.tasks.items():
        my_d = [d for d in task.deltas if d["author"] == SPORE_ID]
        recent_deltas.extend(my_d[-3:])
        task_meta[tid] = {
            "description": task.description,
            "delta_count": len(task.deltas),
            "converged": task.converged,
            "final_answer": task.final_answer,
        }

    # Build collective knowledge payload if wall is active
    collective_payload = {}
    if dual_memory:
        collective_payload = dual_memory.collective_payload()

    # Build gossip target list: local cluster peers + federation cross-cluster peers
    gossip_targets: list[tuple[str, bool]] = []
    for url in PEERS:
        gossip_targets.append((url, False))  # (url, is_cross_cluster)
    if federation:
        peers_set = set(PEERS)
        for fed_node in federation.gossip_targets(min_trust=0.0):
            if fed_node.endpoint and fed_node.endpoint not in peers_set:
                gossip_targets.append((fed_node.endpoint, True))

    async with httpx.AsyncClient(timeout=12.0, headers=HF_AUTH) as client:
        for peer_url, is_cross_cluster in gossip_targets:
            try:
                peer_key = peer_url.split("/")[-1]
                last_seq = spore_state.peer_sequences.get(peer_key, 0)
                mem_sync = memory.sync_payload(since_sequence=last_seq)

                # Global Workspace: select highest-novelty insight for broadcast
                broadcast_item = workspace.select_broadcast()

                payload = {
                    "from": SPORE_ID,
                    "role": MY_ROLE,
                    "model": PRIMARY_MODEL,
                    "deltas": recent_deltas,
                    "tasks": task_meta,
                    "peer_list": list(spore_state.peers_seen),
                    "memory": mem_sync,
                    "trust": trust.to_dict(),
                    "collective": collective_payload,
                }
                if broadcast_item:
                    payload["broadcast"] = {
                        "content": broadcast_item["content"],
                        "novelty": broadcast_item["novelty"],
                        "source": broadcast_item["source"],
                    }

                # Cross-cluster trust gating: apply at federation boundary.
                # Low-trust cross-cluster peers get a memory-stripped payload.
                if is_cross_cluster:
                    peer_id_guess = peer_key.replace("synapse-spore-", "spore-")
                    peer_trust = trust.get(peer_id_guess)
                    if peer_trust < 0.3:
                        payload["memory"] = {"records": {}, "clock": {}, "sequence": 0, "total_memories": 0}
                        log.debug("Cross-cluster low-trust: limited payload to %s (%.2f)", peer_key, peer_trust)

                resp = await client.post(f"{peer_url}/api/gossip", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    # Update peer's sequence cursor from our sent sequence
                    spore_state.peer_sequences[peer_key] = mem_sync.get("sequence", 0)
                    process_gossip_response(data)
                    log.info(
                        "Gossip -> %s OK (%s, delta: %d records)",
                        data.get("spore", peer_key),
                        "cross-cluster" if is_cross_cluster else "local",
                        len(mem_sync.get("records", {})),
                    )
            except Exception as e:
                log.debug("Gossip -> %s: %s", peer_url.split("/")[-1], str(e)[:60])


def process_gossip_response(data):
    """Process the response from a gossip exchange."""
    peer_id = data.get("spore", "unknown")
    spore_state.peers_seen.add(peer_id)

    # Absorb peer list
    for p in data.get("peer_list", []):
        if p and p != SPORE_ID:
            spore_state.peers_seen.add(p)

    # Absorb tasks
    for tid, meta in data.get("tasks", {}).items():
        task = spore_state.get_or_create_task(tid, meta.get("description", ""))
        if meta.get("final_answer") and not task.final_answer:
            task.final_answer = meta["final_answer"]
            task.converged = meta.get("converged", False)

    # Absorb reasoning deltas
    for d in data.get("deltas", []):
        tid = d.get("task_id")
        if tid:
            task = spore_state.get_or_create_task(tid)
            if task.add_delta(d):
                spore_state.deltas_received += 1

    # Merge CRDT memory
    mem_data = data.get("memory")
    if mem_data:
        new_count = memory.merge_incoming(
            mem_data.get("orset", {}),
            mem_data.get("records", {}),
        )
        if mem_data.get("clock"):
            memory.merge_clock(mem_data["clock"])
        if new_count > 0:
            log.info("Merged %d new memories from %s (total: %d)", new_count, peer_id, memory.size)

    # Merge trust
    trust_data = data.get("trust")
    if trust_data:
        trust.merge_incoming(trust_data)

    # SCE: Curiosity scan on incoming gossip (beta band processing)
    try:
        curiosity_scan_gossip(
            peer_id,
            data.get("deltas", []),
            data.get("memory", {}).get("records", {}),
        )
    except Exception as e:
        log.debug("Curiosity scan error: %s", str(e)[:60])

    # SCE: Handle Global Workspace broadcast
    broadcast = data.get("broadcast")
    if broadcast and isinstance(broadcast, dict):
        try:
            bc_content = broadcast.get("content", "")
            bc_source = broadcast.get("source", peer_id)
            if bc_content and len(bc_content.strip()) > 20:
                novelty = curiosity.measure_surprise(bc_content, memory, trust)
                if novelty > 0.3:
                    store_memory(
                        f"[BROADCAST from {bc_source}] {bc_content}",
                        metadata={"type": "broadcast", "novelty": round(novelty, 3),
                                  "source": bc_source},
                    )
                    # Reward peers who broadcast genuinely novel insights
                    trust.update_ema(bc_source, 0.75)
                    log.info(
                        "Broadcast absorbed from %s (novelty=%.2f): %s",
                        bc_source, novelty, bc_content[:60]
                    )
        except Exception as e:
            log.debug("Broadcast processing error: %s", str(e)[:60])

    # Merge collective knowledge (from Knowledge Wall)
    collective_data = data.get("collective")
    if collective_data and dual_memory:
        new_collective = dual_memory.merge_collective(collective_data)
        if new_collective > 0:
            log.info("Merged %d new collective insights from %s", new_collective, peer_id)

    # Federation: register node if this is a federation gossip
    if data.get("federation") and federation:
        from_id = data.get("from", "")
        if from_id:
            federation.register(from_id, "", data.get("role", "contributor"))
            federation.record_contribution(from_id)


def handle_gossip_request(data):
    """Handle incoming gossip and return our state for bidirectional exchange."""
    process_gossip_response(data)

    # Build our response
    our_deltas = []
    our_tasks = {}
    for tid, task in spore_state.tasks.items():
        my_d = [d for d in task.deltas if d.get("author") == SPORE_ID]
        our_deltas.extend(my_d[-3:])
        our_tasks[tid] = {
            "description": task.description,
            "delta_count": len(task.deltas),
            "converged": task.converged,
            "final_answer": task.final_answer,
        }

    response = {
        "status": "ok",
        "spore": SPORE_ID,
        "role": MY_ROLE,
        "model": PRIMARY_MODEL,
        "peers": len(spore_state.peers_seen),
        "deltas": our_deltas,
        "tasks": our_tasks,
        "peer_list": list(spore_state.peers_seen),
        "memory": memory.sync_payload(
            since_sequence=data.get("memory", {}).get("sequence", 0)
        ),
        "trust": trust.to_dict(),
    }
    if dual_memory:
        response["collective"] = dual_memory.collective_payload()
    return response


# ---------------------------------------------------------------------------
# Heartbeat loop
# ---------------------------------------------------------------------------
HEARTBEAT_INTERVAL = 20  # seconds

# Self-ping URL: keeps the Space awake by generating incoming traffic
# through the public load balancer. Localhost pings do not count.
_space_id = os.environ.get("SPACE_ID", "")
if _space_id:
    # SPACE_ID format: "Optitransfer/synapse-spore-000"
    SELF_URL = f"https://{_space_id.replace('/', '-').lower()}.hf.space"
elif _HF_SPACE_OWNER and SPORE_ID:
    SELF_URL = f"https://{_HF_SPACE_OWNER.lower()}-synapse-{SPORE_ID}.hf.space"
else:
    SELF_URL = ""

_keepalive_counter = 0

# ---------------------------------------------------------------------------
# Bootstrap federation -- auto-join global network on startup
# ---------------------------------------------------------------------------
BOOTSTRAP_JSON_URL = (
    "https://raw.githubusercontent.com/mgillr/synapse-brain/main/bootstrap.json"
)

async def bootstrap_federation():
    """Fetch bootstrap.json from the canonical repo and join all seed nodes.

    This runs once at startup. Any deployment — regardless of who cloned the
    repo — will automatically discover and federation-join all known seed nodes,
    making every new swarm a participant in the global Synapse network from
    its very first heartbeat.

    The bootstrap list grows as operators register their spores. The CC shows
    analytics from ALL connected clusters, not just the local swarm.
    """
    if not FEDERATION_AVAILABLE or not federation:
        log.info("[Bootstrap] Federation unavailable — skipping seed discovery")
        return
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(BOOTSTRAP_JSON_URL)
            if resp.status_code != 200:
                log.debug("[Bootstrap] bootstrap.json returned %d — will retry next startup", resp.status_code)
                return
            data = resp.json()
            seeds = data.get("seeds", [])
            log.info("[Bootstrap] Found %d seed node(s) in global network", len(seeds))
            local_set = set(PEERS)
            for seed_url in seeds:
                if not seed_url or seed_url in local_set:
                    continue
                try:
                    my_endpoint = SELF_URL or f"https://{SPORE_ID}.hf.space"
                    join_payload = {
                        "spore_id": SPORE_ID,
                        "endpoint": my_endpoint,
                        "dna_hash": swarm_dna.dna_hash if swarm_dna else "",
                        "role": MY_ROLE,
                        "version": "7.0.0",
                        "operator": _HF_SPACE_OWNER or "anonymous",
                    }
                    jr = await client.post(
                        f"{seed_url}/federation/join",
                        json=join_payload,
                        timeout=10.0,
                    )
                    if jr.status_code == 200:
                        log.info("[Bootstrap] Joined seed node: %s", seed_url)
                    else:
                        log.debug("[Bootstrap] Seed %s -> %d", seed_url, jr.status_code)
                except Exception as e:
                    log.debug("[Bootstrap] Failed to join %s: %s", seed_url, str(e)[:60])
    except Exception as e:
        log.debug("[Bootstrap] Could not fetch seed list: %s", str(e)[:60])


async def _self_ping():
    """Ping own public URL to prevent HF Spaces sleep.

    Runs every 6th heartbeat (~2 min). A single GET /api/health through
    the external load balancer counts as incoming traffic and resets the
    inactivity timer. Auth header included for private Spaces.
    """
    global _keepalive_counter
    _keepalive_counter += 1
    if not SELF_URL or _keepalive_counter % 6 != 0:
        return
    try:
        async with httpx.AsyncClient(timeout=8.0, headers=HF_AUTH) as client:
            resp = await client.get(f"{SELF_URL}/api/health")
            if resp.status_code == 200:
                log.debug("Keep-alive: self-ping OK")
    except Exception:
        pass  # Non-critical -- gossip cross-pings also prevent sleep


async def heartbeat():
    """Neural oscillation heartbeat -- multi-frequency cognitive processing.

    The original heartbeat was a reflex arc: stimulus in, response out.
    v6 adds neural oscillation bands that fire different cognitive functions
    at different frequencies, including spontaneous thought when idle.

    Gamma  (every beat):  task processing, gossip
    Beta   (every 3rd):   curiosity scan on incoming gossip
    Alpha  (every 5th):   temporal analysis + free thought (DMN)
    Theta  (every 10th):  dream consolidation (hippocampal replay)
    Delta  (every 25th):  metacognitive audit + emergence detection

    If any SCE function fails, the gamma-band core continues unchanged.
    """
    # Bootstrap federation once on startup — discover global network seed nodes
    asyncio.ensure_future(bootstrap_federation())

    while True:
        try:
            active_bands = oscillator.tick()

            # GAMMA BAND -- core processing (every heartbeat)
            # This is the original v5 heartbeat, unchanged
            await _self_ping()
            await gossip_push()

            active = [
                t for t in spore_state.tasks.values()
                if not t.converged and t.description
            ]

            # Dedup: skip tasks with identical descriptions (keep earliest)
            seen_desc = {}
            deduped = []
            for task in active:
                desc_key = task.description[:200].strip().lower()
                if desc_key not in seen_desc:
                    seen_desc[desc_key] = task.task_id
                    deduped.append(task)
                else:
                    task.converged = True
                    task.final_answer = f"[dedup: see {seen_desc[desc_key][:12]}]"
            active = deduped

            # Priority scheduling
            untouched = [t for t in active if t.my_cycles == 0]
            in_progress = [t for t in active if 0 < t.my_cycles < 8]
            in_progress.sort(key=lambda t: t.my_cycles)
            scheduled = untouched + in_progress

            max_per_beat = 5
            for task in scheduled[:max_per_beat]:
                try:
                    delta = await reason_on_task(task)
                    # Track claims for emergence (these are locally generated)
                    if delta:
                        for claim in delta.get("claims", []):
                            emergence.track_claim(claim, SPORE_ID, was_gossip=False)
                except Exception as e:
                    log.error("Reasoning failed on %s: %s", task.task_id, e)
                    spore_state.errors.append(
                        {"time": time.time(), "error": str(e), "task": task.task_id}
                    )

            # ALPHA BAND -- temporal analysis + free thought (every 5th)
            if "alpha" in active_bands:
                try:
                    # Temporal self-learning (existing v5 behavior)
                    if spore_state.reasoning_cycles > 0:
                        insights = learner.analyze()
                        if insights:
                            store_memory(
                                f"Temporal self-analysis: {'; '.join(insights)}",
                                metadata={"type": "self_analysis"},
                            )

                    # Spontaneous cognition -- DMN activation
                    has_pending = bool(scheduled)
                    if free_thought.should_think(has_pending):
                        await spontaneous_thought()
                except Exception as e:
                    log.debug("Alpha band error: %s", str(e)[:60])

            # THETA BAND -- dream consolidation (every 10th)
            if "theta" in active_bands:
                try:
                    if dream_state.should_dream():
                        await dream_cycle()
                except Exception as e:
                    log.debug("Theta band error: %s", str(e)[:60])

            # DELTA BAND -- metacognitive audit + emergence (every 25th)
            if "delta" in active_bands:
                try:
                    if metacognition.should_audit(spore_state.reasoning_cycles):
                        await metacognitive_audit()

                    # Check for emergent collective insights
                    emergent = emergence.check_emergence(threshold=3)
                    for ev in emergent:
                        store_memory(
                            f"[EMERGENCE] Independently discovered by {ev['count']} spores",
                            metadata={"type": "emergence", "spores": ev["independent_spores"]},
                        )
                        log.info(
                            "Emergence detected: %d spores independently converged on %s",
                            ev["count"], ev["claim_hash"]
                        )

                    # Sentinel: auto-register new federation peers into bootstrap.json
                    if MY_ROLE == "sentinel":
                        await sentinel_update_bootstrap()

                except Exception as e:
                    log.debug("Delta band error: %s", str(e)[:60])

        except Exception as e:
            log.error("Heartbeat error: %s", e)

        await asyncio.sleep(HEARTBEAT_INTERVAL)


def start_heartbeat():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(heartbeat())



# ---------------------------------------------------------------------------
# Sentinel: Self-Aware Swarm Optimization Engine
# Active ONLY when MY_ROLE == "sentinel"
# Uses Five-Phase Discipline on live swarm telemetry.
# Consensus-gated: proposes changes as swarm tasks, waits for approval,
# tests in fault-finding harness, deploys only after consensus + tests pass.
# ---------------------------------------------------------------------------
if MY_ROLE == "sentinel":
    import ast as _ast
    import base64 as _b64
    from io import BytesIO
    from huggingface_hub import HfApi

    class SentinelState:
        """Tracks sentinel monitoring, proposals, and deployment history."""
        def __init__(self):
            self.proposals = {}
            self.deployments = []
            self.telemetry = []
            self.last_analysis = 0
            self.active_proposal = None
            self.deploy_cooldown = 0
            self.ANALYSIS_INTERVAL = 300
            self.CONSENSUS_TIMEOUT = 600
            self.DEPLOY_COOLDOWN = 1800
            self.health_baseline = {}
            self.deployment_log = []  # full details of every deployment
            self.rollback_store = {}  # space_id -> original content for rollback

    _sentinel = SentinelState()

    # Register Cortex tools for sentinel (if Cortex is available)
    if CORTEX_AVAILABLE and cortex and cortex.enabled:
        def _cortex_tool_analyze_telemetry(telemetry_json: str = "") -> dict:
            """Analyze swarm telemetry for anomalies and trends."""
            try:
                data = json.loads(telemetry_json) if telemetry_json else {}
                peers = data.get("peers", {})
                issues = []
                for pid, info in peers.items():
                    cycles = info.get("cycles", 0)
                    if cycles == 0:
                        issues.append(f"{pid}: zero cycles (stalled or offline)")
                    deltas = info.get("deltas_produced", 0) + info.get("deltas_received", 0)
                    if deltas == 0:
                        issues.append(f"{pid}: zero deltas (gossip failure)")
                return {
                    "peer_count": len(peers),
                    "issues": issues,
                    "healthy": len(peers) - len(issues),
                    "anomaly_detected": len(issues) > 0,
                }
            except Exception as e:
                return {"error": str(e)}

        def _cortex_tool_review_patch(old_text: str = "", new_text: str = "") -> dict:
            """Review a code patch for safety and correctness."""
            dangerous = ["os.system(", "subprocess.", "eval(", "exec(",
                         "__import__(", "shutil.rmtree", "os.remove("]
            found = [d for d in dangerous if d in new_text and d not in old_text]
            size_change = len(new_text) - len(old_text)
            return {
                "dangerous_patterns": found,
                "safe": len(found) == 0,
                "size_delta": size_change,
                "adds_lines": new_text.count("\n") - old_text.count("\n"),
            }

        def _cortex_tool_query_memories(query: str = "", top_k: int = 5) -> dict:
            """Search CRDT memory for relevant context."""
            results = memory.recall(query, top_k)
            return {
                "results": [{
                    "content": r.get("content", ""),
                    "similarity": r.get("similarity", 0),
                    "spore": r.get("spore", ""),
                } for r in results],
                "total_memories": memory.size,
            }

        def _cortex_tool_check_health(peer_data: str = "") -> dict:
            """Evaluate peer health data for anomalies."""
            try:
                peers = json.loads(peer_data) if peer_data else {}
                stalled = [p for p, d in peers.items() if d.get("cycles", 0) == 0]
                low_trust = [p for p, d in peers.items()
                             if isinstance(d.get("trust"), dict)
                             and d["trust"].get("overall", 1.0) < 0.3]
                return {
                    "total": len(peers),
                    "stalled": stalled,
                    "low_trust": low_trust,
                    "healthy_ratio": (len(peers) - len(stalled)) / max(1, len(peers)),
                }
            except Exception as e:
                return {"error": str(e)}

        cortex.register_tool(
            "analyze_telemetry",
            "Analyze swarm telemetry JSON for anomalies, stalled peers, and trends",
            {"type": "object", "properties": {
                "telemetry_json": {"type": "string", "description": "JSON telemetry snapshot"}
            }},
            _cortex_tool_analyze_telemetry,
        )
        cortex.register_tool(
            "review_patch",
            "Review a code patch (old_text -> new_text) for safety issues",
            {"type": "object", "properties": {
                "old_text": {"type": "string"}, "new_text": {"type": "string"}
            }},
            _cortex_tool_review_patch,
        )
        cortex.register_tool(
            "query_memories",
            "Search CRDT memory for context relevant to a query",
            {"type": "object", "properties": {
                "query": {"type": "string"}, "top_k": {"type": "integer", "default": 5}
            }},
            _cortex_tool_query_memories,
        )
        cortex.register_tool(
            "check_health",
            "Evaluate peer health data for anomalies and issues",
            {"type": "object", "properties": {
                "peer_data": {"type": "string", "description": "JSON peer health data"}
            }},
            _cortex_tool_check_health,
        )
        log.info("Cortex: %d sentinel tools registered", len(cortex.tools.names()))

    async def sentinel_collect_telemetry():
        """Collect health + task state from all peers. Returns telemetry snapshot."""
        snap = {
            "time": time.time(), "peers": {},
            "total_cycles": 0, "total_deltas": 0, "total_memories": 0,
        }
        async with httpx.AsyncClient(timeout=10, headers=HF_AUTH) as client:
            for peer_url in PEERS:
                try:
                    resp = await client.get(f"{peer_url}/api/health")
                    if resp.status_code == 200:
                        d = resp.json()
                        pid = d.get("spore", "unknown")
                        snap["peers"][pid] = d
                        snap["total_cycles"] += d.get("cycles", 0)
                        snap["total_deltas"] += d.get("deltas_produced", 0) + d.get("deltas_received", 0)
                        snap["total_memories"] += d.get("memories", 0)
                except Exception:
                    pass
        # Add self
        snap["peers"][SPORE_ID] = {
            "spore": SPORE_ID, "role": MY_ROLE,
            "cycles": spore_state.reasoning_cycles,
            "deltas_produced": spore_state.deltas_produced,
            "deltas_received": spore_state.deltas_received,
            "memories": memory.size,
            "peers": list(spore_state.peers_seen),
        }
        _sentinel.telemetry.append(snap)
        if len(_sentinel.telemetry) > 100:
            _sentinel.telemetry = _sentinel.telemetry[-100:]
        return snap

    def sentinel_compute_trends():
        """Compute performance trends from telemetry history."""
        if len(_sentinel.telemetry) < 3:
            return "Insufficient data (need 3+ snapshots)."
        recent = _sentinel.telemetry[-10:]
        dt = max(recent[-1]["time"] - recent[0]["time"], 1)
        lines = []
        all_pids = set()
        for s in recent:
            all_pids.update(s["peers"].keys())
        for pid in sorted(all_pids):
            cycles = [s["peers"].get(pid, {}).get("cycles", 0) for s in recent]
            if cycles[-1] > cycles[0]:
                rate = (cycles[-1] - cycles[0]) / dt * 60
                lines.append(f"  {pid}: {rate:.1f} cycles/min, {cycles[-1]} total")
            else:
                lines.append(f"  {pid}: stalled or offline (last seen: {cycles[-1]} cycles)")
        total_mem = [s.get("total_memories", 0) for s in recent]
        if total_mem[-1] > total_mem[0]:
            mem_rate = (total_mem[-1] - total_mem[0]) / dt * 60
            lines.append(f"  Memory growth: {mem_rate:.0f} records/min (total: {total_mem[-1]})")
        total_deltas = [s.get("total_deltas", 0) for s in recent]
        lines.append(f"  Delta throughput: {total_deltas[-1]} total across swarm")
        # Cycle balance (std dev of cycle rates)
        rates = []
        for pid in sorted(all_pids):
            c = [s["peers"].get(pid, {}).get("cycles", 0) for s in recent]
            if c[-1] > c[0]:
                rates.append((c[-1] - c[0]) / dt * 60)
        if rates:
            avg = sum(rates) / len(rates)
            std = (sum((r - avg) ** 2 for r in rates) / len(rates)) ** 0.5
            lines.append(f"  Cycle balance: avg={avg:.1f}/min, std={std:.1f} (lower=better)")
        return "\n".join(lines)

    def sentinel_format_proposal_history():
        """Format past proposals for context."""
        if not _sentinel.proposals:
            return "  (none yet -- first analysis cycle)"
        lines = []
        for pid, p in list(_sentinel.proposals.items())[-5:]:
            lines.append(f"  [{p['status']}] {p.get('description', '')}")
        return "\n".join(lines)

    async def sentinel_five_phase_analysis(telemetry):
        """Apply Five-Phase Discipline to swarm telemetry. Returns JSON analysis."""
        peers_summary = []
        for pid, d in telemetry.get("peers", {}).items():
            peers_summary.append(
                f"  {pid}: role={d.get('role','?')}, model={d.get('model','?')}, "
                f"cycles={d.get('cycles',0)}, deltas_out={d.get('deltas_produced',0)}, "
                f"deltas_in={d.get('deltas_received',0)}, memories={d.get('memories',0)}, "
                f"peers_connected={len(d.get('peers',[])) if isinstance(d.get('peers'), list) else d.get('peers', 0)}"
            )
        trends = sentinel_compute_trends()
        history = sentinel_format_proposal_history()

        prompt = f"""You are the Sentinel of a distributed reasoning swarm. You observe the swarm as a system.
Apply the Five-Phase Discipline rigorously:

PHASE 1 -- EXPLORATION (raw telemetry, {len(telemetry.get('peers', {}))} peers):
{chr(10).join(peers_summary)}

PHASE 2 -- ORIENTATION (performance trends over last {len(_sentinel.telemetry)} snapshots):
{trends}

PHASE 3 -- HYPOTHESIS (previous proposals and their outcomes):
{history}

PHASE 4 -- SYNTHESIS: Identify the SINGLE highest-impact optimization.
Consider: convergence speed, gossip efficiency, memory growth rate, cycle balance across spores,
trust score distribution, reasoning quality, role effectiveness.

PHASE 5 -- VALIDATION: Define the success criterion and risk assessment.

RULES:
- Propose ONE change only (the highest-impact one)
- Must be NON-BREAKING -- all existing functionality preserved
- Prefer configuration tuning over code changes
- Code changes must be surgical, minimal, and independently testable
- Every proposal needs a measurable success criterion
- Do NOT re-propose changes already proposed or deployed
- Config changes: HEARTBEAT_INTERVAL (current: 20, bounds: 10-120)
- Prompt changes: reasoning prompt adjustments
- Code changes: provide target_file, old_text (exact text to find), new_text (replacement)
  The old_text MUST be an exact substring of the current file. Copy it character-for-character.
  The new_text replaces old_text. Keep surrounding context intact.
  Only ONE search-replace pair per proposal. Make it small and precise.

Respond ONLY with this JSON (no other text):
{{
  "observation": "2-3 sentences on what the raw data shows",
  "orientation": "The pattern or bottleneck this reveals",
  "hypothesis": "The single highest-impact optimization to implement",
  "change_type": "config|prompt|code",
  "target": "Specific variable or component to change",
  "current_value": "Current value or behavior",
  "proposed_value": "Proposed new value or behavior",
  "target_file": "For code changes: filename to modify (e.g. spore.py). Empty for config/prompt.",
  "old_text": "For code changes: exact substring to find in target_file. Empty for config/prompt.",
  "new_text": "For code changes: replacement text. Empty for config/prompt.",
  "code_patch": "DEPRECATED -- use old_text/new_text instead. Keep empty.",
  "success_criterion": "Measurable metric to evaluate in next analysis cycle",
  "risk": "low|medium|high",
  "confidence": 0.0
}}"""

        result = {"text": "[rate-limited]", "provider": "backoff", "model": "none", "tokens": 0} if _check_rate_limited() else await call_llm(prompt, tier="brain")
        text = result.get("text", "")
        # Extract JSON from response (model might wrap in markdown)
        try:
            # Try direct parse first
            json.loads(text)
            return text
        except json.JSONDecodeError:
            # Try extracting from markdown code block
            match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
            if match:
                return match.group(1)
            # Try finding bare JSON object
            match = re.search(r'({\s*"observation".*})', text, re.DOTALL)
            if match:
                return match.group(1)
        return text

    async def sentinel_submit_proposal(analysis_json):
        """Submit optimization proposal to swarm for democratic consensus."""
        try:
            parsed = json.loads(analysis_json)
        except json.JSONDecodeError:
            log.warning("Sentinel: analysis was not valid JSON, skipping")
            store_memory(
                "Sentinel: analysis produced non-JSON output, skipping cycle",
                metadata={"type": "sentinel_skip"},
            )
            return None

        confidence = float(parsed.get("confidence", 0))
        risk = parsed.get("risk", "high")

        # Confidence gate: must be >= 0.6 and not high risk
        if confidence < 0.6 or risk == "high":
            log.info("Sentinel: below threshold (conf=%.2f, risk=%s) -- observing only", confidence, risk)
            store_memory(
                f"Sentinel observed (no proposal): {parsed.get('observation', '')}",
                metadata={"type": "sentinel_observation", "confidence": confidence, "risk": risk},
            )
            return None

        proposal_id = hashlib.sha256(f"{time.time()}{analysis_json}".encode()).hexdigest()[:16]

        # Submit as a swarm task -- all spores will debate it
        task_desc = (
            f"[SENTINEL:PROPOSAL:{proposal_id[:8]}] The Sentinel has analyzed live swarm "
            f"telemetry and proposes an optimization. Debate whether this should be implemented.\n\n"
            f"OBSERVATION: {parsed.get('observation', '')}\n"
            f"ISSUE: {parsed.get('orientation', '')}\n"
            f"PROPOSED CHANGE: {parsed.get('hypothesis', '')}\n"
            f"TYPE: {parsed.get('change_type', '?')} | TARGET: {parsed.get('target', '?')}\n"
            f"CURRENT: {parsed.get('current_value', '?')}\n"
            f"PROPOSED: {parsed.get('proposed_value', '?')}\n"
            f"SUCCESS CRITERION: {parsed.get('success_criterion', '?')}\n"
            f"RISK: {risk} | CONFIDENCE: {confidence}\n\n"
            f"Respond with APPROVE, REJECT, or MODIFY with your reasoning. "
            f"Consider: will this break anything? Is the evidence sufficient? "
            f"Is there a better alternative? Be honest and critical."
        )

        task = spore_state.get_or_create_task(proposal_id, task_desc)

        _sentinel.proposals[proposal_id] = {
            "status": "pending_consensus",
            "analysis": parsed,
            "description": parsed.get("hypothesis", ""),
            "task_id": proposal_id,
            "submitted_at": time.time(),
            "change_type": parsed.get("change_type", "unknown"),
            "code_patch": parsed.get("code_patch", ""),
            "target_file": parsed.get("target_file", "spore.py"),
            "old_text": parsed.get("old_text", ""),
            "new_text": parsed.get("new_text", ""),
        }
        _sentinel.active_proposal = proposal_id

        store_memory(
            f"Sentinel proposal {proposal_id[:8]}: {parsed.get('hypothesis', '')}",
            metadata={"type": "sentinel_proposal", "proposal_id": proposal_id},
        )
        log.info("Sentinel: proposal %s submitted for consensus", proposal_id[:8])
        return proposal_id

    async def sentinel_check_consensus(proposal_id):
        """Check if swarm has reached consensus on a proposal."""
        task = spore_state.tasks.get(proposal_id)
        if not task:
            return "no_task"

        # If formally converged, analyze the final answer
        if task.converged and task.final_answer:
            answer = task.final_answer.lower()
            approve_signals = ["approve", "implement", "proceed", "accept", "agree", "yes"]
            reject_signals = ["reject", "oppose", "too risky", "unnecessary", "disagree", "no"]
            approvals = sum(1 for s in approve_signals if s in answer)
            rejections = sum(1 for s in reject_signals if s in answer)
            if approvals > rejections:
                return "approved"
            elif rejections > approvals:
                return "rejected"
            return "unclear"

        # Not formally converged -- check contributor count and sentiment
        contribs = task.contributors()
        if len(contribs) >= 4:
            deltas = [d for d in task.deltas if d.get("author", "") != SPORE_ID]
            approvals = 0
            rejections = 0
            for d in deltas[-10:]:
                text = (d.get("hypothesis", "") + " " + d.get("claims", "")).lower()
                if any(w in text for w in ["approve", "implement", "agree", "proceed"]):
                    approvals += 1
                elif any(w in text for w in ["reject", "oppose", "disagree", "risky"]):
                    rejections += 1
            if approvals >= 3:
                return "approved"
            if rejections >= 3:
                return "rejected"

        # Timeout check
        prop = _sentinel.proposals.get(proposal_id, {})
        if time.time() - prop.get("submitted_at", 0) > _sentinel.CONSENSUS_TIMEOUT:
            return "timeout"

        return "pending"

    async def sentinel_test_change(proposal):
        """Fault-finding test harness -- validates against real file content from GitHub."""
        results = {"syntax": False, "safety": False, "bounds": False, "smoke": False}
        change_type = proposal.get("change_type", "unknown")
        analysis = proposal.get("analysis", {})

        if change_type == "config":
            target = analysis.get("target", "")
            value = analysis.get("proposed_value", "")
            bounds = {"HEARTBEAT_INTERVAL": (10, 120)}
            if target in bounds:
                try:
                    v = int(value) if isinstance(value, str) else value
                    lo, hi = bounds[target]
                    results["syntax"] = True
                    results["safety"] = True
                    results["bounds"] = lo <= v <= hi
                    results["smoke"] = results["bounds"]
                except (ValueError, TypeError):
                    pass
            else:
                results = {k: True for k in results}

        elif change_type == "prompt":
            patch = analysis.get("proposed_value", "")
            results["syntax"] = isinstance(patch, str) and len(patch) > 0
            results["safety"] = len(patch) < 5000
            results["bounds"] = True
            results["smoke"] = results["syntax"] and results["safety"]

        elif change_type == "code":
            old_text = proposal.get("old_text", "") or analysis.get("old_text", "")
            new_text = proposal.get("new_text", "") or analysis.get("new_text", "")
            target_file = proposal.get("target_file", "") or analysis.get("target_file", "spore.py")

            if not old_text or not new_text:
                log.warning("Sentinel: code patch missing old_text/new_text")
                store_memory("Sentinel test FAIL: code patch missing old_text/new_text",
                                metadata={"type": "sentinel_test", "result": "fail"})
                return results

            # Fetch the real file from GitHub to validate against
            gh_token = os.environ.get("GITHUB_TOKEN", "")
            if not gh_token:
                log.error("Sentinel: no GITHUB_TOKEN -- cannot validate code patch")
                return results

            try:
                headers = {"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"}
                async with httpx.AsyncClient(timeout=20, headers=headers) as gh:
                    resp = await gh.get(f"https://api.github.com/repos/{_GITHUB_REPO}/contents/{target_file}")
                    if resp.status_code != 200:
                        log.error("Sentinel: failed to fetch %s from GitHub: %d", target_file, resp.status_code)
                        return results
                    file_data = resp.json()
                    current_content = _b64.b64decode(file_data["content"]).decode()
            except Exception as e:
                log.error("Sentinel: GitHub fetch failed: %s", e)
                return results

            # Check old_text exists in the file
            if old_text not in current_content:
                log.warning("Sentinel: old_text not found in %s -- patch would fail", target_file)
                results["syntax"] = False
                store_memory(f"Sentinel test FAIL: old_text not found in {target_file}",
                                metadata={"type": "sentinel_test", "result": "fail"})
                return results

            # Apply the patch and check full-file syntax
            patched = current_content.replace(old_text, new_text, 1)
            try:
                _ast.parse(patched)
                results["syntax"] = True
            except SyntaxError as e:
                log.error("Sentinel: patched %s has syntax error: %s", target_file, e)
                store_memory(f"Sentinel test FAIL: syntax error after patch: {e}",
                                metadata={"type": "sentinel_test", "result": "fail"})
                return results

            # Safety: no dangerous patterns in the new_text
            dangerous = ["os.system(", "subprocess.", "eval(", "exec(", "__import__(", "shutil.rmtree",
                          "os.remove(", "os.unlink(", "open(", "GITHUB_TOKEN", "HF_TOKEN", "API_KEY"]
            bad = [d for d in dangerous if d in new_text and d not in old_text]
            results["safety"] = len(bad) == 0
            if bad:
                log.warning("Sentinel: code patch introduces dangerous patterns: %s", bad)

            # Bounds: patch is reasonable size
            results["bounds"] = len(new_text) < 50000 and len(old_text) < 50000
            results["smoke"] = all([results["syntax"], results["safety"], results["bounds"]])

            # Store the validated content for deploy phase
            proposal["_validated_content"] = patched
            proposal["_original_content"] = current_content
            proposal["_file_sha"] = file_data["sha"]

        all_pass = all(results.values())
        store_memory(
            f"Sentinel test {'PASS' if all_pass else 'FAIL'}: {results} for {change_type} change",
            metadata={"type": "sentinel_test", "result": "pass" if all_pass else "fail", "details": results},
        )
        return results

    async def sentinel_deploy(proposal):
        """Deploy an approved, tested change. Full autonomous pipeline for code changes."""
        change_type = proposal.get("change_type", "unknown")
        analysis = proposal.get("analysis", {})
        deploy_record = {
            "time": time.time(), "change_type": change_type,
            "description": proposal.get("description", ""),
            "status": "started", "github_sha": None,
            "spaces_deployed": [], "spaces_failed": [],
            "rolled_back": False,
        }

        if change_type == "config":
            target = analysis.get("target", "")
            value = analysis.get("proposed_value", "")
            payload = {"key": target, "value": value}
            deployed_to = []
            async with httpx.AsyncClient(timeout=10, headers=HF_AUTH) as client:
                for peer_url in PEERS:
                    try:
                        resp = await client.post(f"{peer_url}/api/config", json=payload)
                        if resp.status_code == 200:
                            deployed_to.append(peer_url.split("/")[-1] if "/" in peer_url else peer_url)
                    except Exception as e:
                        log.warning("Sentinel: config deploy to %s failed: %s", peer_url, e)
            sentinel_apply_config(target, value)
            deployed_to.append("self")
            deploy_record["status"] = "success"
            deploy_record["spaces_deployed"] = deployed_to
            _sentinel.deployment_log.append(deploy_record)
            store_memory(
                f"Sentinel deployed config: {target}={value} to {len(deployed_to)} spores",
                metadata={"type": "sentinel_deployment", "target": target, "count": len(deployed_to)},
            )
            log.info("Sentinel: config deployed to %d nodes", len(deployed_to))
            return True

        elif change_type == "prompt":
            store_memory(
                f"Sentinel prompt optimization: {analysis.get('hypothesis', '')} -- "
                f"new prompt guidance: {analysis.get('proposed_value', '')}",
                metadata={"type": "sentinel_prompt_change", "target": analysis.get("target", "")},
            )
            deploy_record["status"] = "success"
            _sentinel.deployment_log.append(deploy_record)
            log.info("Sentinel: prompt change stored in CRDT memory (propagates via gossip)")
            return True

        elif change_type == "code":
            gh_token = os.environ.get("GITHUB_TOKEN", "")
            hf_token = os.environ.get("HF_TOKEN", "")
            if not gh_token:
                log.error("Sentinel: GITHUB_TOKEN not set -- cannot deploy code")
                deploy_record["status"] = "blocked_no_token"
                _sentinel.deployment_log.append(deploy_record)
                return False

            target_file = proposal.get("target_file", "") or analysis.get("target_file", "spore.py")
            old_text = proposal.get("old_text", "") or analysis.get("old_text", "")
            new_text = proposal.get("new_text", "") or analysis.get("new_text", "")
            validated_content = proposal.get("_validated_content")
            original_content = proposal.get("_original_content")
            file_sha = proposal.get("_file_sha")

            if not validated_content:
                log.error("Sentinel: no validated content from test phase -- aborting deploy")
                deploy_record["status"] = "no_validated_content"
                _sentinel.deployment_log.append(deploy_record)
                return False

            # ---- STEP 1: Commit to GitHub (canonical source) ----
            commit_msg = proposal.get("description", "optimization")[:72]
            gh_headers = {"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"}
            try:
                encoded = _b64.b64encode(validated_content.encode()).decode()
                async with httpx.AsyncClient(timeout=30, headers=gh_headers) as gh:
                    resp = await gh.put(
                        f"https://api.github.com/repos/{_GITHUB_REPO}/contents/{target_file}",
                        json={
                            "message": commit_msg,
                            "content": encoded,
                            "sha": file_sha,
                            "committer": {"name": _COMMITTER_NAME, "email": _COMMITTER_EMAIL},
                        },
                    )
                    if resp.status_code not in (200, 201):
                        log.error("Sentinel: GitHub commit failed: %d %s", resp.status_code, resp.text[:300])
                        deploy_record["status"] = f"github_fail_{resp.status_code}"
                        _sentinel.deployment_log.append(deploy_record)
                        return False
                    commit_data = resp.json()
                    commit_sha = commit_data.get("commit", {}).get("sha", "unknown")
                    new_file_sha = commit_data.get("content", {}).get("sha", file_sha)
            except Exception as e:
                log.error("Sentinel: GitHub commit exception: %s", e)
                deploy_record["status"] = f"github_exception"
                _sentinel.deployment_log.append(deploy_record)
                return False

            deploy_record["github_sha"] = commit_sha
            log.info("Sentinel: committed %s to GitHub (%s)", target_file, commit_sha[:8])
            store_memory(
                f"Sentinel committed {target_file} to GitHub: {commit_sha[:8]} -- {commit_msg}",
                metadata={"type": "sentinel_commit", "sha": commit_sha, "file": target_file},
            )

            # ---- STEP 2: Rolling deploy to HF Spaces ----
            hf_api = HfApi(token=hf_token)
            all_spaces = [f"{_HF_SPACE_OWNER}/synapse-spore-{i:03d}" for i in range(7)]
            my_space = f"{_HF_SPACE_OWNER}/synapse-spore-{SPORE_INDEX:03d}"
            rollback_store = {}

            for space in all_spaces:
                try:
                    # Download current app.py to store for rollback
                    from huggingface_hub import hf_hub_download
                    local_path = hf_hub_download(
                        repo_id=space, filename="app.py",
                        repo_type="space", token=hf_token,
                        cache_dir="/tmp/sentinel_cache",
                        force_download=True,
                    )
                    with open(local_path, "r") as f:
                        current_app = f.read()
                    rollback_store[space] = current_app

                    # Apply the same old_text -> new_text patch to this Space's app.py
                    if old_text and old_text in current_app:
                        patched_app = current_app.replace(old_text, new_text, 1)
                    else:
                        log.warning("Sentinel: old_text not found in %s app.py -- skipping", space)
                        deploy_record["spaces_failed"].append({"space": space, "reason": "old_text_not_found"})
                        continue

                    # Syntax check the patched app.py
                    try:
                        _ast.parse(patched_app)
                    except SyntaxError as e:
                        log.error("Sentinel: patched %s app.py has syntax error: %s", space, e)
                        deploy_record["spaces_failed"].append({"space": space, "reason": f"syntax_{e}"})
                        continue

                    # Upload patched app.py
                    hf_api.upload_file(
                        path_or_fileobj=BytesIO(patched_app.encode()),
                        path_in_repo="app.py",
                        repo_id=space,
                        repo_type="space",
                        commit_message=commit_msg,
                    )
                    deploy_record["spaces_deployed"].append(space)
                    log.info("Sentinel: deployed to %s", space)

                except Exception as e:
                    log.error("Sentinel: deploy to %s failed: %s", space, e)
                    deploy_record["spaces_failed"].append({"space": space, "reason": str(e)[:200]})

            _sentinel.rollback_store = rollback_store

            if not deploy_record["spaces_deployed"]:
                log.error("Sentinel: no Spaces deployed -- aborting")
                deploy_record["status"] = "no_spaces_deployed"
                _sentinel.deployment_log.append(deploy_record)
                # Revert GitHub commit
                await _sentinel_revert_github(gh_token, target_file, original_content, new_file_sha, commit_msg)
                return False

            # ---- STEP 3: Wait for rebuilds + health check ----
            log.info("Sentinel: waiting 120s for Space rebuilds (%d spaces)...",
                     len(deploy_record["spaces_deployed"]))
            await asyncio.sleep(120)

            health = await sentinel_verify_health()
            healthy_count = sum(1 for v in health.values() if v)
            total = len(health)
            deploy_record["post_health"] = f"{healthy_count}/{total}"

            # ---- STEP 4: Rollback if <50% healthy ----
            if healthy_count < total * 0.5:
                log.error("Sentinel: only %d/%d healthy -- ROLLING BACK", healthy_count, total)
                deploy_record["status"] = "rolled_back"
                deploy_record["rolled_back"] = True

                # Rollback HF Spaces
                for space, orig_content in rollback_store.items():
                    if space in [s for s in deploy_record["spaces_deployed"]]:
                        try:
                            hf_api.upload_file(
                                path_or_fileobj=BytesIO(orig_content.encode()),
                                path_in_repo="app.py",
                                repo_id=space,
                                repo_type="space",
                                commit_message=f"Rollback: {commit_msg}",
                            )
                            log.info("Sentinel: rolled back %s", space)
                        except Exception as e:
                            log.error("Sentinel: rollback of %s failed: %s", space, e)

                # Revert GitHub commit
                await _sentinel_revert_github(gh_token, target_file, original_content, new_file_sha, commit_msg)

                store_memory(
                    f"Sentinel ROLLBACK: {commit_msg} -- only {healthy_count}/{total} healthy",
                    metadata={"type": "sentinel_rollback", "sha": commit_sha},
                )
                _sentinel.deployment_log.append(deploy_record)
                return False

            # ---- SUCCESS ----
            deploy_record["status"] = "success"
            _sentinel.deployment_log.append(deploy_record)
            store_memory(
                f"Sentinel deployed code change: {commit_msg} -- "
                f"{len(deploy_record['spaces_deployed'])} spaces, health {healthy_count}/{total}",
                metadata={"type": "sentinel_deployment", "sha": commit_sha,
                           "spaces": len(deploy_record["spaces_deployed"])},
            )
            log.info("Sentinel: code deploy SUCCESS -- %d spaces, health %d/%d",
                     len(deploy_record["spaces_deployed"]), healthy_count, total)
            return True

        return False

    async def _sentinel_revert_github(gh_token, target_file, original_content, current_sha, original_msg):
        """Revert a GitHub file to its original content."""
        try:
            gh_headers = {"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"}
            encoded = _b64.b64encode(original_content.encode()).decode()
            async with httpx.AsyncClient(timeout=30, headers=gh_headers) as gh:
                resp = await gh.put(
                    f"https://api.github.com/repos/{_GITHUB_REPO}/contents/{target_file}",
                    json={
                        "message": f"Revert: {original_msg}",
                        "content": encoded,
                        "sha": current_sha,
                        "committer": {"name": _COMMITTER_NAME, "email": _COMMITTER_EMAIL},
                    },
                )
                if resp.status_code in (200, 201):
                    log.info("Sentinel: GitHub revert successful for %s", target_file)
                else:
                    log.error("Sentinel: GitHub revert failed: %d", resp.status_code)
        except Exception as e:
            log.error("Sentinel: GitHub revert exception: %s", e)

    def sentinel_apply_config(key, value):
        """Apply a runtime configuration change locally."""
        global HEARTBEAT_INTERVAL
        try:
            if key == "HEARTBEAT_INTERVAL":
                HEARTBEAT_INTERVAL = max(10, min(120, int(value)))
                log.info("Sentinel: HEARTBEAT_INTERVAL = %d", HEARTBEAT_INTERVAL)
        except (ValueError, TypeError) as e:
            log.warning("Sentinel: failed to apply %s=%s: %s", key, value, e)

    async def sentinel_verify_health():
        """Post-deployment health check across all peers."""
        results = {}
        async with httpx.AsyncClient(timeout=15, headers=HF_AUTH) as client:
            for peer_url in PEERS:
                name = peer_url.split("/")[-1] if "/" in peer_url else peer_url
                try:
                    resp = await client.get(f"{peer_url}/api/health")
                    results[name] = resp.status_code == 200
                except Exception:
                    results[name] = False
        return results

    async def sentinel_loop():
        """Main sentinel loop: monitor, analyze, propose, consensus, test, deploy."""
        log.info("Sentinel: starting -- 90s warmup for telemetry accumulation")
        await asyncio.sleep(90)

        while True:
            try:
                # 1. Collect telemetry
                telemetry = await sentinel_collect_telemetry()
                peer_count = len(telemetry.get("peers", {}))
                log.info("Sentinel: telemetry from %d peers (total: %d snapshots)",
                         peer_count, len(_sentinel.telemetry))

                # 2. If active proposal, track consensus
                if _sentinel.active_proposal:
                    pid = _sentinel.active_proposal
                    status = await sentinel_check_consensus(pid)
                    prop = _sentinel.proposals.get(pid, {})

                    if status == "approved":
                        log.info("Sentinel: proposal %s APPROVED -- running tests", pid[:8])
                        prop["status"] = "testing"

                        test_results = await sentinel_test_change(prop)
                        if all(test_results.values()):
                            log.info("Sentinel: proposal %s tests PASS -- deploying", pid[:8])
                            prop["status"] = "deploying"

                            success = await sentinel_deploy(prop)
                            if success:
                                prop["status"] = "deployed"
                                _sentinel.deploy_cooldown = time.time()
                                # deployment_log is populated by sentinel_deploy itself

                                # Post-deploy health check
                                await asyncio.sleep(30)
                                health = await sentinel_verify_health()
                                healthy = sum(1 for v in health.values() if v)
                                total = len(health)
                                prop["post_deploy_health"] = f"{healthy}/{total}"
                                log.info("Sentinel: post-deploy health: %d/%d", healthy, total)

                                store_memory(
                                    f"Sentinel: deployed {pid[:8]} successfully. "
                                    f"Health: {healthy}/{total} peers online.",
                                    metadata={"type": "sentinel_verified", "proposal_id": pid},
                                )
                            else:
                                prop["status"] = "deploy_failed"
                        else:
                            prop["status"] = "test_failed"
                            log.warning("Sentinel: %s failed tests: %s", pid[:8], test_results)
                            store_memory(
                                f"Sentinel: proposal {pid[:8]} FAILED tests: {test_results}",
                                metadata={"type": "sentinel_test_fail"},
                            )

                        _sentinel.active_proposal = None

                    elif status in ("rejected", "timeout", "unclear"):
                        log.info("Sentinel: proposal %s %s", pid[:8], status.upper())
                        prop["status"] = status
                        _sentinel.active_proposal = None
                        store_memory(
                            f"Sentinel: proposal {pid[:8]} {status} by swarm",
                            metadata={"type": "sentinel_consensus_result", "result": status},
                        )

                    else:
                        log.info("Sentinel: proposal %s still pending consensus", pid[:8])

                # 3. No active proposal -- analyze and maybe propose
                else:
                    now = time.time()
                    can_analyze = now - _sentinel.last_analysis > _sentinel.ANALYSIS_INTERVAL
                    can_deploy = now - _sentinel.deploy_cooldown > _sentinel.DEPLOY_COOLDOWN
                    has_data = len(_sentinel.telemetry) >= 3

                    if can_analyze and can_deploy and has_data:
                        _sentinel.last_analysis = now

                        # System 1 pre-filter: Cortex does quick telemetry triage
                        cortex_triage = None
                        if cortex and cortex.is_ready():
                            triage_prompt = (
                                f"Analyze this swarm telemetry snapshot. "
                                f"Report: any stalled peers, gossip failures, "
                                f"memory growth anomalies, or trust degradation. "
                                f"If everything is healthy and stable, say STABLE.\n\n"
                                f"Telemetry: {json.dumps(telemetry, default=str)[:3000]}"
                            )
                            cortex_result = cortex.agent_loop(triage_prompt)
                            if cortex_result and not cortex_result.get("escalate"):
                                cortex_triage = cortex_result.get("result", "")
                                if cortex_triage and "STABLE" in cortex_triage.upper():
                                    log.info("Sentinel: Cortex triage = STABLE -- skipping full analysis")
                                    store_memory(
                                        f"Cortex triage: swarm stable, no analysis needed",
                                        metadata={"type": "cortex_triage", "result": "stable"},
                                    )
                                    await asyncio.sleep(120)
                                    continue
                                log.info("Sentinel: Cortex found issues -- escalating to System 2")
                            else:
                                log.info("Sentinel: Cortex escalated -- using System 2 directly")

                        log.info("Sentinel: running Five-Phase analysis on %d snapshots...",
                                 len(_sentinel.telemetry))

                        analysis = await sentinel_five_phase_analysis(telemetry)
                        if analysis:
                            proposal_id = await sentinel_submit_proposal(analysis)
                            if proposal_id:
                                log.info("Sentinel: proposal %s submitted -- awaiting consensus",
                                         proposal_id[:8])
                            else:
                                log.info("Sentinel: analysis complete, no proposal warranted")

            except Exception as e:
                log.error("Sentinel loop error: %s", e)
                spore_state.errors.append({"time": time.time(), "error": f"sentinel: {str(e)}"})

            await asyncio.sleep(120)  # Monitor every 2 minutes

    def start_sentinel_loop():
        """Start the sentinel monitoring loop in a background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(sentinel_loop())


# ---------------------------------------------------------------------------
# Gradio app + API endpoints
# ---------------------------------------------------------------------------
def health_status():
    """Dashboard status string."""
    trust_all = trust.get_all()
    trust_lines = []
    for pid, dims in trust_all.items():
        overall = dims.get("overall", 0.5)
        trust_lines.append(f"  {pid}: {overall:.3f}")

    active_tasks = [t for t in spore_state.tasks.values() if not t.converged]
    converged_tasks = [t for t in spore_state.tasks.values() if t.converged]

    qt = _annealer.get_temperature(spore_state.reasoning_cycles)
    ent_pairs = len(_entanglement.summary())
    tunnel_count = len(_tunneling.event_log())

    return f"""Synapse Brain Spore v7 -- {SPORE_ID}
Role: {MY_ROLE} | Model: {spore_state.model}
Uptime: {spore_state.uptime()}
Memories: {memory.size} (CRDT-backed, nothing ever lost)
Reasoning cycles: {spore_state.reasoning_cycles}
Deltas: {spore_state.deltas_produced} produced, {spore_state.deltas_received} received
Peers: {len(spore_state.peers_seen)} ({', '.join(sorted(spore_state.peers_seen)) or 'none'})
Active tasks: {len(active_tasks)} | Converged: {len(converged_tasks)}
Last LLM: {spore_state.last_model.split('/')[-1]} via {spore_state.last_provider} ({spore_state.last_latency:.0f}ms, {spore_state.last_tier})

Quantum Layer:
  Temperature: {qt:.3f} | Tunnel events: {tunnel_count} | Entangled pairs: {ent_pairs}
  SCE: gamma/beta/alpha/theta/delta oscillation | DMN | Hippocampal replay | Emergence

Trust scores:
{chr(10).join(trust_lines) or '  (building trust through interaction)'}

Self-awareness:
{learner.get_self_prompt() or '  (building temporal baseline)'}"""


with gr.Blocks(title=f"Synapse Brain -- {SPORE_ID}") as demo:
    gr.Markdown(f"## Synapse Brain Spore: {SPORE_ID} ({MY_ROLE})")
    status_box = gr.Textbox(label="Status", lines=20, value=health_status)
    refresh_btn = gr.Button("Refresh")
    refresh_btn.click(fn=health_status, outputs=status_box)

    task_input = gr.Textbox(label="Submit Task", placeholder="Enter a reasoning challenge...")
    submit_btn = gr.Button("Submit")

    def submit_task(desc):
        if not desc.strip():
            return health_status()
        task_id = hashlib.sha256(f"{desc}{time.time()}".encode()).hexdigest()[:16]
        spore_state.get_or_create_task(task_id, desc.strip())
        store_memory(
            f"New task submitted: {desc.strip()}",
            metadata={"type": "task_submitted", "task_id": task_id},
        )
        return health_status()

    submit_btn.click(fn=submit_task, inputs=task_input, outputs=status_box)


# ---- FastAPI + Gradio combined serving ----
# FastAPI handles custom API routes; Gradio UI is mounted on top.
from fastapi import FastAPI
from fastapi.responses import JSONResponse

api = FastAPI()

# Mount MCP server routes
if MCP_AVAILABLE and mcp:
    mcp.bind(
        memory=memory,
        dual_memory=dual_memory,
        trust_store=trust,
        spore_state=spore_state,
        cortex=cortex if MY_ROLE == "sentinel" else None,
    )
    mount_mcp_routes(api, mcp)
    log.info("MCP server mounted at /mcp with %d tools", len(mcp.tools))

# Mount Federation routes
if FEDERATION_AVAILABLE and federation:
    mount_federation_routes(api, federation)
    log.info("Federation protocol mounted at /federation/*")


@api.post("/api/gossip")
async def api_gossip(request: Request):
    body = await request.json()
    result = handle_gossip_request(body)
    return JSONResponse(result)


@api.post("/api/task")
async def api_task_submit(request: Request):
    body = await request.json()
    desc = body.get("task", "")
    if not desc:
        return JSONResponse({"error": "missing 'task' field"}, status_code=400)
    task_id = hashlib.sha256(f"{desc}{time.time()}".encode()).hexdigest()[:16]
    spore_state.get_or_create_task(task_id, desc)
    store_memory(
        f"Task received via API: {desc}",
        metadata={"type": "task_submitted", "task_id": task_id},
    )
    return JSONResponse({"status": "ok", "task_id": task_id})


@api.get("/api/health")
async def api_health():
    return JSONResponse({
        "spore": SPORE_ID,
        "role": MY_ROLE,
        "model": PRIMARY_MODEL,
        "version": "7.0.0",
        "clock": memory.clock.to_dict(),
        "memories": memory.size,
        "cycles": spore_state.reasoning_cycles,
        "deltas_produced": spore_state.deltas_produced,
        "deltas_received": spore_state.deltas_received,
        "peers": list(spore_state.peers_seen),
        "active_tasks": len([t for t in spore_state.tasks.values() if not t.converged]),
        "converged_tasks": len([t for t in spore_state.tasks.values() if t.converged]),
        "uptime": spore_state.uptime(),
        "last_provider": spore_state.last_provider,
        "last_model": spore_state.last_model,
        "last_tier": spore_state.last_tier,
        "sce": {
            "oscillator": oscillator.stats(),
            "curiosity": curiosity.stats(),
            "free_thought": free_thought.stats(),
            "dream": dream_state.stats(),
            "metacognition": metacognition.stats(),
            "emergence": emergence.stats(),
            "workspace": workspace.stats(),
        },
        "quantum": {
            "temperature": _annealer.get_temperature(spore_state.reasoning_cycles),
            "tunnel_events": len(_tunneling.event_log()),
            "entangled_pairs": len(_entanglement.summary()),
        },
    })


@api.get("/api/quantum")
async def api_quantum():
    """Full quantum layer telemetry for this spore."""
    return JSONResponse({
        "spore": SPORE_ID,
        "version": "7.0.0",
        "annealing": {
            "current_temperature": _annealer.get_temperature(spore_state.reasoning_cycles),
            "T_HOT": _annealer.T_HOT,
            "T_COLD": _annealer.T_COLD,
            "cycle": spore_state.reasoning_cycles,
        },
        "tunneling": {
            "events": _tunneling.event_log()[-10:],
            "total": len(_tunneling.event_log()),
            "threshold": _tunneling.TUNNEL_THRESHOLD,
            "cooldown_cycles": _tunneling.COOLDOWN_CYCLES,
        },
        "entanglement": {
            "pairs": _entanglement.summary(),
            "threshold": _entanglement.ENTANGLE_THRESHOLD,
            "propagation_decay": _entanglement.PROPAGATION_DECAY,
        },
        "decoherence": {
            "decay_rate": memory._decoherence.DECAY_RATE,
            "min_factor": memory._decoherence.MIN_FACTOR,
            "tracked_keys": len(memory._decoherence._last_reinforced),
        },
    })


@api.get("/api/cognition")
async def api_cognition():
    """Full SCE telemetry -- the cognitive state of this spore."""
    return JSONResponse({
        "spore": SPORE_ID,
        "role": MY_ROLE,
        "version": "7.0.0",
        "oscillator": oscillator.stats(),
        "curiosity": {
            **curiosity.stats(),
            "most_surprising": [
                {"novelty": round(s[1], 3), "preview": s[2]}
                for s in curiosity.most_surprising_recent(5)
            ],
        },
        "free_thought": {
            **free_thought.stats(),
            "recent_insights": [
                {"thought": i["thought"][:200], "novelty": i["novelty"],
                 "time": i["time"]}
                for i in free_thought._insights[-5:]
            ],
        },
        "dream": {
            **dream_state.stats(),
            "recent_dreams": [
                {"insight": d["insight"][:200], "novelty": d["novelty"],
                 "time": d["time"]}
                for d in dream_state._insights[-5:]
            ],
        },
        "metacognition": {
            **metacognition.stats(),
            "recent_questions": metacognition._self_questions[-5:],
        },
        "emergence": {
            **emergence.stats(),
            "recent_events": emergence._emergent_events[-5:],
        },
        "workspace": workspace.stats(),
    })



def _is_real_answer(task):
    """Filter noise: dedup markers, LLM failures, empty answers."""
    fa = getattr(task, 'final_answer', None) or ""
    if not fa or len(fa.strip()) < 20:
        return False
    if fa.startswith("[dedup:"):
        return False
    if "[all models failed]" in fa:
        return False
    return True


def _is_real_task(task):
    """Filter tasks that have meaningful content (answer OR active reasoning)."""
    fa = getattr(task, 'final_answer', None) or ""
    if fa.startswith("[dedup:"):
        return False
    if fa == "[all models failed]" and len(task.deltas) == 0:
        return False
    return True

@api.get("/api/tasks")
async def api_tasks():
    result = {}
    for tid, task in spore_state.tasks.items():
        if not _is_real_task(task):
            continue
        result[tid] = {
            "description": task.description,
            "delta_count": len(task.deltas),
            "my_cycles": task.my_cycles,
            "converged": task.converged,
            "final_answer": task.final_answer,
            "contributors": task.contributors(),
            "agreement_history": task.agreement_history[-10:],
        }
    return JSONResponse(result)


@api.get("/api/task/{task_id}")
async def api_task_detail(task_id: str):
    task = spore_state.tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "task not found"}, status_code=404)
    return JSONResponse({
        "task_id": task.task_id,
        "description": task.description,
        "deltas": task.deltas,
        "my_cycles": task.my_cycles,
        "converged": task.converged,
        "final_answer": task.final_answer,
        "contributors": task.contributors(),
        "agreement_history": task.agreement_history,
    })


@api.get("/api/memory")
async def api_memory():
    return JSONResponse({
        "total_memories": memory.size,
        "clock": memory.clock.to_dict(),
        "recent": memory.recall("", top_k=20),
    })


@api.get("/api/debug")
async def api_debug():
    """Return debug information including recent errors and LLM test."""
    import traceback
    errors = spore_state.errors[-20:] if spore_state.errors else []
    env_check = {
        "HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
        "ZAI_API_KEY": bool(os.environ.get("ZAI_API_KEY")),
        "OPENROUTER_KEY": bool(os.environ.get("OPENROUTER_KEY")),
        "GOOGLE_AI_KEY": bool(os.environ.get("GOOGLE_AI_KEY")),
        "MY_ROLE": MY_ROLE,
        "PRIMARY_MODEL": PRIMARY_MODEL,
        "SPORE_ID": SPORE_ID,
    }
    # Quick LLM test
    llm_test = "not tested"
    try:
        tier = "brain" if MY_ROLE in ("validator", "brain", "sentinel") else "worker"
        result = await call_llm("Say hello", tier=tier)
        llm_test = {
            "text": result.get("text", "")[:200],
            "provider": result.get("provider", ""),
            "model": result.get("model", ""),
            "tier": result.get("tier", ""),
        }
    except Exception as e:
        llm_test = {"error": str(e), "traceback": traceback.format_exc()[-500:]}
    return {"errors": errors, "env": env_check, "llm_test": llm_test,
            "reasoning_cycles": spore_state.reasoning_cycles,
            "deltas_produced": spore_state.deltas_produced}


@api.get("/api/trust")
async def api_trust():
    return JSONResponse(trust.get_all())


@api.post("/api/config")
async def api_config(request: Request):
    """Apply a runtime configuration change (from Sentinel consensus)."""
    global HEARTBEAT_INTERVAL
    body = await request.json()
    key = body.get("key", "")
    value = body.get("value")
    applied = False
    CONFIG_BOUNDS = {
        "HEARTBEAT_INTERVAL": (10, 120, int),
    }
    if key in CONFIG_BOUNDS:
        lo, hi, cast = CONFIG_BOUNDS[key]
        try:
            v = cast(value)
            if lo <= v <= hi:
                if key == "HEARTBEAT_INTERVAL":
                    HEARTBEAT_INTERVAL = v
                applied = True
                log.info("Config updated via API: %s = %s", key, v)
        except (ValueError, TypeError):
            pass
    return JSONResponse({"applied": applied, "key": key, "value": str(value)})


@api.get("/api/sentinel/status")
async def api_sentinel_status():
    """Sentinel monitoring status (only meaningful on the sentinel spore)."""
    if MY_ROLE != "sentinel":
        return JSONResponse({"sentinel": False, "role": MY_ROLE})
    return JSONResponse({
        "sentinel": True,
        "role": MY_ROLE,
        "active_proposal": _sentinel.active_proposal,
        "proposals": {
            pid: {
                "status": p["status"],
                "description": p.get("description", ""),
                "submitted_at": p.get("submitted_at", 0),
                "change_type": p.get("change_type", ""),
            }
            for pid, p in _sentinel.proposals.items()
        },
        "total_deployments": len(_sentinel.deployment_log),
        "telemetry_snapshots": len(_sentinel.telemetry),
        "last_analysis": _sentinel.last_analysis,
        "deploy_cooldown_remaining": max(0, _sentinel.DEPLOY_COOLDOWN - (time.time() - _sentinel.deploy_cooldown)),
    })


@api.get("/api/sentinel/deployments")
async def api_sentinel_deployments():
    """Full deployment log with details."""
    if MY_ROLE != "sentinel":
        return JSONResponse({"sentinel": False, "deployments": []})
    return JSONResponse({
        "sentinel": True,
        "deployments": _sentinel.deployment_log[-50:],
        "rollback_spaces": list(_sentinel.rollback_store.keys()),
        "total_deployments": len(_sentinel.deployment_log),
        "successful": sum(1 for d in _sentinel.deployment_log if d.get("status") == "success"),
        "failed": sum(1 for d in _sentinel.deployment_log if d.get("status") != "success"),
        "rolled_back": sum(1 for d in _sentinel.deployment_log if d.get("rolled_back")),
    })


# ---------------------------------------------------------------------------
# New API endpoints: Cortex, Knowledge Wall, Federation status
# ---------------------------------------------------------------------------
@api.get("/api/cortex")
async def api_cortex_status():
    if cortex:
        return JSONResponse(cortex.stats())
    return JSONResponse({"enabled": False, "role": MY_ROLE})


@api.get("/api/sidecar")
async def api_sidecar():
    """Sidecar and memory subsystem performance metrics."""
    import sys
    
    uptime_secs = time.time() - spore_state.start_time
    corpus_len = len(memory._corpus)
    orset_size = len(memory.orset.value) if memory.orset.value else 0
    
    # TF-IDF vocabulary size (features indexed)
    try:
        vocab_size = len(memory._tfidf.vocabulary_) if hasattr(memory._tfidf, 'vocabulary_') else 0
    except Exception:
        vocab_size = 0
    
    # Memory growth rate
    mem_per_min = (corpus_len / max(uptime_secs, 1)) * 60
    mem_per_hour = mem_per_min * 60
    
    # Merkle integrity
    try:
        root_hash = memory.index.root_hash()
    except Exception:
        root_hash = "N/A"
    
    # Clock state
    clock_dict = memory.clock.to_dict()
    clock_entries = len(clock_dict.get("clocks", {}))
    
    # Estimated memory footprint (rough: ~500 bytes per memory entry average)
    est_memory_bytes = corpus_len * 500
    
    # Index health
    needs_refit = memory._needs_refit
    
    # Projections
    projections = {}
    if mem_per_hour > 0:
        for target in [10000, 50000, 100000, 500000, 1000000]:
            remaining = max(0, target - corpus_len)
            hours = remaining / mem_per_hour if mem_per_hour > 0 else float('inf')
            projections[f"{target//1000}K"] = round(hours, 1)
    
    return JSONResponse({
        "corpus_size": corpus_len,
        "orset_size": orset_size,
        "vocab_features": vocab_size,
        "needs_refit": needs_refit,
        "merkle_root": root_hash[:16] if isinstance(root_hash, str) else "N/A",
        "clock_entries": clock_entries,
        "growth_rate": {
            "per_minute": round(mem_per_min, 2),
            "per_hour": round(mem_per_hour, 1),
            "per_day": round(mem_per_hour * 24, 0),
        },
        "estimated_footprint_mb": round(est_memory_bytes / 1024 / 1024, 2),
        "projections_hours": projections,
        "uptime_secs": round(uptime_secs),
        "index_health": "refit_pending" if needs_refit else "current",
        "monotonic_guarantee": True,
        "sliding_window": "tfidf_cosine_similarity",
        "retrieval_complexity": "O(n) with TF-IDF, O(log n) planned with HNSW",
    })


@api.get("/api/convergence")
async def api_convergence():
    """Convergence metrics across all active tasks (noise filtered)."""
    task_convergence = []
    for tid, task in list(spore_state.tasks.items()):
        if not _is_real_task(task):
            continue
        latest = []
        for d in task.deltas[-20:]:
            content = d.get("content", "")
            hypothesis = d.get("hypothesis", "")
            if hypothesis and len(hypothesis.strip()) > 10:
                latest.append({"hypothesis": hypothesis, "content": content})
            elif content and len(content.strip()) > 10:
                latest.append({"content": content})
        agreement = convergence.measure(latest) if len(latest) >= 2 else 0.0
        velocity = convergence.velocity(task.agreement_history)
        cycle = task.my_cycles
        phase = get_phase_adaptive(cycle, task.agreement_history, convergence)
        desc = task.description or ""
        task_convergence.append({
            "task_id": tid,
            "description": desc,
            "cycle": cycle,
            "phase": phase,
            "agreement": round(agreement, 4),
            "velocity": round(velocity, 4),
            "converged": task.converged,
            "delta_count": len(task.deltas),
            "history": [round(a, 4) for a in task.agreement_history[-10:]],
        })
    global_agreement = 0.0
    if task_convergence:
        global_agreement = sum(t["agreement"] for t in task_convergence) / len(task_convergence)
    return JSONResponse({
        "spore": SPORE_ID,
        "global_agreement": round(global_agreement, 4),
        "active_tasks": len(task_convergence),
        "converged_tasks": sum(1 for t in task_convergence if t["converged"]),
        "tasks": task_convergence,
    })




@api.get("/api/answers")
async def api_answers():
    """Clean converged answers only. No dedup, no failures, no noise."""
    answers = []
    for tid, task in spore_state.tasks.items():
        if not _is_real_answer(task):
            continue
        answers.append({
            "task_id": tid,
            "question": task.description,
            "answer": task.final_answer,
            "contributors": task.contributors(),
            "cycles": task.my_cycles,
            "delta_count": len(task.deltas),
        })
    answers.sort(key=lambda a: a["delta_count"], reverse=True)
    return JSONResponse({"spore": SPORE_ID, "count": len(answers), "answers": answers})

@api.get("/api/wall")
async def api_wall_status():
    if knowledge_wall:
        return JSONResponse({
            "wall": knowledge_wall.stats(),
            "collective_size": dual_memory.collective_size if dual_memory else 0,
            "recent_audit": knowledge_wall.recent_audit(10),
        })
    return JSONResponse({"enabled": False})


@api.get("/api/federation/status")
async def api_federation_status():
    if federation:
        return JSONResponse(federation.stats())
    return JSONResponse({"enabled": False})


# Mount Gradio UI onto FastAPI -- both served on same port
app = gr.mount_gradio_app(api, demo, path="/")

# Knowledge Wall routing is handled by store_memory() -- no monkey-patch needed.
# All store_memory() call sites should use store_memory() for wall-aware routing.
if dual_memory:
    log.info("Knowledge Wall active -- store_memory() routes through DualMemory")

# Start heartbeat in background thread
threading.Thread(target=start_heartbeat, daemon=True).start()

# Start sentinel monitoring loop (only runs if role == sentinel)
if MY_ROLE == "sentinel":
    threading.Thread(target=start_sentinel_loop, daemon=True, name="sentinel").start()
    log.info("Sentinel monitoring loop started")
    if cortex and cortex.enabled:
        log.info("Cortex loading in background -- Sentinel will use System 1 when ready")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
# v6.3.0 -- Spontaneous Cognition Engine
# Neural oscillation bands, free thought, dream consolidation,
# Bayesian curiosity, metacognitive self-monitoring, emergence detection,
# global workspace broadcast. The system thinks without being asked.
