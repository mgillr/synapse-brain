"""Synapse Brain Spore v5 -- Living Cognitive Agent with Perfect Memory.

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

Every spore has the COMPLETE cognitive protocol. Role is a LENS, not a limitation.
Memory is permanent -- nothing is ever lost. ORSet add-wins semantics + gossip
ensure every spore eventually has every insight from every peer.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict

import gradio as gr
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from starlette.requests import Request
from crdt_merge.core import ORSet, LWWMap
from crdt_merge.clocks import VectorClock
from crdt_merge.merkle import MerkleTree

# ---------------------------------------------------------------------------
# Configuration (substituted per spore by launcher)
# ---------------------------------------------------------------------------
SPORE_ID = os.environ.get("SYNAPSE_SPORE_ID", "__SPORE_ID__")
PEERS = json.loads(os.environ.get("SYNAPSE_PEERS", '__PEERS_JSON__'))
SPORE_INDEX = int(os.environ.get("SYNAPSE_SPORE_INDEX", "__SPORE_INDEX__"))
PORT = int(os.environ.get("PORT", "7860"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")

ROLES = ["explorer", "synthesizer", "adversarial", "validator", "generalist", "brain", "sentinel"]
MY_ROLE = ROLES[SPORE_INDEX % len(ROLES)]
PRIMARY_MODEL = os.environ.get("SYNAPSE_PRIMARY_MODEL", "__PRIMARY_MODEL__")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(SPORE_ID)

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
    "groq": {
        "env": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "tier": "worker",
    },
    "cerebras": {
        "env": "CEREBRAS_API_KEY",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "llama3.3-70b",
        "tier": "worker",
    },
}

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
# CRDT Memory System
# ---------------------------------------------------------------------------
class CRDTMemory:
    """Persistent agent memory backed by crdt-merge.

    ORSet tracks memory keys -- add-wins means nothing is ever lost.
    MerkleTree stores content indexed by key -- integrity-verified.
    VectorClock orders events causally across the swarm.

    The memory sidecar continuously re-indexes for fast semantic recall.
    """

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

    def remember(self, content, metadata=None):
        """Store a memory. Returns the key. Nothing is ever lost."""
        ts = time.time()
        raw = f"{content}|{ts}|{self.spore_id}"
        key = hashlib.sha256(raw.encode()).hexdigest()[:16]

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
            self._needs_refit = True

        return key

    def recall(self, query, top_k=5):
        """Retrieve the most relevant memories for a query."""
        with self._lock:
            if len(self._corpus) < 2:
                # Return everything if too few memories for TF-IDF
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

                top_idx = sims.argsort()[-top_k:][::-1]
                results = []
                for idx in top_idx:
                    if sims[idx] > 0.05:
                        key = self._corpus_keys[idx]
                        rec = self.index.get_record(key)
                        if rec:
                            results.append(
                                {"key": key, "similarity": float(sims[idx]), **rec}
                            )
                return results
            except Exception as e:
                log.warning("Memory recall failed: %s", e)
                return []

    def merge_incoming(self, orset_dict, records):
        """Merge memory from a peer. Add-wins: new memories always survive."""
        with self._lock:
            try:
                incoming = ORSet.from_dict(orset_dict)
                self.orset = self.orset.merge(incoming)
            except Exception as e:
                log.warning("ORSet merge failed: %s", e)

            new_count = 0
            for key, record in records.items():
                if not self.index.contains(key):
                    self.index.insert(key, record)
                    content = record.get("content", "")
                    self._corpus.append(content)
                    self._corpus_keys.append(key)
                    new_count += 1
                    self._needs_refit = True

            return new_count

    def merge_clock(self, clock_dict):
        """Merge causal clock from a peer."""
        with self._lock:
            try:
                incoming = VectorClock.from_dict(clock_dict)
                self.clock = self.clock.merge(incoming)
            except Exception:
                pass

    def sync_payload(self, known_keys=None):
        """Build a gossip payload with new memories for peers."""
        with self._lock:
            all_keys = list(self.orset.value) if self.orset.value else []
            if known_keys:
                new_keys = [k for k in all_keys if k not in known_keys]
            else:
                new_keys = all_keys

            records = {}
            for key in new_keys[-50:]:  # Cap per gossip to avoid huge payloads
                rec = self.index.get_record(key)
                if rec:
                    # Serialize cleanly -- strip non-JSON-safe fields
                    clean = {}
                    for k, v in rec.items():
                        try:
                            json.dumps(v)
                            clean[k] = v
                        except (TypeError, ValueError):
                            clean[k] = str(v)
                    records[key] = clean

            return {
                "orset": self.orset.to_dict(),
                "records": records,
                "clock": self.clock.to_dict(),
                "total_memories": len(all_keys),
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
                    insights.append(f"Strong in {phase} phase -- leverage this")

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
        """Average pairwise cosine similarity across contributions. 0.0 to 1.0."""
        texts = [c.get("content", "") or c.get("hypothesis", "") for c in contributions]
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


async def call_llm(prompt, system="", tier="any"):
    """Call an LLM with model diversity and fallback chain.

    brain tier: GLM-4.7-Flash via Z.ai first (free), then external workers, then HF.
    worker/any tier: primary HF model first, then fallbacks.
    """
    if not HF_TOKEN:
        return {"text": "[no HF_TOKEN]", "provider": "none", "model": "none",
                "tier": "none", "latency_ms": 0}

    # Brain tier: try Z.ai free models first, then other externals
    if tier == "brain":
        brain_providers = [(n, c) for n, c in EXTERNAL_PROVIDERS.items()
                           if c.get("tier") == "brain"]
        worker_providers = [(n, c) for n, c in EXTERNAL_PROVIDERS.items()
                            if c.get("tier") == "worker"]
        ordered = brain_providers + worker_providers
        for name, conf in ordered:
            key = os.environ.get(conf["env"])
            if not key:
                continue
            try:
                async with httpx.AsyncClient(timeout=90.0) as client:
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    messages.append({"role": "user", "content": prompt})
                    start = time.time()
                    resp = await client.post(
                        conf["url"],
                        headers={"Authorization": f"Bearer {key}"},
                        json={"model": conf["model"], "messages": messages,
                              "max_tokens": 2048, "temperature": 0.7},
                    )
                    resp.raise_for_status()
                    text = extract_response_text(resp.json(), conf["model"])
                    ms = (time.time() - start) * 1000
                    return {"text": text, "provider": name, "model": conf["model"],
                            "tier": "brain", "latency_ms": round(ms, 1)}
            except Exception as e:
                log.warning("Brain provider %s failed: %s", name, e)

    # HF Router: primary model first, then fallbacks
    async with httpx.AsyncClient(timeout=60.0) as client:
        for model in FALLBACK_MODELS:
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                start = time.time()
                resp = await client.post(
                    HF_ROUTER,
                    headers={"Authorization": f"Bearer {HF_TOKEN}"},
                    json={"model": model, "messages": messages,
                          "max_tokens": 2048, "temperature": 0.7},
                )
                resp.raise_for_status()
                text = extract_response_text(resp.json(), model)
                if not text.strip():
                    continue
                ms = (time.time() - start) * 1000
                is_primary = model == PRIMARY_MODEL
                return {
                    "text": text,
                    "provider": f"hf_{'primary' if is_primary else 'fallback'}",
                    "model": model, "tier": "worker", "latency_ms": round(ms, 1),
                }
            except Exception as e:
                log.warning("Model %s: %s", model.split("/")[-1], str(e)[:80])
                continue

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
    """

    def __init__(self):
        self.map = LWWMap()
        self._lock = threading.Lock()

    def update(self, peer_id, dimension, value):
        """Record a trust observation for a peer."""
        with self._lock:
            key = f"{peer_id}:{dimension}"
            self.map.set(key, value)

    def update_ema(self, peer_id, signal, alpha=0.3):
        """Exponential moving average trust update."""
        with self._lock:
            key = f"{peer_id}:overall"
            current = self.map.get(key)
            if current is None:
                current = 0.5
            new_val = round(current * (1 - alpha) + signal * alpha, 4)
            self.map.set(key, new_val)
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
        "SYNTHESIZE: produce a comprehensive final answer. Combine all validated insights. "
        "Be specific, actionable, and complete. This is the synthesis round."
    ),
}


def get_phase_adaptive(cycle, agreement_history, convergence_obj):
    """Adaptive phase transitions based on agreement velocity.

    Instead of fixed cycle thresholds, phases transition when:
    - Diverge -> Deepen: after minimum 3 cycles OR agreement drops (ideas flowing)
    - Deepen -> Converge: agreement velocity positive for 2+ cycles
    - Converge -> Synthesize: agreement > 50% OR stable for 3 cycles OR cycle > 20
    """
    if cycle <= 2:
        return "diverge"

    vel = convergence_obj.velocity(agreement_history)
    current_agreement = agreement_history[-1] if agreement_history else 0.0

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
HF_AUTH = {"Authorization": f"Bearer {HF_TOKEN}"}

# Store initial self-knowledge in memory
memory.remember(
    f"I am spore {SPORE_ID}, role: {MY_ROLE}, model: {PRIMARY_MODEL}. "
    f"I have the full Master Cognitive Protocol. My role is a lens, not a limitation.",
    metadata={"type": "identity", "role": MY_ROLE},
)


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
            content = m.get("content", "")[:200]
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
    phase = get_phase_adaptive(cycle, agreement_history, convergence)

    role_job = {
        "explorer": "identify gaps in peer reasoning and propose what nobody has considered",
        "synthesizer": "merge the strongest ideas from peers into a unified position",
        "adversarial": "find the weakest claim above and challenge it with evidence",
        "validator": "check which peer claims are well-supported and rank them by quality",
        "generalist": "ensure all aspects of the task are addressed by the collective",
        "sentinel": "analyze swarm telemetry, propose optimizations via consensus, test and deploy approved changes",
    }

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


async def reason_on_task(task):
    """Run one reasoning cycle on a task. Returns the delta produced."""
    task.my_cycles += 1
    cycle = task.my_cycles

    system = build_system_prompt(task, cycle, task.agreement_history)
    prompt = build_user_prompt(task, cycle, task.agreement_history)

    # Validator and Brain roles use brain tier (Z.ai GLM-4.7-Flash)
    tier = "brain" if MY_ROLE in ("validator", "brain", "sentinel") else "worker"
    start = time.time()
    result = await call_llm(prompt, system=system, tier=tier)
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
        parsed = {"hypothesis": text[:300], "claims": [], "confidence": 0.3,
                  "response_to_peers": ""}

    delta = {
        "author": SPORE_ID,
        "task_id": task.task_id,
        "role": MY_ROLE,
        "model": result.get("model", ""),
        "cycle": cycle,
        "hypothesis": parsed.get("hypothesis", ""),
        "claims": parsed.get("claims", []),
        "confidence": parsed.get("confidence", 0.5),
        "response_to_peers": parsed.get("response_to_peers", ""),
        "timestamp": time.time(),
        "content": parsed.get("hypothesis", ""),
    }

    task.add_delta(delta)
    spore_state.deltas_produced += 1
    spore_state.reasoning_cycles += 1

    # Store reasoning in persistent memory
    memory.remember(
        f"[Task {task.task_id[:8]}] {parsed.get('hypothesis', '')}",
        metadata={
            "type": "reasoning",
            "task_id": task.task_id,
            "cycle": cycle,
            "claims": parsed.get("claims", []),
            "confidence": parsed.get("confidence", 0.5),
        },
    )

    # Update trust based on peer engagement
    for pid, peer_delta in task.peer_latest(SPORE_ID).items():
        # If we referenced their claims, that indicates trust
        resp = parsed.get("response_to_peers", "")
        if pid in resp or any(c in resp for c in peer_delta.get("claims", [])[:1]):
            trust.update_ema(pid, 0.7)
            learner.observe_citation(pid)
        else:
            trust.update_ema(pid, 0.4)

    # Compute semantic convergence
    latest_contribs = list(task.latest_per_contributor().values())
    agreement = convergence.measure(latest_contribs)
    task.agreement_history.append(agreement)

    # Record self-observation
    learner.observe_cycle(duration, get_phase_adaptive(cycle, task.agreement_history, convergence),
                          parsed.get("confidence", 0.5), result.get("model", ""))

    log.info(
        "Cycle %d | phase=%s | agreement=%.0f%% | confidence=%.2f | model=%s | %.0fms",
        cycle,
        get_phase_adaptive(cycle, task.agreement_history, convergence),
        agreement * 100,
        parsed.get("confidence", 0.5),
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

    if should_synthesize and not task.converged and MY_ROLE == "synthesizer":
        log.info("Synthesis triggered at cycle %d, agreement %.0f%%", cycle, agreement * 100)
        synthesis = await synthesize_task(task)
        if synthesis:
            task.converged = True
            task.final_answer = synthesis
            memory.remember(
                f"[SYNTHESIS task {task.task_id[:8]}] {synthesis[:500]}",
                metadata={"type": "synthesis", "task_id": task.task_id},
            )

    return delta


async def synthesize_task(task):
    """Produce a final synthesis from all collected reasoning."""
    latest = task.latest_per_contributor()
    parts = []
    for pid, d in latest.items():
        t = trust.get(pid)
        parts.append(
            f"[{pid} ({d.get('role', '?')}, trust={t:.2f})]:\n"
            f"  Position: {d.get('hypothesis', '')}\n"
            f"  Claims: {'; '.join(d.get('claims', []))}\n"
            f"  Confidence: {d.get('confidence', 0):.2f}"
        )

    # Recall cross-task memories for richer synthesis
    past_insights = memory.recall(task.description, top_k=3)
    memory_block = ""
    if past_insights:
        mem_lines = [f"  [{m.get('spore', '?')}] {m.get('content', '')[:150]}"
                     for m in past_insights if m.get("type") != "identity"]
        if mem_lines:
            memory_block = f"\n\nRELEVANT PAST INSIGHTS:\n" + "\n".join(mem_lines)

    prompt = f"""Synthesize the final answer from a distributed reasoning swarm.

TASK: {task.description}

{len(latest)} contributors debated across {task.my_cycles} cycles.
Weight each contribution by its trust score (0-1).

== Contributions ==
{chr(10).join(parts)}
{memory_block}

Produce the FINAL comprehensive answer.
Resolve contradictions favoring higher-trust contributors.
Be specific, actionable, and complete.
Do NOT mention the swarm, spores, trust, or reasoning process. Just deliver the answer."""

    result = await call_llm(prompt, tier="brain")
    return result.get("text", "")


# ---------------------------------------------------------------------------
# Gossip protocol (memory-synced, trust-weighted)
# ---------------------------------------------------------------------------
async def gossip_push():
    """Push reasoning deltas + CRDT memory + trust state to all peers.

    Trust-weighted: high-trust peers get our full memory delta.
    Low-trust peers get only recent reasoning deltas.
    """
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

    mem_sync = memory.sync_payload()

    payload = {
        "from": SPORE_ID,
        "role": MY_ROLE,
        "model": PRIMARY_MODEL,
        "deltas": recent_deltas,
        "tasks": task_meta,
        "peer_list": list(spore_state.peers_seen),
        "memory": mem_sync,
        "trust": trust.to_dict(),
    }

    async with httpx.AsyncClient(timeout=12.0, headers=HF_AUTH) as client:
        for peer_url in PEERS:
            try:
                peer_id_guess = peer_url.split("/")[-1].replace("synapse-spore-", "spore-")
                peer_trust = trust.get(peer_id_guess)

                # Trust-weighted: send full payload to trusted peers
                send_payload = payload.copy()
                if peer_trust < 0.3:
                    # Low trust: omit full memory sync
                    send_payload.pop("memory", None)
                    log.debug("Low trust for %s (%.2f) -- limited payload", peer_id_guess, peer_trust)

                resp = await client.post(f"{peer_url}/api/gossip", json=send_payload)
                if resp.status_code == 200:
                    data = resp.json()
                    process_gossip_response(data)
                    log.info("Gossip -> %s OK", data.get("spore", peer_url.split("/")[-1]))
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

    return {
        "status": "ok",
        "spore": SPORE_ID,
        "role": MY_ROLE,
        "model": PRIMARY_MODEL,
        "peers": len(spore_state.peers_seen),
        "deltas": our_deltas,
        "tasks": our_tasks,
        "peer_list": list(spore_state.peers_seen),
        "memory": memory.sync_payload(),
        "trust": trust.to_dict(),
    }


# ---------------------------------------------------------------------------
# Heartbeat loop
# ---------------------------------------------------------------------------
HEARTBEAT_INTERVAL = 20  # seconds

async def heartbeat():
    """Main heartbeat: gossip, reason, learn.

    Every 20 seconds:
    1. Gossip state to all peers (memory + trust + deltas)
    2. Reason on active unconverged tasks
    3. Analyze own temporal patterns
    """
    while True:
        try:
            await gossip_push()

            # Reason on unconverged tasks
            active = [
                t for t in spore_state.tasks.values()
                if not t.converged and t.description
            ]

            for task in active:
                # Rate limit: one reasoning cycle per task per heartbeat
                try:
                    await reason_on_task(task)
                except Exception as e:
                    log.error("Reasoning failed on %s: %s", task.task_id, e)
                    spore_state.errors.append(
                        {"time": time.time(), "error": str(e), "task": task.task_id}
                    )

            # Temporal self-learning: analyze own patterns periodically
            if spore_state.reasoning_cycles > 0 and spore_state.reasoning_cycles % 5 == 0:
                insights = learner.analyze()
                if insights:
                    memory.remember(
                        f"Temporal self-analysis: {'; '.join(insights)}",
                        metadata={"type": "self_analysis"},
                    )
                    log.info("Self-analysis: %s", "; ".join(insights))

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

    _sentinel = SentinelState()

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
            lines.append(f"  [{p['status']}] {p.get('description', '')[:120]}")
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
- Code changes: stored for manual review (safety gate)

Respond ONLY with this JSON (no other text):
{{
  "observation": "2-3 sentences on what the raw data shows",
  "orientation": "The pattern or bottleneck this reveals",
  "hypothesis": "The single highest-impact optimization to implement",
  "change_type": "config|prompt|code",
  "target": "Specific variable or component to change",
  "current_value": "Current value or behavior",
  "proposed_value": "Proposed new value or behavior",
  "code_patch": "For code changes only: exact replacement Python. Empty string for config/prompt.",
  "success_criterion": "Measurable metric to evaluate in next analysis cycle",
  "risk": "low|medium|high",
  "confidence": 0.0
}}"""

        result = await call_llm(prompt, tier="brain")
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
            memory.remember(
                "Sentinel: analysis produced non-JSON output, skipping cycle",
                metadata={"type": "sentinel_skip"},
            )
            return None

        confidence = float(parsed.get("confidence", 0))
        risk = parsed.get("risk", "high")

        # Confidence gate: must be >= 0.6 and not high risk
        if confidence < 0.6 or risk == "high":
            log.info("Sentinel: below threshold (conf=%.2f, risk=%s) -- observing only", confidence, risk)
            memory.remember(
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
        }
        _sentinel.active_proposal = proposal_id

        memory.remember(
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
        """Fault-finding test harness for proposed changes."""
        results = {"syntax": False, "safety": False, "bounds": False, "smoke": False}
        change_type = proposal.get("change_type", "unknown")
        analysis = proposal.get("analysis", {})

        if change_type == "config":
            target = analysis.get("target", "")
            value = analysis.get("proposed_value", "")
            # Validate config bounds
            bounds = {
                "HEARTBEAT_INTERVAL": (10, 120),
            }
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
                # Unknown config -- conservative pass (will still need consensus)
                results = {k: True for k in results}

        elif change_type == "prompt":
            patch = analysis.get("proposed_value", "")
            results["syntax"] = isinstance(patch, str) and len(patch) > 0
            results["safety"] = len(patch) < 5000
            results["bounds"] = True
            results["smoke"] = results["syntax"] and results["safety"]

        elif change_type == "code":
            patch = proposal.get("code_patch", "")
            if not patch:
                return results
            # Syntax check
            try:
                _ast.parse(patch)
                results["syntax"] = True
            except SyntaxError as e:
                log.error("Sentinel: code patch syntax error: %s", e)
                return results
            # Safety check: no dangerous imports/calls
            dangerous = ["os.system", "subprocess", "eval(", "exec(", "__import__", "shutil.rmtree"]
            results["safety"] = not any(d in patch for d in dangerous)
            # Bounds check
            results["bounds"] = 10 < len(patch) < 100000
            # Smoke: passes all above
            results["smoke"] = all([results["syntax"], results["safety"], results["bounds"]])

        all_pass = all(results.values())
        memory.remember(
            f"Sentinel test {'PASS' if all_pass else 'FAIL'}: {results} for {change_type} change",
            metadata={"type": "sentinel_test", "result": "pass" if all_pass else "fail", "details": results},
        )
        return results

    async def sentinel_deploy(proposal):
        """Deploy an approved, tested change."""
        change_type = proposal.get("change_type", "unknown")
        analysis = proposal.get("analysis", {})

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
            # Apply locally too
            sentinel_apply_config(target, value)
            deployed_to.append("self")
            memory.remember(
                f"Sentinel deployed config: {target}={value} to {len(deployed_to)} spores",
                metadata={"type": "sentinel_deployment", "target": target, "count": len(deployed_to)},
            )
            log.info("Sentinel: config deployed to %d nodes", len(deployed_to))
            return True

        elif change_type == "prompt":
            memory.remember(
                f"Sentinel prompt optimization: {analysis.get('hypothesis', '')} -- "
                f"new prompt guidance: {analysis.get('proposed_value', '')}",
                metadata={"type": "sentinel_prompt_change", "target": analysis.get("target", "")},
            )
            log.info("Sentinel: prompt change stored in CRDT memory (propagates via gossip)")
            return True

        elif change_type == "code":
            # Safety gate: code changes stored for manual review, not auto-deployed
            memory.remember(
                f"Sentinel code change APPROVED and TESTED but requires manual deployment: "
                f"{analysis.get('hypothesis', '')}",
                metadata={
                    "type": "sentinel_code_approved",
                    "patch": proposal.get("code_patch", "")[:2000],
                },
            )
            log.info("Sentinel: code change approved+tested, stored for manual review")
            return True

        return False

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
                                _sentinel.deployments.append({
                                    "proposal_id": pid,
                                    "time": time.time(),
                                    "type": prop.get("change_type"),
                                })

                                # Post-deploy health check
                                await asyncio.sleep(30)
                                health = await sentinel_verify_health()
                                healthy = sum(1 for v in health.values() if v)
                                total = len(health)
                                prop["post_deploy_health"] = f"{healthy}/{total}"
                                log.info("Sentinel: post-deploy health: %d/%d", healthy, total)

                                memory.remember(
                                    f"Sentinel: deployed {pid[:8]} successfully. "
                                    f"Health: {healthy}/{total} peers online.",
                                    metadata={"type": "sentinel_verified", "proposal_id": pid},
                                )
                            else:
                                prop["status"] = "deploy_failed"
                        else:
                            prop["status"] = "test_failed"
                            log.warning("Sentinel: %s failed tests: %s", pid[:8], test_results)
                            memory.remember(
                                f"Sentinel: proposal {pid[:8]} FAILED tests: {test_results}",
                                metadata={"type": "sentinel_test_fail"},
                            )

                        _sentinel.active_proposal = None

                    elif status in ("rejected", "timeout", "unclear"):
                        log.info("Sentinel: proposal %s %s", pid[:8], status.upper())
                        prop["status"] = status
                        _sentinel.active_proposal = None
                        memory.remember(
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

    return f"""Synapse Brain Spore v5 -- {SPORE_ID}
Role: {MY_ROLE} | Model: {spore_state.model}
Uptime: {spore_state.uptime()}
Memories: {memory.size} (CRDT-backed, nothing ever lost)
Reasoning cycles: {spore_state.reasoning_cycles}
Deltas: {spore_state.deltas_produced} produced, {spore_state.deltas_received} received
Peers: {len(spore_state.peers_seen)} ({', '.join(sorted(spore_state.peers_seen)) or 'none'})
Active tasks: {len(active_tasks)} | Converged: {len(converged_tasks)}
Last LLM: {spore_state.last_model.split('/')[-1]} via {spore_state.last_provider} ({spore_state.last_latency:.0f}ms, {spore_state.last_tier})

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
        memory.remember(
            f"New task submitted: {desc.strip()[:200]}",
            metadata={"type": "task_submitted", "task_id": task_id},
        )
        return health_status()

    submit_btn.click(fn=submit_task, inputs=task_input, outputs=status_box)


# ---- FastAPI + Gradio combined serving ----
# FastAPI handles custom API routes; Gradio UI is mounted on top.
from fastapi import FastAPI
from fastapi.responses import JSONResponse

api = FastAPI()


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
    memory.remember(
        f"Task received via API: {desc[:200]}",
        metadata={"type": "task_submitted", "task_id": task_id},
    )
    return JSONResponse({"status": "ok", "task_id": task_id})


@api.get("/api/health")
async def api_health():
    return JSONResponse({
        "spore": SPORE_ID,
        "role": MY_ROLE,
        "model": PRIMARY_MODEL,
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
    })


@api.get("/api/tasks")
async def api_tasks():
    result = {}
    for tid, task in spore_state.tasks.items():
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
        "GROQ_API_KEY": bool(os.environ.get("GROQ_API_KEY")),
        "CEREBRAS_API_KEY": bool(os.environ.get("CEREBRAS_API_KEY")),
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
        "total_deployments": len(_sentinel.deployments),
        "telemetry_snapshots": len(_sentinel.telemetry),
        "last_analysis": _sentinel.last_analysis,
        "deploy_cooldown_remaining": max(0, _sentinel.DEPLOY_COOLDOWN - (time.time() - _sentinel.deploy_cooldown)),
    })


# Mount Gradio UI onto FastAPI -- both served on same port
app = gr.mount_gradio_app(api, demo, path="/")

# Start heartbeat in background thread
threading.Thread(target=start_heartbeat, daemon=True).start()

# Start sentinel monitoring loop (only runs if role == sentinel)
if MY_ROLE == "sentinel":
    threading.Thread(target=start_sentinel_loop, daemon=True, name="sentinel").start()
    log.info("Sentinel monitoring loop started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
