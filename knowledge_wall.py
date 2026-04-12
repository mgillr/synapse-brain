"""Synapse Knowledge Wall -- BastionWall Privacy Boundary Transplant.

Enforces cryptographic isolation between private conversations and
collective intelligence. Derived from the BastionWall Brewer-Nash
isolation gateway (project-discovery/bastion-wall branch).

Architecture:
  - Two-layer CRDT memory: private (local only) + collective (gossiped)
  - One-way distillation: conversations -> factual insights
  - HMAC-SHA256 provenance binding (anonymous source proof)
  - PII detection and destruction at the boundary
  - Full audit trail of every wall crossing
  - Conflict class isolation for multi-tenant deployments

Nothing from private memory enters collective memory without passing
through the Knowledge Wall. Raw conversations are destroyed, not encrypted.
They cannot be recovered because they do not exist after the boundary.
"""
import hashlib
import hmac
import json
import logging
import re
import time
from typing import Any

log = logging.getLogger("knowledge_wall")

# PII patterns to detect and strip
PII_PATTERNS = [
    # Email addresses
    (re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'), "[EMAIL]"),
    # IP addresses
    (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), "[IP]"),
    # URLs with paths (may contain user info)
    (re.compile(r'https?://[^\s]+'), "[URL]"),
    # API keys / tokens (common patterns)
    (re.compile(r'(?:hf_|sk-|ghp_|gho_|github_pat_)[a-zA-Z0-9_-]{10,}'), "[TOKEN]"),
    # Phone numbers
    (re.compile(r'\+?\d[\d\s-]{8,}\d'), "[PHONE]"),
    # File paths with usernames
    (re.compile(r'/(?:home|Users)/[a-zA-Z0-9_.-]+'), "[PATH]"),
]

# Conversation markers to strip
CONVERSATION_MARKERS = re.compile(
    r'(?:^|\s)(?:'
    r'(?:I|you|they|we|he|she)\s+(?:asked|said|responded|replied|told|mentioned|wrote|noted)'
    r'|user\s+(?:query|request|question|input)'
    r'|in\s+(?:our|my|their)\s+conversation'
    r'|as\s+discussed'
    r'|(?:commander|operator)\s+\S+'
    r')(?:\s|$)',
    re.IGNORECASE
)

# Identity markers
IDENTITY_MARKERS = re.compile(
    r'Commander-[A-Za-z0-9_-]+|'
    r'(?:user|operator|admin)[_-]?[A-Za-z0-9]+|'
    r'@[A-Za-z0-9_]+',
    re.IGNORECASE
)

# Factual insight extraction patterns
INSIGHT_INDICATORS = [
    "achieves", "accuracy", "performance", "outperforms", "benchmark",
    "strategy", "parameter", "convergence", "throughput", "latency",
    "model", "architecture", "merge", "weight", "tensor", "evaluation",
    "discovered", "observed", "measured", "tested", "verified",
    "pattern", "anomaly", "trend", "correlation", "regression",
    "trust", "score", "metric", "delta", "gossip", "peer",
]


def hmac_sha256(secret: bytes, data: bytes) -> bytes:
    """HMAC-SHA256 binding -- same primitive as BastionWall."""
    return hmac.new(secret, data, hashlib.sha256).digest()


class AuditEntry:
    """Record of a wall crossing or block."""

    __slots__ = ("timestamp", "action", "provenance_prefix", "reason")

    def __init__(self, action: str, provenance_prefix: str = "",
                 reason: str = ""):
        self.timestamp = time.time()
        self.action = action
        self.provenance_prefix = provenance_prefix
        self.reason = reason

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "provenance": self.provenance_prefix,
            "reason": self.reason,
        }


class ConflictClass:
    """Brewer-Nash conflict class for organization isolation."""

    def __init__(self, name: str, members: set[str] | None = None):
        self.name = name
        self.members = members or set()

    def contains(self, commander_id: str) -> bool:
        return commander_id in self.members

    def add(self, commander_id: str):
        self.members.add(commander_id)

    def conflicts_with(self, commander_id: str) -> bool:
        """True if this commander is in the same conflict class as others."""
        return commander_id in self.members and len(self.members) > 1


class KnowledgeWall:
    """One-way distillation boundary between private and collective memory.

    Mirrors BastionWall's Sanitizer + WallContext pattern:
    - Raw data enters from private side
    - Only sanitized factual insights exit to collective side
    - HMAC-bound anonymous provenance on every crossing
    - PII scan before any crossing
    - Full audit trail

    The wall is irreversible: once data crosses, the original is not
    recoverable from the collective side. This is destruction, not encryption.
    """

    def __init__(self, commander_id: str, hmac_secret: bytes | None = None):
        self.commander_id = commander_id
        self.hmac_secret = hmac_secret or hashlib.sha256(
            commander_id.encode()
        ).digest()
        self.audit_trail: list[AuditEntry] = []
        self.conflict_classes: list[ConflictClass] = []
        self.crossings = 0
        self.blocks = 0

    def distill(self, raw_entry: dict) -> dict | None:
        """Distill a raw memory entry into collective-safe knowledge.

        Process (mirrors BastionWall envelope creation):
        1. EXTRACT: Pull factual claims from conversation context
        2. SANITIZE: Strip all identity, PII, conversation flow
        3. HMAC-BIND: Cryptographic provenance (anonymous source proof)
        4. VALIDATE: Verify zero leakage before crossing the wall
        5. AUDIT: Record the crossing

        Returns None if entry contains nothing worth sharing or if
        PII leakage is detected after sanitization.
        """
        content = raw_entry.get("content", "")
        metadata = raw_entry.get("metadata", {})
        entry_type = metadata.get("type", "general")

        # Step 1: Extract factual insight
        insight = self._extract_insight(content, entry_type, metadata)
        if not insight:
            self.audit_trail.append(AuditEntry(
                "FILTERED", reason="no_extractable_insight"
            ))
            return None

        # Step 2: Sanitize
        sanitized_content = self._sanitize(insight)
        if not sanitized_content or len(sanitized_content.strip()) < 10:
            self.audit_trail.append(AuditEntry(
                "FILTERED", reason="empty_after_sanitization"
            ))
            return None

        # Step 3: HMAC-bind provenance
        provenance_data = f"{self.commander_id}:{sanitized_content}:{time.time()}"
        provenance_hash = hmac_sha256(
            self.hmac_secret, provenance_data.encode()
        ).hex()

        # Step 4: Validate zero leakage
        if self._detect_leakage(sanitized_content):
            self.blocks += 1
            self.audit_trail.append(AuditEntry(
                "BLOCKED", provenance_hash[:16],
                reason="pii_detected_post_sanitization"
            ))
            log.warning("Knowledge Wall BLOCKED: PII in sanitized output")
            return None

        # Step 5: Audit and return
        self.crossings += 1
        self.audit_trail.append(AuditEntry(
            "CROSSED", provenance_hash[:16]
        ))

        return {
            "content": sanitized_content,
            "provenance": provenance_hash,
            "type": self._classify_insight_type(entry_type),
            "timestamp": time.time(),
            "wall_version": "1.0",
        }

    def _extract_insight(self, content: str, entry_type: str,
                         metadata: dict) -> str | None:
        """Extract factual insight from raw content.

        Filters out conversational noise and extracts only content
        that would benefit the collective intelligence.
        """
        if not content or len(content) < 20:
            return None

        # Skip pure sentinel internal state
        skip_types = {
            "sentinel_skip", "sentinel_observation",
            "heartbeat", "gossip_error", "startup",
        }
        if entry_type in skip_types:
            return None

        # Skip entries that are purely conversational
        lower = content.lower()
        if any(marker in lower for marker in [
            "how do i", "can you", "please help",
            "thank you", "thanks for", "hello",
            "what is your", "tell me about",
        ]):
            return None

        # Check if content has factual substance
        insight_score = sum(
            1 for indicator in INSIGHT_INDICATORS
            if indicator in lower
        )

        # Sentinel analyses, proposals, deployments are always valuable
        valuable_types = {
            "sentinel_proposal", "sentinel_deployment",
            "sentinel_test", "sentinel_verified",
            "sentinel_rollback", "sentinel_commit",
            "reasoning", "synthesis", "convergence",
            "evaluation", "benchmark", "discovery",
        }
        if entry_type in valuable_types:
            return content

        # Require minimum insight density
        if insight_score < 2:
            return None

        return content

    def _sanitize(self, text: str) -> str:
        """Strip all identity markers, PII, and conversation context.

        This mirrors BastionWall's Sanitizer -- data is destroyed, not masked.
        """
        result = text

        # Strip identity markers
        result = IDENTITY_MARKERS.sub("", result)

        # Strip conversation markers
        result = CONVERSATION_MARKERS.sub(" ", result)

        # Strip PII patterns
        for pattern, replacement in PII_PATTERNS:
            result = pattern.sub(replacement, result)

        # Strip commander references
        result = re.sub(
            r'Commander[\s-]?[A-Za-z0-9_-]+', '', result, flags=re.IGNORECASE
        )

        # Collapse whitespace
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    def _detect_leakage(self, text: str) -> bool:
        """Final PII scan -- blocks crossing if anything slipped through."""
        for pattern, _ in PII_PATTERNS:
            # Allow generic replacements like [EMAIL] but catch real PII
            matches = pattern.findall(text)
            for match in matches:
                if not match.startswith("["):
                    return True

        # Check for residual identity patterns
        if re.search(r'Commander-\w+', text, re.IGNORECASE):
            return True
        if re.search(
            r'(?:hf_|sk-|ghp_|github_pat_)\w+', text, re.IGNORECASE
        ):
            return True

        return False

    def _classify_insight_type(self, entry_type: str) -> str:
        """Map internal entry types to collective insight categories."""
        mapping = {
            "reasoning": "analysis",
            "synthesis": "synthesis",
            "convergence": "convergence",
            "sentinel_proposal": "optimization",
            "sentinel_deployment": "deployment",
            "sentinel_test": "validation",
            "sentinel_verified": "verification",
            "sentinel_rollback": "incident",
            "evaluation": "evaluation",
            "benchmark": "benchmark",
            "discovery": "discovery",
        }
        return mapping.get(entry_type, "general")

    def check_conflict_class(self, source_commander: str,
                              target_commander: str) -> bool:
        """Brewer-Nash check: can knowledge flow from source to target?

        Returns True if flow is ALLOWED (no conflict).
        Returns False if flow is BLOCKED (same conflict class).
        """
        for cc in self.conflict_classes:
            if cc.contains(source_commander) and cc.contains(target_commander):
                return False
        return True

    def register_conflict_class(self, name: str,
                                 members: set[str]) -> ConflictClass:
        """Register a Brewer-Nash conflict class."""
        cc = ConflictClass(name, members)
        self.conflict_classes.append(cc)
        return cc

    def stats(self) -> dict:
        return {
            "commander_id": self.commander_id,
            "crossings": self.crossings,
            "blocks": self.blocks,
            "audit_entries": len(self.audit_trail),
            "conflict_classes": len(self.conflict_classes),
            "block_rate": (
                round(self.blocks / max(1, self.crossings + self.blocks), 4)
            ),
        }

    def recent_audit(self, count: int = 20) -> list[dict]:
        return [e.to_dict() for e in self.audit_trail[-count:]]


class DualMemory:
    """Two-layer memory architecture with Knowledge Wall separation.

    Layer 1 (private): raw conversations, tasks, identity -- never gossiped.
    Layer 2 (collective): distilled knowledge only -- gossiped to swarm.

    The wall sits between them. Nothing crosses without distillation.
    """

    def __init__(self, base_memory, wall: KnowledgeWall):
        """
        Args:
            base_memory: The existing CRDTMemory instance (becomes private layer).
            wall: The KnowledgeWall for distillation.
        """
        self.private = base_memory  # existing CRDTMemory -- raw data
        self.wall = wall

        # Collective layer uses same structure but separate instance
        # Populated only by wall-distilled entries
        self._collective_keys: set[str] = set()
        self._collective_records: dict[str, dict] = {}

    def remember(self, content: str, metadata: dict | None = None) -> str:
        """Store to private memory AND attempt distillation to collective."""
        # Always store in private (existing behavior preserved)
        key = self.private.remember(content, metadata)

        # Attempt distillation through the wall
        raw_entry = {
            "content": content,
            "metadata": metadata or {},
        }
        distilled = self.wall.distill(raw_entry)
        if distilled:
            # Store in collective layer
            collective_key = hashlib.sha256(
                f"collective:{distilled['content']}:{distilled['provenance']}"
                .encode()
            ).hexdigest()[:16]
            self._collective_keys.add(collective_key)
            self._collective_records[collective_key] = distilled
            log.debug("Wall crossing: %s -> collective", collective_key[:8])

        return key

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Recall from private memory (local queries see everything)."""
        return self.private.recall(query, top_k)

    def collective_payload(self) -> dict:
        """Build gossip payload from collective layer ONLY.

        This is what goes over the wire. Private memory never appears here.
        No record cap -- delta gossip handles bandwidth at the transport layer.
        """
        return {
            "collective_records": dict(self._collective_records),
            "collective_keys": list(self._collective_keys),
            "wall_stats": self.wall.stats(),
        }

    def recall_collective(self, query, top_k=5):
        """Semantic search over collective knowledge.

        Public API for MCP and external callers -- no underscore access needed.
        """
        if not self._collective_records:
            return []
        contents = [r.get("content", "") for r in self._collective_records.values()]
        keys = list(self._collective_records.keys())
        if len(contents) < 2:
            return list(self._collective_records.values())[:top_k]

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        tfidf.fit(contents)
        query_vec = tfidf.transform([query])
        corpus_vecs = tfidf.transform(contents)
        sims = cosine_similarity(query_vec, corpus_vecs)[0]
        top_idx = sims.argsort()[-top_k:][::-1]
        return [
            {**self._collective_records[keys[i]], "similarity": float(sims[i])}
            for i in top_idx if sims[i] > 0.05
        ]

    def merge_collective(self, incoming: dict) -> int:
        """Merge collective knowledge from a peer."""
        new_count = 0
        records = incoming.get("collective_records", {})
        for key, record in records.items():
            if key not in self._collective_keys:
                self._collective_keys.add(key)
                self._collective_records[key] = record
                new_count += 1
        return new_count

    @property
    def size(self):
        return self.private.size

    @property
    def collective_size(self):
        return len(self._collective_keys)
