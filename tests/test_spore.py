"""Tests for Spore runtime -- CRDT state, delta insertion, merge, trust."""

import pytest
from synapse_brain.spore.runtime import Spore, SporeConfig, ReasoningDelta


class TestReasoningDelta:
    def test_create_delta(self):
        delta = ReasoningDelta(
            spore_id="spore-001",
            task_id="task-abc",
            content={"reasoning": "The answer is 42"},
            confidence=0.85,
        )
        assert delta.spore_id == "spore-001"
        assert delta.task_id == "task-abc"
        assert delta.confidence == 0.85

    def test_fingerprint_deterministic(self):
        delta = ReasoningDelta(
            delta_id="fixed-id",
            spore_id="spore-001",
            task_id="task-abc",
            content={"reasoning": "test"},
            confidence=0.5,
            timestamp=1000.0,
        )
        fp1 = delta.fingerprint()
        fp2 = delta.fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 16

    def test_different_deltas_different_fingerprints(self):
        d1 = ReasoningDelta(delta_id="a", spore_id="s1", content={"x": 1}, timestamp=1.0)
        d2 = ReasoningDelta(delta_id="b", spore_id="s2", content={"x": 2}, timestamp=2.0)
        assert d1.fingerprint() != d2.fingerprint()

    def test_serialization_roundtrip(self):
        delta = ReasoningDelta(
            spore_id="spore-001",
            task_id="task-abc",
            content={"reasoning": "test data"},
            confidence=0.75,
        )
        data = delta.to_dict()
        restored = ReasoningDelta.from_dict(data)
        assert restored.spore_id == delta.spore_id
        assert restored.task_id == delta.task_id
        assert restored.confidence == delta.confidence


class TestSpore:
    def test_create_spore(self):
        config = SporeConfig(spore_id="test-spore-001")
        spore = Spore(config)
        assert spore.spore_id == "test-spore-001"

    def test_insert_delta(self):
        config = SporeConfig(spore_id="test-spore-002")
        spore = Spore(config)

        delta = ReasoningDelta(
            spore_id="test-spore-002",
            task_id="task-001",
            content={"reasoning": "The sky is blue"},
            confidence=0.9,
        )

        fp = spore.insert_delta(delta)
        assert len(fp) == 16
        assert fp in spore._delta_store

    def test_insert_increments_clock(self):
        config = SporeConfig(spore_id="test-spore-003")
        spore = Spore(config)

        initial_clock = spore._clock.logical_time

        delta = ReasoningDelta(
            spore_id="test-spore-003",
            task_id="task-001",
            content={"reasoning": "test"},
        )
        spore.insert_delta(delta)

        assert spore._clock.logical_time > initial_clock

    def test_merge_remote_delta(self):
        config = SporeConfig(spore_id="local-spore")
        spore = Spore(config)

        remote_delta = ReasoningDelta(
            spore_id="remote-spore",
            task_id="task-001",
            content={"reasoning": "Remote reasoning output"},
            confidence=0.7,
        )

        # Need to add the remote peer to trust lattice first
        spore._lattice = spore._lattice.__class__(
            "local-spore",
            initial_peers={"local-spore", "remote-spore"},
        )

        result = spore.merge_remote_delta(remote_delta, "remote-spore")
        assert result is True

    def test_merge_duplicate_rejected(self):
        config = SporeConfig(spore_id="local-spore")
        spore = Spore(config)

        spore._lattice = spore._lattice.__class__(
            "local-spore",
            initial_peers={"local-spore", "remote-spore"},
        )

        delta = ReasoningDelta(
            spore_id="remote-spore",
            task_id="task-001",
            content={"reasoning": "Same data"},
            confidence=0.6,
        )

        first = spore.merge_remote_delta(delta, "remote-spore")
        second = spore.merge_remote_delta(delta, "remote-spore")
        assert first is True
        assert second is False

    def test_get_task_consensus(self):
        config = SporeConfig(spore_id="consensus-spore")
        spore = Spore(config)

        for i in range(5):
            delta = ReasoningDelta(
                spore_id=f"spore-{i}",
                task_id="task-consensus",
                content={"reasoning": f"Hypothesis {i}"},
                confidence=0.1 * (i + 1),
            )
            spore.insert_delta(delta)

        consensus = spore.get_task_consensus("task-consensus")
        assert len(consensus) == 5
        # Sorted by confidence descending
        assert consensus[0].confidence >= consensus[-1].confidence

    def test_state_snapshot(self):
        config = SporeConfig(spore_id="snapshot-spore")
        spore = Spore(config)

        delta = ReasoningDelta(
            spore_id="snapshot-spore",
            task_id="task-snap",
            content={"reasoning": "test"},
        )
        spore.insert_delta(delta)

        snapshot = spore.state_snapshot()
        assert "spore_id" in snapshot
        assert "merkle_root" in snapshot
        assert "tasks" in snapshot
        assert "task-snap" in snapshot["tasks"]

    def test_metrics(self):
        config = SporeConfig(spore_id="metrics-spore")
        spore = Spore(config)

        metrics = spore.metrics()
        assert metrics["spore_id"] == "metrics-spore"
        assert "state" in metrics
        assert "clock" in metrics
        assert "merkle_root" in metrics


class TestSporeConfig:
    def test_default_config(self):
        config = SporeConfig()
        assert config.spore_id.startswith("spore-")
        assert config.listen_port == 8470
        assert config.cycle_interval == 30.0

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_SPORE_ID", "env-spore")
        monkeypatch.setenv("SYNAPSE_PORT", "9999")
        monkeypatch.setenv("SYNAPSE_MESH_SEEDS", "http://a:8470,http://b:8470")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        config = SporeConfig.from_env()
        assert config.spore_id == "env-spore"
        assert config.listen_port == 9999
        assert len(config.mesh_seeds) == 2
        assert "GROQ_API_KEY" in config.provider_keys


class TestCRDTConvergence:
    """Verify that two spores with the same deltas converge to identical state."""

    def test_two_spore_convergence(self):
        spore_a = Spore(SporeConfig(spore_id="spore-a"))
        spore_b = Spore(SporeConfig(spore_id="spore-b"))

        # Add trust between spores
        for s in [spore_a, spore_b]:
            s._lattice = s._lattice.__class__(
                s.spore_id, initial_peers={"spore-a", "spore-b"},
            )

        # Spore A produces a delta
        delta_a = ReasoningDelta(
            spore_id="spore-a", task_id="task-1",
            content={"reasoning": "A says yes"}, confidence=0.8,
        )
        spore_a.insert_delta(delta_a)

        # Spore B produces a delta
        delta_b = ReasoningDelta(
            spore_id="spore-b", task_id="task-1",
            content={"reasoning": "B says maybe"}, confidence=0.6,
        )
        spore_b.insert_delta(delta_b)

        # Gossip: A sends to B, B sends to A
        spore_b.merge_remote_delta(delta_a, "spore-a")
        spore_a.merge_remote_delta(delta_b, "spore-b")

        # Both spores should have same deltas
        consensus_a = spore_a.get_task_consensus("task-1")
        consensus_b = spore_b.get_task_consensus("task-1")

        assert len(consensus_a) == 2
        assert len(consensus_b) == 2

        # Same fingerprints in both
        fps_a = {d.fingerprint() for d in consensus_a}
        fps_b = {d.fingerprint() for d in consensus_b}
        assert fps_a == fps_b

    def test_three_spore_convergence_different_order(self):
        """Three spores receive deltas in different orders but converge."""
        spores = [Spore(SporeConfig(spore_id=f"spore-{i}")) for i in range(3)]
        ids = {f"spore-{i}" for i in range(3)}
        for s in spores:
            s._lattice = s._lattice.__class__(s.spore_id, initial_peers=ids)

        # Each spore produces a delta
        deltas = []
        for i, s in enumerate(spores):
            d = ReasoningDelta(
                spore_id=s.spore_id, task_id="task-conv",
                content={"reasoning": f"Answer {i}"}, confidence=0.5 + i * 0.1,
            )
            s.insert_delta(d)
            deltas.append(d)

        # Gossip in different orders
        # Spore 0 gets 1, then 2
        spores[0].merge_remote_delta(deltas[1], "spore-1")
        spores[0].merge_remote_delta(deltas[2], "spore-2")
        # Spore 1 gets 2, then 0
        spores[1].merge_remote_delta(deltas[2], "spore-2")
        spores[1].merge_remote_delta(deltas[0], "spore-0")
        # Spore 2 gets 0, then 1
        spores[2].merge_remote_delta(deltas[0], "spore-0")
        spores[2].merge_remote_delta(deltas[1], "spore-1")

        # All converge
        for s in spores:
            consensus = s.get_task_consensus("task-conv")
            assert len(consensus) == 3
