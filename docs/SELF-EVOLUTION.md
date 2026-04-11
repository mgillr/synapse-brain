# Self-Evolution Protocol

## Design Principle

Synapse Brain is a living system. It does not wait for external updates. It
continuously observes itself, identifies weaknesses, proposes improvements, and
tracks the effect of changes over time. Evolution is built into the architecture,
not bolted on.

## Three Evolution Mechanisms

### 1. Self-Analysis Tasks

The swarm can reason about its own architecture by submitting self-analysis tasks:

```json
{
  "task": "Analyze the current Synapse Brain architecture. What are the three highest-impact improvements that would increase convergence quality and reasoning depth?"
}
```

Each spore applies its full cognitive protocol to the analysis:
- The Explorer suggests novel architectural patterns
- The Adversarial identifies structural weaknesses
- The Synthesizer integrates proposals into actionable plans
- The Validator assesses feasibility and risk
- The Generalist grounds proposals in practical constraints
- The Brain provides meta-architectural oversight

The converged synthesis becomes a concrete improvement proposal, stored in the
knowledge store for future reference.

**Proven result**: The v3 swarm's first self-analysis produced 111 reasoning deltas
across 5 model families and independently recommended model diversity as the
highest-priority improvement -- before it had been implemented. The swarm
co-evolved with its own recommendations.

### 2. Temporal Self-Learning (TAI Pattern)

Each spore maintains a rolling window of operational metrics:

| Metric | Window | Purpose |
|--------|--------|---------|
| Cycle time | Last 100 cycles | Detect latency degradation |
| Error rate | Last 50 cycles | Detect reliability issues |
| Quality score | Last 30 syntheses | Detect reasoning quality drift |
| Convergence speed | Last 10 tasks | Detect synthesis efficiency changes |
| Trust trajectory | All time | Long-term reliability tracking |

The temporal baseline is computed as a rolling mean + standard deviation.
Any observation more than 2 standard deviations from the baseline triggers
an anomaly flag.

This is the TAI (Temporal Accumulating Intelligence) pattern: the system learns
what "normal" looks like by observing itself. No rules, no thresholds, no
external monitoring -- just temporal provenance.

**Key insight**: Anything that appears without temporal history is anomalous.
A spore that has been running for 30 days has 30 days of established patterns.
A sudden change in behavior stands out against that history without any
explicit rules.

### 3. Trust Recalibration

Trust scores evolve continuously based on actual contribution quality:

```
trust_delta = f(
    contribution_quality,     # How useful was this delta to synthesis?
    convergence_alignment,    # Did this delta move toward or away from consensus?
    novelty_value,           # Did this delta introduce genuinely new ideas?
    consistency,             # Is this spore reliable over time?
)
```

Trust affects synthesis weighting. A spore that consistently produces
high-quality contributions gains more influence. A spore that produces
noise or misleading claims loses influence gradually.

This creates a self-correcting feedback loop: better reasoning leads to
higher trust, which leads to more influence on synthesis, which improves
the collective output, which provides better feedback for trust calibration.

## Memory as Evolution Substrate

The CRDT memory store is the substrate for all evolution:

- **Reasoning store**: Every delta is permanent. The system can analyze its own
  reasoning history to identify patterns, recurring errors, and successful strategies.

- **Knowledge store**: Converged syntheses accumulate over time. Each new task
  benefits from all prior knowledge -- the system gets smarter with use.

- **Trust store**: Trust scores encode the collective's learned assessment of
  each spore's reliability. This persists across tasks and sessions.

- **Temporal store**: Self-observation data enables drift detection. The system
  knows when something has changed because it remembers what "before" looked like.

Because CRDT OR-Sets are add-only (nothing is ever deleted), the memory grows
monotonically. The sidecar indexer ensures retrieval remains fast regardless of
size by maintaining embedding indices that are updated incrementally.

## Evolution Cycle

```
[Reasoning task]
       |
       v
[Independent reasoning per spore]
       |
       v
[Convergence + synthesis]
       |
       v
[Knowledge store updated]
       |
       v
[Quality metrics recorded]
       |
       v
[Temporal baseline updated]
       |
       v
[Trust scores recalibrated]
       |
       v
[Next task benefits from all accumulated knowledge + trust + temporal awareness]
```

Each cycle through this loop makes the system marginally better. Over hundreds
of tasks, the improvement compounds.

## Guardrails

Evolution is not unconstrained:

1. **Trust floor**: No spore's trust can drop below a minimum threshold. This
   prevents a single bad interaction from permanently excluding a perspective.

2. **Diversity protection**: The system monitors claim diversity during DIVERGE
   phase. If all spores are converging too quickly (premature consensus), the
   phase is extended.

3. **Knowledge promotion gate**: Syntheses must pass Validator review before
   entering the knowledge store. Low-quality syntheses are flagged, not promoted.

4. **Temporal anomaly response**: Anomaly flags trigger increased logging and
   peer validation, not automatic action. The system investigates before reacting.

## Measuring Evolution

Track these metrics to observe the system evolving:

- **Convergence speed**: Tasks should reach synthesis faster as the knowledge
  store grows (more relevant prior reasoning to draw from)
- **Synthesis quality**: Validator approval rate should increase over time
- **Trust stability**: Trust scores should stabilize as spores establish track records
- **Cross-pollination rate**: Knowledge from one task should appear in reasoning
  about unrelated tasks (measured by semantic similarity of retrieved context)
- **Anomaly rate**: Should decrease as the temporal baseline becomes more robust
