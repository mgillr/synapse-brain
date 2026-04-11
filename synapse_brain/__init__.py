"""Synapse Brain -- distributed reasoning swarm.

1000 cheap agents. One converged intelligence.
Built on crdt-merge for mathematically guaranteed convergence
and E4 for Byzantine fault detection.
"""

__version__ = "0.1.0"

from synapse_brain.spore.runtime import Spore, SporeConfig, ReasoningDelta
from synapse_brain.cortex.orchestrator import Orchestrator, Task
from synapse_brain.cortex.protocol import SporeRole, CognitivePhase

__all__ = [
    "Spore",
    "SporeConfig",
    "ReasoningDelta",
    "Orchestrator",
    "Task",
    "SporeRole",
    "CognitivePhase",
]
