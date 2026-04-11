"""Mesh -- peer-to-peer gossip and state synchronization.

The mesh layer handles:
- Peer discovery via seed nodes
- Gossip-based state propagation
- Anti-entropy reconciliation using Merkle roots
- Failure detection via heartbeats
"""
