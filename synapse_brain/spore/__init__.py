"""Spore -- the atomic unit of the Synapse Brain swarm.

A spore is a lightweight agent that:
- Receives task fragments from the mesh
- Reasons about them using any available LLM provider
- Produces reasoning deltas (hypotheses, evidence, conclusions)
- Gossips its CRDT state to peers
- Merges incoming state from peers

Spores are designed to run anywhere: Oracle Cloud VMs, HuggingFace Spaces,
Render free tier, Cloudflare Workers, or a Raspberry Pi in a closet.
"""
