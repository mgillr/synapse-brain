# Contributing to Synapse Brain

This is an experimental project exploring the limits of swarm intelligence
and never-forgetting memory. Contributions that push those boundaries are welcome.

## Ways to Contribute

### Run Your Own Swarm

The most valuable contribution right now is data. Deploy your own swarm,
run it for a while, and share what you observe:

- Memory growth rates over time
- Convergence quality at different spore counts
- Trust distribution evolution
- Failure modes you encounter
- Performance characteristics on different hardware

Open an issue with your findings.

### Code Contributions

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Run the test suite: `pytest tests/`
5. Open a PR with a clear description of what changed and why

### Areas of Interest

**Memory scaling** -- The OR-Set grows without bound. The semantic indexer
keeps retrieval fast, but we have not stress-tested beyond ~7K memories
per spore. What happens at 100K? 1M? Where does the curve bend?

**Provider diversity** -- Adding new free-tier LLM providers increases
model diversity. The provider interface in spore.py is simple to extend.
Each provider needs: an API call function, error handling, and rate-limit
detection.

**Evaluation harnesses** -- Rigorous measurement of convergence quality.
Does a 7-spore answer actually beat a single-model answer? By how much?
On which tasks?

**Reasoning protocols** -- Each spore gets a system prompt defining its
reasoning strategy. Alternative approaches could improve convergence
speed or quality.

**Sidecar optimization** -- The semantic indexing sidecar is a prime
optimization target. Faster indexing = faster retrieval = better context
per reasoning cycle.

**Federation** -- Multiple independent commanders running connected swarms.
The protocol exists but has not been tested across different operators.

## Code Style

- Python 3.10+
- Type hints where practical
- Docstrings on public functions
- No emoji in code or comments
- Descriptive variable names

## Architecture Notes

- `spore.py` is the core runtime. It is large (2,500+ lines) because
  a spore is a self-contained agent. Splitting it into modules is a
  valid contribution if done without breaking the single-file deployment
  model (HF Spaces expects one `app.py`).

- The CRDT layer comes from `crdt-merge` (pip package). Do not
  reimplement CRDT primitives -- use the package API.

- Memory is append-only by design. If you find yourself wanting to
  delete memories, reconsider the approach. The never-forgetting
  property is not a bug.

## Questions

Open an issue. There is no mailing list or Discord yet -- if the
community grows, we will set one up.
