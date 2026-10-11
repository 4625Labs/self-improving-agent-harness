# Self-improving agent harness

A working playground for an agent whose rules, risk thresholds and keywords live as versioned
data in MongoDB. A fixed evaluator tests every proposed change. Humans own the limits and approve
risky changes.

The worked example is an auto-insurance claims agent built on the MongoDB Atlas Agent Engine with
LangGraph. The point is the pattern, which is meant to be rebuilt for a different problem.

**Start here:** [HANDOFF.md](HANDOFF.md) explains the philosophy, the principles, and a step-by-step
playbook for recreating this for a new problem statement.

## The idea in one paragraph

An agent is fixed machinery plus policy. Only the policy evolves, only as data, only through an
evaluator the proposer cannot edit, and only inside invariants humans own. The model may propose,
the evaluator decides, a human owns the limits.

## What it did

| | Baseline | Evolved |
|---|---|---|
| Dev accuracy | 0.68 | 0.88 |
| Held-out accuracy | 0.40 | 0.70 |
| Critical violations (dev) | 1 | 0 |
| Tool-access accuracy (dev / held-out) | 0.26 / 0.29 | 1.00 / 1.00 |

On the live agent, an arson claim was auto-approved under the baseline and sent to human review
under the evolved manifest. With the evolved tool access the agent offered 10 of 17 tools for
information and quote requests, and a full claim lifecycle still completed.

The eval sets are small, so read the gains as directional. The proposers used here are offline
heuristics; the LLM proposer is written but has not been run.

## Layout

```
.
├── HANDOFF.md            philosophy and recreation playbook
├── project-config.yaml   project-level memory settings
└── agents/agent46/       the workspace: agent, evaluator, proposer, tests
```

See [agents/agent46/README.md](agents/agent46/README.md) for setup and commands.

## Quick start

```bash
cd agents/agent46
cp env.example .env          # set LLM_API_KEY and VOYAGE_API_KEY
agentengine dev up           # starts the local Docker stack, including MongoDB
uv sync && uv run pytest -q
```

Requires the `agentengine` CLI, Docker, and `uv`.

## Status

Built and tested locally (291 tests): the versioned manifest, the pure risk function, fixed
evaluators for risk and for tool access, the proposer and patch language, the approve and rollback
cycle, per-intent tool access enforced at two points in the agent graph, and the memory label
invariant. Details and known limits are in HANDOFF.md.
