# Agent46

An insurance assistant (Alex, for SecureLife Insurance) built from the Atlas Agent Engine
Insurance Agent template, extended into a **self-improving harness**: its risk rules, thresholds
and keywords are versioned data that a proposer can change through a fixed evaluator.

The philosophy and a playbook for recreating this for a new problem are in
[`../../HANDOFF.md`](../../HANDOFF.md). This file covers setup and day-to-day commands.

**Features:** Long-term memory · Human-in-the-loop (suspend/resume) · Versioned harness manifest ·
Offline evaluator and evolution loop

## What is in here

- A runnable SDK app with policy, quote and claim tools, backed by MongoDB
- Long-term memory (`features.memory: true` in `agent.yaml`), configured at project level in
  `../../project-config.yaml`
- `harness_config.py`: the versioned manifest store and the invariants a manifest must satisfy
- `risk.py`: the pure risk function the evaluator scores
- `tool_access.py` and `tool_guard.py`: per-intent tool allow-lists and their enforcement
- `evals/`: the fixed evaluators for risk and for tool access (dev and held-out scenarios, scorer, gate)
- `harness_evolve/`: the proposer, patch language, evolution cycle and CLI. It does not ship in the
  deployed agent.

## Quick start

```bash
cp env.example .env
agentengine dev up
```

Before you start the app, edit `.env` and:

- Leave `MONGODB_URI` empty for `agentengine dev`; set it only when running against your own
  MongoDB deployment
- Set `LLM_API_KEY` for the configured client
- Set `VOYAGE_API_KEY` (memory embeddings; `features.memory` is on)
- Review the LLM builder in `src/agent_agent46/llm.py`. To switch providers, update the import,
  constructor and `MODEL`; the key remains `LLM_API_KEY`.

Memory extraction runs in a separate service. `../../project-config.yaml` points it at
`LLM_API_KEY` through `extraction_llm.api_key_secret`, so one key serves both the agent and
extraction. After editing that file run `agentengine memory configure` then
`agentengine memory apply`.

Once the stack is running:

- Open the playground at `http://localhost:3000` (port set in `dev.yaml`)
- `agentengine dev status` prints every service URL, including the MongoDB port
- `agentengine dev logs`, `agentengine dev stop` and `agentengine dev clean` manage the stack.
  Running `agentengine dev` alone only prints help; the start command is `dev up`.

To call the agent directly, a `user_id` is required or the runtime rejects the turn:

```bash
curl -X POST 'http://localhost:3000/api/v1/projects/<project-id>/workspaces/agent46/invoke' \
  -H 'Content-Type: application/json' \
  -d '{"message": "Hi", "user_id": "test-user-1"}'
```

The exact URL is printed by `agentengine dev up`.

## The harness

```bash
uv sync
uv run pytest -q                                                   # 291 tests
uv run python -m evals.run_eval --manifest baseline --split dev    # score risk decisions
uv run python -m evals.tool_eval --manifest baseline --split dev   # score tool access

uv run python -m harness_evolve.cli cycle        # propose, store a candidate, score, gate
uv run python -m harness_evolve.cli cycle --surface tools   # or: risk | all (default)
uv run python -m harness_evolve.cli status       # list manifest versions and gate results
uv run python -m harness_evolve.cli approve 2    # activate a candidate that passed the gate
uv run python -m harness_evolve.cli rollback 1   # activate an older version
```

The CLI talks to the local Docker MongoDB. From the host that needs `?directConnection=true`, and
the port is assigned per machine. Pass `--uri`, or export `EVOLVE_MONGODB_URI`. `env.example`
explains the host-side variables.

What may evolve, and what may not:

| Evolvable (manifest in MongoDB) | Fixed (code) |
|---|---|
| up to 20 learned prompt rules | invariants and their bounds |
| risk thresholds, inside bounds | the evaluator, scenarios and gate |
| risk keywords, added only | the claim state machine and `human_review` |
| per-intent tool allow-lists, narrowing only | the required-tool floor and the intent classifier |
| | memory label scoping, tool sandboxes, egress |

A patch that changes a threshold, removes a rule or widens tool access needs approval. A patch that
only adds a rule or keyword, or narrows tool access, is low risk. Nothing is activated automatically unless `--auto-activate-low-risk` is passed.

## Memory notes

- Every fact label is built by `user_scoped_label(user_id, key)`. The memory service enforces
  label uniqueness per org, so a shared label lets one user overwrite another's fact. A test fails
  if any `save_semantic` call bypasses the helper.
- The public memory client has no update or delete. Saving an existing label overwrites it in place.

## Register it on the platform

When you are happy with the local app and want to connect it to the platform:

```bash
agentengine init
agentengine build
agentengine deploy
```

`agentengine init` is where org and project selection and workspace registration happen. Secrets
stay in `.env`, runtime feature flags live in `agent.yaml`, and the model lives in the LLM builder.
Deployed agents invoked by a service account share one memory identity; read the memory notes in
`../../HANDOFF.md` before deploying.

## Project layout

```text
default/
├── HANDOFF.md
├── README.md
├── project-config.yaml
└── agents/agent46/
    ├── agent.yaml
    ├── dev.yaml
    ├── env.example
    ├── pyproject.toml
    ├── src/agent_agent46/
    │   ├── main.py              agent graph, tools, prompt
    │   ├── harness_config.py    manifest store, invariants, prompt rendering
    │   ├── risk.py              pure risk decision
    │   ├── tool_access.py       tool inventory, required floor, intent classifier, resolver
    │   ├── tool_guard.py        offers only allowed tools and blocks denied calls
    │   ├── memory_policy.py     user-scoped memory labels
    │   ├── policy_store.py      policy and claim storage, claim state machine
    │   └── llm.py
    ├── evals/                   fixed evaluator
    ├── harness_evolve/          proposer, patch language, cycle, CLI
    └── tests/
```
