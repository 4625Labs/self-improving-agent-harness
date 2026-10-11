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

## Raindrop signals into the court (advisory)

Raindrop is advisory: it can suggest claims for the court to look at, but only a human verdict creates a scenario.

1. In the Raindrop dashboard open app.raindrop.ai/signals and choose "New Signal".
2. Describe it as "claim approved despite fraud-like cues" and set the sentiment to Negative.
3. Export the matching events (including `claim_id`) to a JSON file, as a list or JSON lines.
4. Preview with `uv run python -m harness_evolve.cli court ingest-events events.json --dry-run`, then run it without `--dry-run`.

Each claim with a signal in the allow-list becomes a court case with source `raindrop_signal` (idempotent per claim) and an
audit row `court.ingest`. `step_cap_hit` and `tool_limit_hit` events are printed as `limit_pressure` for the limits surface, not
nominated, and `review_resumed` disagreements stay with `court scan`. Mutating runs against a non-local Mongo need `--allow-remote`.

Not automated: there is no Raindrop read API wired up. Ingestion is from an exported file, or from events pasted from the
Raindrop MCP into a file; nothing polls Raindrop.

## Copy manifest history to Atlas (atlas-sync)

Copies only the `harness_manifests` history (versions >= 2) from local dev Mongo to the Atlas Mongo, keeping Atlas's own
v1 and ending with exactly one active version. Set `ATLAS_EVOLVE_URI` and `ATLAS_EVOLVE_DB` in `agents/agent46/.env`.
Dry run is the default and writes nothing (no `--allow-remote` needed). No URI or password is ever printed.

```
cd agents/agent46
uv run python -m harness_evolve.cli atlas-sync
```

Review the plan, then run this yourself in the Claude Code prompt (the `! ` prefix runs it in your shell):

```
! cd agents/agent46 && uv run python -m harness_evolve.cli atlas-sync --apply --yes --by YOURNAME --allow-remote
```

It refuses (exit 2) on unknown target versions or conflicting manifests (`--force` overrides only those), is idempotent,
verifies the target after writing, and records one `atlas_sync.manifests` row in the target's `audit_log`.

## Judge demo

A simulated production incident stream that ends at a human approval in the Regression Court. Everything it creates is tagged
`demo` (claims `CLM-SIM-*`, policy `POL-SIM-DEMO`, their court cases, and the candidate version it stores). The adjuster verdict is
simulated and labelled so in the audit log (`Demo Adjuster (simulated)`). It uses the rule-based proposer only (no LLM) and never
approves, activates or rolls back a version: that is your click. It refuses to run while a candidate is already awaiting approval.

```
cd agents/agent46
uv run python -m harness_evolve.cli demo seed                       # 3 Nominated + 2 Filed cases, 10 claims
uv run python -m harness_evolve.court_ui --port 8765                # open http://127.0.0.1:8765 in a browser
uv run python -m harness_evolve.cli demo run --scenario parts-theft # narrated; --pace 2 by default
```

Then in the browser: Policy versions, find the new candidate, approve it. Afterwards:

```
uv run python -m harness_evolve.cli demo status
uv run python -m harness_evolve.cli demo reset --yes                # removes only demo data; rejects an unapproved demo candidate
```

Scenarios (each stores a candidate; the first three pass the gate, the last is a rejected-proposal demo):
`parts-theft` ('stripped', 'cannibalized'), `staged-accident` ('staged'), `cloned-vehicle` ('cloned'), `abandoned-lookalike`
('abandoned', rejected: it also catches legitimate claims that only mention the word). If the live policy is still the baseline the
proposal also carries the existing dev fixes (arson, submerged, hit and run, no witnesses, a threshold), so it needs approval.
Approve or reject a candidate before the next `demo run`.

The seed adapts to the live policy: each claim's stored recommendation comes from the real risk function under the current active
manifest, and if a claim's wording is already handled (for example `staged` is now a keyword) it switches to an alternative natural
wording that the live policy still misses. A claim with no such wording is reported as already fixed and replaced in the
Nominated/Filed set by another claim, so the 3 Nominated + 2 Filed cases are always real misses. `demo run` on a scenario whose
keyword is already live prints `already fixed by vN; choose another scenario: ...` and creates nothing. `demo status` labels each
case `real miss` or `already handled` under the live policy. If claims were seeded under an older policy, `demo reset --yes` then
`demo seed` refreshes them.

To point it at Atlas, run these yourself (`!` prefix in the Claude Code prompt) with `--uri` or `EVOLVE_MONGODB_URI` set to the Atlas
URI; mutating commands need `--allow-remote`, and so does the UI:

```
! cd agents/agent46 && uv run python -m harness_evolve.cli demo seed --allow-remote
! cd agents/agent46 && uv run python -m harness_evolve.court_ui --port 8765 --allow-remote
! cd agents/agent46 && uv run python -m harness_evolve.cli demo run --scenario staged-accident --allow-remote
! cd agents/agent46 && uv run python -m harness_evolve.cli demo reset --yes --allow-remote
```

`demo run` convicts the scenario's cases into `evals/scenarios/docket.json` (through the court, audited); `demo reset` retracts them.
