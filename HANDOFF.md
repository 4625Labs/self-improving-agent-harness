# Hand-off: building a self-improving agent harness

This document is for the next person, or the next session, who wants to rebuild this
architecture for a different problem. It starts with the philosophy because the philosophy is
what transfers. The insurance agent here is only the worked example.

Everything marked **verified** was run on a local Docker stack. Anything else is labeled as
designed, untested, or unknown. Section 6 starts with a five-minute way to check the headline
numbers yourself.

Repository: `4625Labs/self-improving-agent-harness` (private). The first build (risk and tool access)
is described by commit `13f3323`. A second round of work added Raindrop observability, the
Regression Court, a third evolvable surface (agent limits), an audit log and the guardrails; it is
described in sections 5 and 9 and is in the commits after `4eecead` on `main`. A commit cannot cite
its own hash, so use `git log` for the newest one. Test count at the time of writing: **536**.

---

## 1. The thesis

An agent's behavior is **fixed machinery plus policy**. A self-improving harness rewrites only
the policy, only as data, only through a fixed evaluator, and only inside limits that humans
own.

```
            evolvable (data, in MongoDB)                fixed (code, owned by humans)
        ┌──────────────────────────────────┐        ┌───────────────────────────────────┐
        │ rules · thresholds · keywords    │        │ invariants · evaluator · gate     │
        │ tool access (per intent)         │        │ required-tool floor · classifier  │
        │ versioned manifest               │        │ control flow · human-review step  │
        └───────────────┬──────────────────┘        │ sandboxes · egress                │
                        │ read at runtime           └───────────────────────────────────┘
   live traffic ────────▼────────┐                                    ▲ checks every change
   decisions tagged with version │                                    │
                        ┌────────▼─────────┐   failures   ┌───────────┴───────────┐
                        │  runtime agent   │─────────────▶│ collect → propose →   │
                        └──────────────────┘              │ patch → evaluate →    │
                                  ▲   activate / rollback │ gate → promote        │
                                  └───────────────────────┴───────────────────────┘
```

If you remember one sentence: **the model may propose, the evaluator decides, a human owns the
limits.**

---

## 2. The principles

Each principle says what it means, why it exists, where it lives in this repo, and the failure
that tells you it was broken.

1. **Evolve policy, never control flow.** The proposer changes thresholds, rules, keywords and tool
   allow-lists. It never changes the claim state machine, the human-review interrupt, which tools
   exist in the sandboxes, or egress.
   *Why:* policy can be validated by numbers; control flow needs a human reading code.
   *Broken when:* a proposal needs a code change to take effect.

2. **The evaluator is the product. Build it first.** Ground truth must come from what a competent
   human wants, not from what the system does today, so the baseline fails some scenarios. If the
   baseline scores 100%, there is nothing to improve toward. (`evals/`; baseline dev 0.68,
   held-out 0.40.)

3. **Make the decision a pure function of (input, manifest).** `risk.assess_claim_risk` has no I/O.
   That is what lets the evaluator run in milliseconds with no LLM. *Broken when:* you cannot
   score a decision without calling the whole agent. (Section 7 shows where this gets hard.)

4. **Give the proposer a closed language.** It emits whitelisted operations (`add_rule`,
   `remove_rule`, `set_threshold`, `add_keyword`, `disable_tool`, `enable_tool`), never code or
   free-form edits.
   (`harness_evolve/patch.py`.) The result must still pass the invariants.

5. **Humans own the limits, and they are checked in more than one place.** Invariants run when a
   manifest is proposed, when it is activated, and when it is loaded. A stored manifest that fails
   at load falls back to the baseline. (`harness_config.validate_manifest`, `INVARIANTS`.)

6. **Gate on several things at once.** A candidate must not add a critical violation, lose
   accuracy, or raise over-escalation or under-escalation on any split. Our first gate checked
   only accuracy and would have passed a candidate that sent more claims to reviewers.
   (`evals/run_eval.compare` for risk, `evals/tool_eval.compare_tools` for tool access. A cycle must
   pass the gate of every surface its patch touches.)

7. **Held-out data is for the gate, never the proposer.** The proposer reads dev failures only.
   Known leak: held-out failure ids are stored in the scores document, so anyone with database
   access can read them.

8. **Version and attribute every decision.** Each claim records the `harness_version` that decided
   it. That is what makes an A/B comparison and a rollback meaningful.

9. **Tier approval by risk.** Adding a rule or keyword is low risk. Changing a threshold or
   removing a rule needs a human. Auto-activation is opt-in and applies to low-risk patches only.
   Tools follow the same rule: narrowing (`disable_tool`) is low risk, widening (`enable_tool`)
   needs approval.

10. **Bound every open field.** An unbounded `add_keyword` could add "door". The invariants cap
    keyword count, length and characters, and the manifest rejects any field it does not know.

11. **Test the guards, not only the features.** Our first static check for memory labels passed
    even after the label was replaced with an unscoped string, because it checked that the helper was
    called somewhere, not that its result was used. A negative control caught it. Break the thing on
    purpose and confirm the test fails. The scenario honesty check did the same job for the intent
    classifier: tightening keyword matching stopped "vandalized" from counting as a claim, and the
    check caught it before anything shipped.

12. **Never evolve user data.** The harness may change policy around memory (what to save, which
    sources to read). It must not rewrite what a user's memory contains. Memory labels are
    user-scoped by an invariant, because the memory service enforces uniqueness per org.

13. **Prove it on the live path.** Unit tests do not show the manifest is in use. We sent the same
    kind of claim before and after approving a version and read the version tag back.

14. **Enforce at more than one point, from one resolver.** Tool access is applied when the model
    is offered its tools and again before a tool runs, because a model can name a tool it was never
    offered. The agent, the runtime guard and the evaluator all call `resolve_tool_access`, so they
    cannot disagree. (`tool_access.py`, `tool_guard.py`.)

15. **Ambiguity must fail open, and a fixed floor must protect workflows.** An unclear intent maps
    to `general`, whose tools can never be narrowed. A required-tool floor per intent, plus a
    claim-in-flight rule, means a misclassified message can never strand a half-finished claim.

16. **Fail safe, and fail loudly.** If the store is unreachable the agent runs the baseline.
    Known gap: that fallback is currently silent apart from a log line.

17. **Observability nominates, a human convicts, the evaluator decides.** Raindrop signals and
    reviewer disagreements only nominate cases. A nominated case becomes a test scenario only after a
    named adjuster rules on it. An LLM-judged signal never decides the gate.
    *Broken when:* an unreviewed production event changes what the evaluator scores.
    (`harness_evolve/court.py`.)

18. **Read the diff, not only the gate.** The first LLM-proposed candidate (v4) passed the four-part
    gate and still loosened two thresholds, one of which contradicted an adjuster's verdict. The gate
    measures what the scenarios cover. A human reads what the patch actually changes.

19. **Replay the world as it was.** A scenario rebuilt from today's data can silently differ from the
    original decision (a later claim on the same policy changed the risk score, so the missed
    fraud was no longer auto-approved on replay). Check that the stored decision and the replay agree
    before convicting a case.

20. **Destructive actions need a confirmation and leave a record.** Approve, reject, rollback,
    verdict, dismiss, overturn and retract require an explicit confirmation and write an append-only
    audit row, including refusals. Mutating commands refuse a non-local database unless told otherwise.
    (`harness_evolve/audit.py`.)

21. **Limits can evolve only inside human-owned bounds, and raising them is the risky direction.**
    The step and tool-call caps are manifest data with fixed floors and ceilings. Lowering a limit is
    low risk. Raising one needs approval and is never auto-activated. A limit that never resets must
    not strand a half-finished claim, so claim-completing tools are exempt from the per-thread cap.
    (`tool_limits.py`, `evals/limits_eval.py`.)

22. **Tests must not read live state.** The real court docket holds an adjuster's convictions. Three
    tests started failing when it changed. `tests/conftest.py` now points every test at an empty
    docket. Any file that tests and humans both touch needs this.

---

## 3. Recreating this for a new problem

### 3.1 Fill in the problem statement first

Do not write code until these are answered in a paragraph each.

| Question | Insurance example | Your answer |
|---|---|---|
| Which decision do you want to improve? | route a claim to auto-approval or human review | |
| What does a wrong decision cost, and in which direction? | auto-approving fraud is worse than a needless review | |
| Which 1 to 3 things may evolve first? | rules, risk thresholds, risk keywords (tool access came second) | |
| What must never change? | claim state machine, human review, egress | |
| Who knows the right answer, and where is it written down? | an adjuster's labeled cases | |
| Which outcomes must never happen (critical cases)? | auto-approving arson, theft, over-limit claims | |
| What are the metrics and the gate? | accuracy, both escalation rates, zero critical | |
| Which changes need a human? | any threshold change, any removal | |
| What is user data, and how is it kept separate? | customer facts in memory, label-scoped per user | |

Start with one surface. The second surface here, tool access, took as much work as the first
because it needed its own evaluator, its own human-owned floor and its own enforcement.

### 3.2 Is the problem a good fit?

A good fit has a decision that can be **checked offline**: you can write down the right answer for
a scenario without running the whole agent. Warning signs:

- Only the final conversation reveals whether the decision was right. You will need trajectory
  evals, which are slower and often need an LLM judge. Budget for that.
- Ground truth is a matter of taste. The gate will be arguing with itself.
- There is no deterministic core to extract. You will end up evolving prompts only, which is
  weaker than evolving parameters.

### 3.3 The build order

Each step ends with something you can run. Do not reorder them.

| # | Step | Done when |
|---|---|---|
| 0 | Write the problem statement and the red lines (3.1) | both lists exist |
| 1 | Extract the manifest into versioned storage with a baseline fallback | `pytest tests/test_harness_config.py` passes against a real database |
| 2 | Pull the decision into a pure function and tag every decision with the version | characterization tests reproduce today's behavior exactly |
| 3 | Build the fixed evaluator: scenarios, critical flags, a held-out split, a gate | the baseline fails some scenarios and the gate can say "fail" |
| 4 | Add a proposer behind a whitelist, offline heuristic first, LLM second | an unknown operation is rejected |
| 5 | Add the cycle: propose, store candidate, score, gate, approve, roll back | `cycle`, `status`, `approve N`, `rollback N` all work |
| 6 | Prove it on the live agent | a stored decision carries the new version and the behavior differs |
| 7 | Add a second surface (here: tool access) | it has its own evaluator, human-owned floor and enforcement, and its scenarios existed before its code |
| 8 | Harden what you built | you have tried to break each guard on purpose, and the scenarios are less regular than the first version |

### 3.4 What to swap and what to keep

| Keep as is (the engine) | Swap per problem (the domain) |
|---|---|
| `HarnessConfigStore`: versioning, candidate/active/retired/rejected, activate, fallback | the manifest schema and `DEFAULT_MANIFEST` |
| `validate_manifest` structure: bounds, unknown-field rejection, caps | the actual bounds and caps in `INVARIANTS` |
| `evals/run_eval.py`: `evaluate`, `compare`, CLI | scenarios (`dev.json`, `heldout.json`, `tools_dev.json`, `tools_heldout.json`) and the metric names |
| `harness_evolve/loop.py`: cycle, approve, rollback, score recording | the operations in `patch.py` and their risk tiers |
| `harness_evolve/proposer.py`: interface, strict LLM parsing | the heuristic proposer's lexicon |
| `memory_policy.py`: user-scoped labels | nothing, unless your memory model differs |
| `tool_guard.py`: offer filtered tools, block denied calls | `tool_access.py`: tool inventory, required floor, intent classifier |
| | the pure decision function (`risk.py`) |

### 3.5 A hypothetical second use case (illustration only, not built)

*Support-ticket triage.* Decision: close automatically, route to a queue, or escalate to a human.
Manifest: routing rules, an escalation threshold on a sentiment or urgency score, trigger phrases.
Critical cases: a ticket mentioning legal action, security or a safety issue must never close
automatically. Ground truth: historical tickets labeled by senior agents. Red lines: PII redaction
and the human-handoff step. The pure function takes a ticket and a manifest and returns a route.
Everything in 3.3 then applies unchanged.

---

## 4. What is in this repository

```
.
├── HANDOFF.md                this file
├── README.md                 short overview and quick start
├── project-config.yaml       project-level memory settings (extraction, Voyage, snapshots)
└── agents/agent46/           the workspace
    ├── agent.yaml            features, sandboxes, tool placement, egress
    ├── dev.yaml              local stack ports
    ├── env.example           copy to .env; documents host-side variables too
    ├── src/agent_agent46/
    │   ├── main.py           LangGraph agent, tools, prompt; loads the manifest per request
    │   ├── harness_config.py manifest store, invariants, validation, prompt rendering
    │   ├── risk.py           the pure decision function
    │   ├── tool_access.py    tool inventory, required-tool floor, intent classifier, resolver
    │   ├── tool_guard.py     runtime enforcement: filters offered tools, blocks denied calls
    │   ├── memory_policy.py  user-scoped memory labels
    │   ├── policy_store.py   policy and claim storage, claim state machine
    │   └── llm.py            model selection
    ├── evals/                FIXED evaluators: risk (run_eval) and tool access (tool_eval)
    ├── harness_evolve/       proposer, patch language, cycle, CLI (never ships in the agent)
    └── tests/                536 tests
```

`.claude/skills` and `.agents/skills` are the Agent Engine template's skill docs, kept as shipped.

Added in the second round (all under `agents/agent46`):

```
src/agent_agent46/observability.py   Raindrop events; no-op without RAINDROP_WRITE_KEY
src/agent_agent46/tool_limits.py     code-owned caps, resolve_limits, trace_from_messages
harness_evolve/court.py              docket: nominate, file, verdict, overturn, retract, dismiss
harness_evolve/court_ui/             localhost web docket (FastAPI), host-side only
harness_evolve/audit.py              append-only audit log, confirmation guard, remote-host guard
harness_evolve/budget.py             daily LLM call and token ledger
evals/limits_eval.py                 third evaluator (24 dev, 13 held-out scenarios)
evals/scenarios/docket.json          court convictions; merged into dev only (user data, untracked)
tests/conftest.py                    isolates every test from the real docket
harness_evolve/raindrop_source.py    exported Raindrop events -> advisory nominations
harness_evolve/atlas_sync.py         copy manifests local -> Atlas (dry-run by default, masks URIs)
```

---

## 5. State of the build

**Built and verified**

- Versioned manifest in MongoDB, with invariants checked at propose, activate and load.
- A pure risk function with 22 characterization tests.
- A fixed evaluator: 25 dev and 10 held-out scenarios, 11 of them critical, and a four-part gate.
- Tool access as an evolvable surface: per-intent allow-lists, a human-owned required-tool floor,
  a fixed intent classifier, a second evaluator (31 dev and 14 held-out scenarios), `disable_tool` and
  `enable_tool` patch operations, and enforcement at two points in the agent graph.
- An offline heuristic proposer, the patch language, and the cycle with approve and rollback.
- Memory label invariant, enforced by a source-parsing test that was itself negative-tested.
- Live A/B on the running agent.

Manifest history in the author's local MongoDB: v1 is the baseline (retired), v2 added risk keywords
and a threshold change (retired), v3 narrowed tool access (active). The first table below compares
v1 and v2. The tool-access table compares the baseline and v3.

| | Baseline (v1) | Evolved (v2) |
|---|---|---|
| Dev accuracy | 0.68 | 0.88 |
| Held-out accuracy | 0.40 | 0.70 |
| Critical violations (dev) | 1 | 0 |
| Live arson claim (a $750 claim under v1, a $700 claim under v2) | low risk, auto-approved | high risk, sent to human review |

**Tool access, verified live:** with manifest v3 active the agent offered 10 of 17 tools for
`info` and `quote` and 14 of 17 for `claims`. A claim was filed, suspended for human review,
resumed by the reviewer and closed as approved with no guard denials. The denial path itself (a
model naming a tool it was not offered) is proven offline with a fake model, not live.

| Tool-access eval | Baseline (all tools allowed) | Evolved (v3) |
|---|---|---|
| Dev accuracy | 0.26 | 1.00 |
| Held-out accuracy | 0.29 | 1.00 |
| Critical violations (dev / held-out) | 23 / 10 | 0 / 0 |

**The LLM proposer has now run twice, on purpose.** The first call returned no JSON: the 600-token
output cap was probably spent on hidden reasoning (unconfirmed, because the reply was not kept; it is now
saved to `.evolve_cache/last_failed_reply.txt`). The second call, with `--max-tokens 2000`, produced
candidate v4 (see section 9). Budget: one call per cycle, a capped prompt, a disk cache keyed by the
prompt, a per-process limit and a daily ledger (10 calls, 50,000 tokens).

**Built in the second round**

- **Raindrop observability.** Events for every turn, risk assessment, review suspend and resume, tool
  refusal, limit hit and manifest fallback, each tagged with `harness_version`. Off without a key.
  Conversation text leaves the process only with `RAINDROP_SEND_CONTENT=true`; claim description
  text is never sent, only its length. Verified live: events arrive in Raindrop.
- **Regression Court** (section 9): nominate, file, human verdict, scenario in the dev split, cycle,
  gate, approve. A web docket and a CLI drive the same code. Overturn and retract are audited.
- **Limits as a third evolvable surface.** Steps per turn (baseline 10, bounds 4 to 20) and tool calls
  per turn (baseline 12, bounds 6 to 30). Baseline limits eval: dev 0.667 (8 false caps), held-out
  0.615, 0 missed caps. The heuristic proposer raises them to 14 and 18, which scores 1.0 on both and
  passes the gate; a raise to 20 and 30 lets runaway traces through and fails it.
- **Code-owned guardrails** that do not evolve: identical-call limit (3), per-thread cap (60, with a
  claim-completing exemption), argument sanity, and the agent hand-off when the step cap is hit.
- **Audit log, confirmations and a remote-host guard** for every destructive action.
- **Platform guardrails enabled** (`features.guardrails: true`). Two starter guardrails (PII in
  replies, prompt-injection phrases) exist as log-only and inactive. They are a separate layer from the
  code limits: platform guardrails match text patterns at the LLM input and output only. Not yet
  verified that an active guardrail fires.
- **Held-out leak fixed.** Held-out failure ids are no longer stored in scores or reports.

**Known limits**

- Tool access is chosen by a keyword classifier on the latest user message. Ambiguity fails open to
  all tools, so a user who mixes intent words widens their own access. It protects against model
  mistakes, not against someone controlling the wording.
- The tool scenarios are regular: held-out reached 1.00 as easily as dev. Treat that as a sign the
  set is too easy, not as proof of generalisation. About 85% of dev scenarios are tagged critical,
  so the critical count is nearly a blanket label.
- A denied tool call is just an error message to the model. It may retry or answer wrongly, and the
  system prompt does not mention restrictions.
- A cycle that touches only one surface skips the gate for the other surface (marked `untouched`),
  because the risk gate fails on any critical violation even when the baseline already had one.
- The two gates do not use the same rule. The risk gate fails on any critical violation in the
  candidate. The tool gate fails only when critical violations rise against the baseline. Pick one
  rule before adding a third surface.
- A claim in flight is inferred from tool-call order: a `file_claim` with no later
  `send_notification`. A claim whose notification never fires keeps the claim-flow tools exposed in
  every intent for that thread.
- A disable-only tool patch is low risk and can auto-activate with `--auto-activate-low-risk`. The
  first one disabled 17 tools at once. Consider requiring approval until the tool scenarios grow.
- `enable_tool` is never proposed, so under-privilege failures are never acted on. A patch that
  fails to apply is rejected without a stored record, so there is no audit trail for it.
- The heuristic proposer only finds what its fixed phrase list contains. It demonstrates the loop,
  not intelligence.
- The eval sets are small. A gain of 0.3 is a few scenarios.
- Some failures cannot be fixed through the manifest. Descriptions like "no fire involved" match a
  keyword by substring, and only a code change (negation or word boundaries) fixes that.
- The cross-user memory overwrite and the empty taxonomic extractions were observed on the local
  stack only. Check the hosted service before relying on either finding.
- **Found in the second round:**
  - The court builds a case file from today's data. If a later claim exists on the same policy the
    replay can differ from the original decision. It should count only claims created before the one
    being filed. Not fixed.
  - `cycle --auto-activate-low-risk` activates without `--yes` and without an audit row.
  - Platform guardrails are not tied to the manifest, so the harness cannot evolve them.
  - The playground runs as `local-dev-user`, not the seeded test user, so it creates its own policy.
    `agentengine dev` fallbacks and the memory identity caveat in section 6 still apply when deployed.
  - The court UI has no login. Anyone who can reach localhost:8765 can type any adjuster name.
  - "Negation" is still unhandled in `risk.py`: "no significant damage" matches the keyword.
  - Raindrop's LLM-judged issues have not been exported into the court automatically; nomination from
    Raindrop is manual (`court nominate <claim_id>`).

---

## 6. Verify it, operate it, and the gotchas that cost time

### Verify the build in five minutes

Run these from `agents/agent46` after `uv sync`. The baseline scores come from the built-in
baseline manifest, so they do not depend on your database.

| Command | Expect |
|---|---|
| `uv run pytest -q` | 536 passed. The tests that need the Docker stack skip themselves when it is not running. |
| `uv run python -m evals.run_eval --manifest baseline --split dev` | accuracy 0.68, 1 critical violation |
| `uv run python -m evals.run_eval --manifest baseline --split heldout` | accuracy 0.40, 0 critical violations |
| `uv run python -m evals.tool_eval --manifest baseline --split dev` | accuracy 0.258, 23 critical violations |
| `uv run python -m evals.tool_eval --manifest baseline --split heldout` | accuracy 0.286, 10 critical violations |
| `uv run python -m harness_evolve.cli status` | with the Docker stack up, the stored manifest versions and their gate results |

The live results in section 5 need the stack running and a real model key, so they are not part of
this list.

### Commands

```bash
cd agents/agent46
cp env.example .env                     # then fill in LLM_API_KEY and VOYAGE_API_KEY
agentengine dev up                      # NOT `agentengine dev`: that only prints help
agentengine dev status                  # shows the mongo port and the playground URL

uv sync
uv run pytest -q
uv run python -m evals.run_eval --manifest baseline --split dev
uv run python -m evals.tool_eval --manifest baseline --split dev
uv run python -m harness_evolve.cli cycle --surface all     # or: risk | tools
uv run python -m harness_evolve.cli status
uv run python -m harness_evolve.cli approve 2
uv run python -m harness_evolve.cli rollback 1
```

### Second-round commands

```bash
cd agents/agent46
export EVOLVE_MONGODB_URI="mongodb://localhost:<port>/?directConnection=true"   # port: agentengine dev status

uv run python -m harness_evolve.court_ui --port 8765          # the docket, localhost only
uv run python -m harness_evolve.cli court nominate CLM-1 --reason "..."
uv run python -m harness_evolve.cli court scan                 # reviewer-disagreement nominations
uv run python -m harness_evolve.cli court file <case>
uv run python -m harness_evolve.cli --by NAME court verdict <case> manual_review_required --critical --yes
uv run python -m harness_evolve.cli --by NAME court overturn <case> auto_approve --reason "..." --yes
uv run python -m harness_evolve.cli --by NAME court retract <case> --reason "..." --yes
uv run python -m harness_evolve.cli --surface risk cycle                    # free heuristic proposer
uv run python -m harness_evolve.cli --proposer llm --dry-run cycle          # prompt + token estimate, no call
uv run python -m harness_evolve.cli --proposer llm --max-tokens 2000 --surface risk cycle
uv run python -m harness_evolve.cli budget                     # daily LLM budget left
uv run python -m harness_evolve.cli --by NAME approve 5 --yes   # also reject V / rollback V
uv run python -m harness_evolve.cli audit --n 20
uv run python -m evals.limits_eval --manifest baseline --split dev
uv run python -m harness_evolve.cli court ingest-events events.json --dry-run   # Raindrop export -> nominations
uv run python -m harness_evolve.cli atlas-sync                                  # dry run, writes nothing
# the operator runs the write: atlas-sync --apply --yes --by NAME --allow-remote
```

The verify-in-five-minutes numbers above hold only with an empty docket. Run them with
`EVOLVE_DOCKET_PATH=/nonexistent.json`; the test suite does this for you.

### Gotchas from the second round

- **Hot reload does not re-read egress or env.** A new egress rule in `agent.yaml` and a new
  `.env` value need `agentengine dev up` again. The proxy returns `403 Forbidden` for a domain not
  on the list. Raindrop needs `api.raindrop.ai:443`.
- **A new dependency needs the container to sync.** The 7-day cooldown in `uv.toml` can hold
  `opentelemetry-instrumentation` back while `opentelemetry-instrumentation-logging` floats ahead, and
  `import raindrop.analytics` then fails. `pyproject.toml` pins the logging package to match.
- **`EVOLVE_DOCKET_PATH` must be honoured by the writer as well as the reader.** It was not, and an
  isolated QA run wrote into the real docket. Fixed, with a regression test.
- **Raindrop rejects an event whose input and output are both empty.**
- **Shell checks of the container need its own interpreter:** `/tmp/agentic-venvs/.venv-aer-tool/bin/python`.
- **A nominated claim id cannot be re-nominated** after its case is dismissed. Give the corrected
  scenario a new claim id.
- **The app logs the human-review pause (`GraphInterrupt`) at ERROR.** It is not a failure.
- **A platform Policy Engine can deny your tools in the cloud and not locally.** An `AUTHORIZED_TOOLS`
  allowlist (project or organization scope; Manage > Policies) makes the Orchestration Engine halt a run
  with `POLICY_DENIED` the first time the agent calls a tool the list omits. Local dev does not enforce it.
  Symptom: the playground trace shows BLOCKED on memory and policy tools, the logs show
  `AUTHORIZED_TOOLS policy denied tool "<name>"`. An organization policy cannot be widened from the
  project. The same engine has run caps (`MAX_TOOL_CALLS_PER_EXECUTION` and others) that overlap with the
  evolvable limits in section 5. This is not caused by `features.guardrails`, which is content guardrails.
- **Platform guardrails (content): what was learned.** Created through the admin form or `POST /guardrails`
  with type `output_validation`. Verified in the cloud on 2026-10-11: an input-stage guardrail with action
  `block` stops a prompt-injection message before the agent runs ("Execution was denied by policy",
  `error_code: policy_denied`); an output-stage guardrail with action `modify` and replacement text rewrites
  a reply containing an SSN-like number (the reply became `[BLOCKED]`). Gotchas: a `log_only` guardrail never
  increments `trigger_count` and leaves no trace, so it cannot be demonstrated; a guardrail saved as
  inactive does nothing; the cloud has its own guardrail store, separate from the local one, so create each
  guardrail in both. The SDK evaluator reads `config.regex_patterns` while the form writes `match_patterns`;
  the platform translates between them (a form-only guardrail still blocked). `validate_llm_response` in the
  agent is a pass-through in this SDK version. Raindrop does not receive guardrail events; they are visible
  in the platform's Traces under the Guardrails filter.
- **Atlas from the host can fail transiently.** A first connection saw only one of three nodes
  (`ReplicaSetNoPrimary`); a retry minutes later reached the primary. Retry before changing anything.
- **A permission classifier may refuse writes to shared resources** (a cloud secret that sends
  conversation text out, writes to the Atlas cluster). Treat the refusal as a stop: build a reviewed
  command with a dry-run default and let the operator run the write.
- **Mongo from the host needs `?directConnection=true`.** The Docker Mongo advertises an internal
  hostname, so a plain `mongodb://localhost:<port>` times out. The port is assigned per machine.
- **Invoking the agent needs a `user_id`.** Without one the runtime fails with
  `durable Memory identity is incomplete`.
- **Importing `main.py` outside the runtime needs `RUNNER_MODE=aer`.** `build_agent(llm=fake)` then
  runs offline, which is how the enforcement tests drive the real graph.
- **Resuming a suspended claim over HTTP** takes the original `session_id` and `user_id` plus
  `{"resume_map": {"<interrupt id>": {"decision": "approved", "reviewer_notes": "..."}}}`.
- **A fail-open guard can hide bugs.** A missing import in `tool_guard.py` would have been swallowed
  and silently restored full access. Tests must exercise the resolver, and ERROR logs need an alert.
- **Memory needs the two-level layout:** `project-config.yaml` at the repo root, the workspace
  beneath it.
- **Memory labels are unique per org.** A second user saving the same label with the default
  `upsert=True` replaces the first user's fact. Always build labels with `user_scoped_label`.
- **There is no public update or delete for memory** in the installed SDK version. Re-saving a
  label overwrites in place and loses history. Only extraction creates versioned history.
- **Deployed agents invoked by a service account share one memory identity.** The platform ignores
  the caller's `user_id`. This agent also keys policies and claims by `user_id`, so check this
  before deploying.
- **`.env.*` is git-ignored.** The committed template is `env.example`, with no leading dot.
- **Runtime logs can contain conversation content.** `logs/` is ignored. Do not commit it.

### Local state the playground was left in

This lives in the Docker MongoDB, not in the repo, so a fresh checkout will not have it.

- Manifests v1 and v2 retired, v3 active, in `harness_manifests` of `agent_agent46_local`.
- Five test claims. Three are suspended for human review (`CLM-16734`, `CLM-73506`,
  `CLM-51317`), two are closed (`CLM-57309` auto-approved under v1, `CLM-26611` approved after
  review under v3). Each carries the `harness_version` that decided it.
- A seeded test policy, `POL-HARNESS1`, owned by `harness-test-user-1`.
- Throwaway memory users created during testing were deleted afterwards.

---

## 7. What to do next

Done since the first build: held-out leak, a loud fallback event (partly: it is an event, not yet an
alert), the LLM proposer run, a second and third surface, an audit trail.

1. **Make the court's case file faithful.** Count only claims created before the one being filed, and
   refuse to convict a case whose replay disagrees with the stored decision.
2. **Close the audit gap.** `--auto-activate-low-risk` must write an audit row, and consider requiring
   approval for disable-only tool patches until the tool scenarios grow.
3. ~~Run a limits cycle~~ **Done.** Candidate v6 raised the caps from 10/12 to 14/18 (a raise to 20/30 fails the gate). Approved by the operator and active locally.
4. **Raindrop to court: partly done.** `court ingest-events FILE` turns an exported event list into advisory
   nominations (`harness_evolve/raindrop_source.py`). Still manual: creating the custom signal in the
   dashboard and exporting the events; no Raindrop read API is wired up.
5. **Verify an active platform guardrail fires**, then decide whether text rules belong there.
6. **Handle negation in `risk.py`** so "no significant damage" stops matching the keyword.
7. **Harden tool access** (steerable classifier, trajectory scenarios) and **grow the eval sets**.
8. **Make the gates consistent.** Use one critical-violation rule for every surface.
9. **Auth on the court UI** and a named reviewer list before anyone but you uses it.
10. **Package this as a skill** once a second use case has been built by hand.

---

## 8. Decision log

| Decision | Why |
|---|---|
| Manifests live in MongoDB, not git | the runtime already has Mongo, versions need statuses and scores, and approval is a state change |
| Offline heuristic proposer before the LLM one | the loop is testable and free, and the LLM proposer is then a drop-in |
| Proposer emits operations, not code | closed language, easy to validate and to risk-tier |
| Thresholds may tighten but not loosen past baseline | loosening is the dangerous direction, so it needs a human and a new baseline |
| Baseline keywords can be added to but never removed | protects the cases that matter most |
| Evaluator and invariants sit outside the evolvable layer | otherwise the proposer can grade itself |
| Tool access narrows per intent and can only widen with approval | narrowing cannot break a workflow because required tools are fixed, widening is the risky direction |
| Ambiguous intent maps to `general` (all tools) | a wrong guess can then only widen access, never strand a user |
| A claim in flight keeps the claim-flow tools in every intent | an unrelated question mid-claim must not strand `resolve_claim` or `human_review` |
| Keyword matching is whole-word with simple endings | prefix matching made "fireplace" a claim and "priceless" a quote |
| One evaluator per surface, one combined gate per cycle | each surface has its own ground truth, and a change must not regress the other |
| One resolver (`resolve_tool_access`) shared by agent, guard and evaluator | three copies of the logic would drift apart |
| Memory label separator is `::` | a plain `_` join lets two different (user, key) pairs collide |
| `uv.lock` stays git-ignored | the template's convention: CI resolves fresh under its cooldown policy |
| Raindrop nominates, a human convicts | LLM-judged signals are advisory; ground truth must be a person's ruling |
| Limits are manifest data inside fixed bounds; raising needs approval | a higher cap is the costly direction, a lower one cannot strand a flow |
| Behaviour limits stay in code, not in platform guardrails | platform guardrails match text only; they have no counters or tool stage |
| Metadata-only events by default | customer text may contain personal data; content sending is opt-in |
| Claim description text is never sent to Raindrop, only its length | the description is the most sensitive field |
| Destructive actions need `--yes` / a confirm box and write an audit row | an accidental rollback and a polluted docket happened during testing |
| The court docket lives outside the evolvable layer and outside the tests | it is human ground truth, so neither the proposer nor a test run may change it |
| A rejected candidate keeps its record | the audit trail should show what the model proposed, not only what shipped |

---

## 9. Regression Court and Raindrop

The loop that closes "collect" in the diagram in section 1. It was demonstrated end to end on the
local stack.

### The loop

1. **Incident.** A $700 claim describing a vehicle stripped of wheels, catalytic converter and stereo is
   auto-approved under manifest v3 (`CLM-DEMO11`). No reviewer sees it.
2. **Nominate.** A person, the reviewer-disagreement scan (`court scan`) or a Raindrop signal nominates
   the claim. Nominating changes nothing.
3. **File.** `court file` rebuilds a draft scenario from the claim and policy in Mongo (amount, type,
   description, the agent's recommendation, its `harness_version`, any reviewer resolution).
4. **Verdict.** A named adjuster rules the expected outcome and whether it is critical. Only then is a
   scenario written to `evals/scenarios/docket.json`, merged into the **dev split only**. A
   scenario whose description equals a held-out description is rejected.
5. **Cycle.** The proposer reads dev failures (including court cases) and proposes whitelisted ops.
6. **Gate.** The unchanged gate scores the candidate on dev and held-out, on every surface it touches.
7. **Read the diff.** A human reads what the patch changes (principle 18).
8. **Approve.** `approve N --yes`, audited. Invariants still run at propose, activate and load.
9. **Prove it live.** A claim of the same kind is sent through the playground and the stored claim
   carries `harness_version` 5.

### What happened in the demo

| Step | Result |
|---|---|
| Baseline on dev with the new case | accuracy 0.793, critical violations 1, `court-b3e4bffe` failing |
| LLM proposer, v4 (second call, 2000-token cap) | accuracy 0.862, held-out 0.8, gate **pass**, but it loosened `medium_amount` 5000 to 6000 and `high_score` 0.6 to 0.8, contradicting an adjuster verdict. **Rejected by a human.** |
| Heuristic proposer, v5 | one op: add keyword `cannibalized` at high severity. Dev 0.793 to 0.828, critical 1 to 0, held-out unchanged, gate pass on risk, tools and limits. **Approved.** |
| Live claim `CLM-96610` | `harness_version` 5, risk high, `manual_review_required`, suspended, reviewer denied, closed |

### Raindrop events

`agent_turn`, `risk_assessed`, `review_suspended`, `review_resumed`, `tool_denied`,
`tool_limit_hit`, `step_cap_hit`, `tool_access_failed_open`, `manifest_fallback`. Each carries
`harness_version` where known. Court actions, approvals and cycles are **not** sent to Raindrop; they
are in the Mongo `audit_log`.

### Cluster (MongoDB Atlas)

Workspace `agent46` is linked to `Cluster0` through `agentengine atlas`. Status after the second round:

| Step | State |
|---|---|
| Build and deploy of the second-round code | **done**: image `sha-095c995...`, revision r2, 5 of 5 components ready |
| Deployed egress allow-list | **done**: `api.openai.com` and `api.raindrop.ai` on 443 (`agentengine egress`) |
| Cloud secrets | **done**: `RAINDROP_WRITE_KEY` and `RAINDROP_SEND_CONTENT=false` added (project scope, via `--stdin`). Content sending is off in the cloud on purpose. |
| Cloud invoke | **done**: info questions were answered by the deployed agent after the sync, with `RAINDROP_SEND_CONTENT=true` synced |
| New code confirmed in runtime logs | **not confirmed**: the platform logs only startup lines, not turns. Check for an `agent_turn` event from the cloud in Raindrop. |
| Manifests v1 to v6 in Atlas | **done and verified**: `atlas-sync` copied v2 to v6 next to the cloud's own baseline v1 (retired). v6 is the single active version; a second dry-run reports no changes. The operator ran the apply step; the assistant never wrote to the cluster. Needs `ATLAS_EVOLVE_URI` and `ATLAS_EVOLVE_DB` in `.env`. |
| Court and docket against Atlas | **not done**: `harness_evolve --uri` takes the same string. `docket.json` is a file kept on the host. |

Things learned while deploying:

- `agentengine build` uploads the whole directory. `.agentengineignore` did not exclude `logs/`
  (conversation text), `.evolve_cache/` (saved model replies) or `docket.json`. They are excluded now,
  together with `harness_evolve/` and `tests/`, which do not run in the agent.
- `agentengine invoke --user-id X` is refused with "end-user delegation is not enabled for this
  project". Every cloud caller is the authenticated identity, which is the shared-identity caveat in section 6.
- The platform's own database user cannot be used from the host: its password is not available.
  Create a separate Atlas user for the evolve tooling.
- `agentengine status` kept showing the previous deployment id after a successful redeploy.
  Trust `agentengine deploy` output and `egress`, not that line.


---

## 10. The judge demo, the court UI and the cloud proof

### Court UI (harness_evolve/court_ui)

Plain-language labels live in one dict, `UI_LABELS`: *Policy scorecard* (Original rules / Live policy /
Proposed change; accuracy on known and unseen cases, dangerous misses, unneeded reviews, missed
escalations, a Passed or Failed safety check), *Policy versions* (Live, Previous, Awaiting approval,
Rejected), tabs for Nominated, Filed and Decided with count badges, info icons and a dialog that explain
every field. Every candidate version, whether proposed in the browser or created by the command line,
shows a summary card: bullet list of what would change (reconstructed from the manifest diff against
its parent by `ops_from_manifests`), an amber "Loosens a safeguard" badge for raised thresholds and
limits, the score impact, collapsed evidence, and Approve and Reject with a confirm box. Page height
on a 27-case docket fell from 6,274 px to 1,582 px at 1280 px.

### Simulated incident stream (`demo`)

```bash
cli demo seed        # fresh POL-SIM-DEMO, 10 claims, 3 Nominated + 2 Filed cases (tagged demo)
cli demo run --scenario staged-accident --pace 2   # narrated; stops at "waiting for a human"
cli demo status
cli demo reset --yes # removes only demo data, retracts simulated convictions, rejects a demo candidate
```

Scenarios: `parts-theft`, `staged-accident`, `cloned-vehicle` (each proposes one keyword and passes the
gate) and `abandoned-lookalike`, a plausible patch that the gate **rejects** (accuracy 0.862 to 0.793,
unneeded reviews 0.167 to 0.417). The verdicts are given by "Demo Adjuster (simulated)" and are labelled
that way in the audit log. The command never approves, activates or rolls back; it refuses while a
candidate is waiting.

### Run against Atlas

Same commands with `--allow-remote --db <ATLAS_EVOLVE_DB>` and `EVOLVE_MONGODB_URI` set to the Atlas
string. Use a separate docket copy (`EVOLVE_DOCKET_PATH=.evolve_cache/atlas_docket.json`, a copy of the
real docket) so simulated verdicts never enter the real `docket.json`. Start the court against Atlas on
its own port: `court_ui --allow-remote --db <db> --port 8766`. Played once on 2026-10-11: staged-accident
produced candidate v7 (`add_keyword staged`, low risk, gate pass), the operator approved it, and the
audit log reads `demo.simulated_verdict`, `demo.run`, `manifest.approve`.

### Full run of show, rehearsed on Atlas (2026-10-11)

1. Raindrop: the cloud agent's events carry `harness_version`.
2. Court on Atlas: 3 Nominated and 2 Filed cases, all real misses under the live policy (the seed adapts).
3. The operator ruled on `CLM-SIM-02` (cloned VIN) as adjuster: `manual_review_required`, critical. Audited.
4. `demo run --scenario cloned-vehicle`: the run used the operator's ruling, simulated one verdict, and
   proposed v8 (`add_keyword cloned`, low risk, gate pass, dangerous misses 1 to 0).
5. The operator approved v8 in the browser. Atlas: v8 active, v1 to v7 retired (v4 rejected).
6. Cloud chat, same kind of claim: `CLM-79033`, `harness_version` 8, `manual_review_required`, then denied.
   The previous run of the idea (`CLM-75222`, staged collision) is `harness_version` 7.
7. `demo run --scenario abandoned-lookalike`: v9 `add_keyword abandoned` was **rejected by the gate**
   (accuracy 0.829 to 0.771, unneeded reviews 0.231 to 0.462). Nothing to approve; v8 stayed live.
8. Guardrails in the cloud: an injection message is denied before the agent runs, an SSN-like number in a
   reply is replaced.
Audit trail in Atlas at the end: 28 rows, 4 by the human operator (the sync, two approvals, one verdict)
and 14 by the labelled simulated adjuster.

### Cloud proof

The deployed agent logged `Harness manifest version: 7` and `Tool access: intent=claims, offered 14/17
tools`, so it reads the active manifest from Atlas. The first staged-collision claim sent through the
hosted chat was stopped by the platform's `AUTHORIZED_TOOLS` policy (see section 6), before any claim
was written; clearing that policy is an administrator action in the platform UI.
