# Hand-off: building a self-improving agent harness

This document is for the next person, or the next session, who wants to rebuild this
architecture for a different problem. It starts with the philosophy because the philosophy is
what transfers. The insurance agent here is only the worked example.

Everything marked **verified** was run on a local Docker stack. Anything else is labeled as
designed, untested, or unknown.

---

## 1. The thesis

An agent's behavior is **fixed machinery plus policy**. A self-improving harness rewrites only
the policy, only as data, only through a fixed evaluator, and only inside limits that humans
own.

```
            evolvable (data, in MongoDB)                fixed (code, owned by humans)
        ┌──────────────────────────────────┐        ┌───────────────────────────────────┐
        │ rules · thresholds · keywords    │        │ invariants · evaluator · gate     │
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

1. **Evolve policy, never control flow.** The proposer changes thresholds, rules and keywords.
   It never changes the claim state machine, the human-review interrupt, tool sandboxes or egress.
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
   `remove_rule`, `set_threshold`, `add_keyword`), never code or free-form edits.
   (`harness_evolve/patch.py`.) The result must still pass the invariants.

5. **Humans own the limits, and they are checked in more than one place.** Invariants run when a
   manifest is proposed, when it is activated, and when it is loaded. A stored manifest that fails
   at load falls back to the baseline. (`harness_config.validate_manifest`, `INVARIANTS`.)

6. **Gate on several things at once.** A candidate must not add a critical violation, lose
   accuracy, or raise over-escalation or under-escalation on any split. Our first gate checked
   only accuracy and would have passed a candidate that sent more claims to reviewers.
   (`evals/run_eval.compare`.)

7. **Held-out data is for the gate, never the proposer.** The proposer reads dev failures only.
   Known leak: held-out failure ids are stored in the scores document, so anyone with database
   access can read them.

8. **Version and attribute every decision.** Each claim records the `harness_version` that decided
   it. That is what makes an A/B comparison and a rollback meaningful.

9. **Tier approval by risk.** Adding a rule or keyword is low risk. Changing a threshold or
   removing a rule needs a human. Auto-activation is opt-in and applies to low-risk patches only.

10. **Bound every open field.** An unbounded `add_keyword` could add "door". The invariants cap
    keyword count, length and characters, and the manifest rejects any field it does not know.

11. **Test the guards, not only the features.** Our first static check for memory labels passed
    even after I replaced the label with an unscoped string, because it checked that the helper was
    called somewhere, not that its result was used. A negative control caught it. Break the thing on
    purpose and confirm the test fails.

12. **Never evolve user data.** The harness may change policy around memory (what to save, which
    sources to read). It must not rewrite what a user's memory contains. Memory labels are
    user-scoped by an invariant, because the memory service enforces uniqueness per org.

13. **Prove it on the live path.** Unit tests do not show the manifest is in use. We sent the same
    kind of claim before and after approving a version and read the version tag back.

14. **Fail safe, and fail loudly.** If the store is unreachable the agent runs the baseline.
    Known gap: that fallback is currently silent apart from a log line.

---

## 3. Recreating this for a new problem

### 3.1 Fill in the problem statement first

Do not write code until these are answered in a paragraph each.

| Question | Insurance example | Your answer |
|---|---|---|
| Which decision do you want to improve? | route a claim to auto-approval or human review | |
| What does a wrong decision cost, and in which direction? | auto-approving fraud is worse than a needless review | |
| Which 1 to 3 things may evolve first? | rules, risk thresholds, risk keywords | |
| What must never change? | claim state machine, human review, egress | |
| Who knows the right answer, and where is it written down? | an adjuster's labeled cases | |
| Which outcomes must never happen (critical cases)? | auto-approving arson, theft, over-limit claims | |
| What are the metrics and the gate? | accuracy, both escalation rates, zero critical | |
| Which changes need a human? | any threshold change, any removal | |
| What is user data, and how is it kept separate? | customer facts in memory, label-scoped per user | |

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
| 7 | Add the next surface (tool access, richer guardrails) | its own scenarios and invariants exist before its code |

### 3.4 What to swap and what to keep

| Keep as is (the engine) | Swap per problem (the domain) |
|---|---|
| `HarnessConfigStore`: versioning, candidate/active/retired/rejected, activate, fallback | the manifest schema and `DEFAULT_MANIFEST` |
| `validate_manifest` structure: bounds, unknown-field rejection, caps | the actual bounds and caps in `INVARIANTS` |
| `evals/run_eval.py`: `evaluate`, `compare`, CLI | scenarios (`dev.json`, `heldout.json`) and the metric names |
| `harness_evolve/loop.py`: cycle, approve, rollback, score recording | the operations in `patch.py` and their risk tiers |
| `harness_evolve/proposer.py`: interface, strict LLM parsing | the heuristic proposer's lexicon |
| `memory_policy.py`: user-scoped labels | nothing, unless your memory model differs |
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
    │   ├── memory_policy.py  user-scoped memory labels
    │   ├── policy_store.py   policy and claim storage, claim state machine
    │   └── llm.py            model selection
    ├── evals/                FIXED evaluator: scenarios, scorer, gate
    ├── harness_evolve/       proposer, patch language, cycle, CLI (never ships in the agent)
    └── tests/                99 tests
```

`.claude/skills` and `.agents/skills` are the Agent Engine template's skill docs, kept as shipped.

---

## 5. State of the build

**Built and verified**

- Versioned manifest in MongoDB, with invariants checked at propose, activate and load.
- A pure risk function with 22 characterization tests.
- A fixed evaluator: 25 dev and 10 held-out scenarios, 11 of them critical, and a four-part gate.
- An offline heuristic proposer, the patch language, and the cycle with approve and rollback.
- Memory label invariant, enforced by a source-parsing test that was itself negative-tested.
- Live A/B on the running agent.

| | Baseline (v1) | Evolved (v2) |
|---|---|---|
| Dev accuracy | 0.68 | 0.88 |
| Held-out accuracy | 0.40 | 0.70 |
| Critical violations (dev) | 1 | 0 |
| Live arson claim (a $750 claim under v1, a $700 claim under v2) | low risk, auto-approved | high risk, sent to human review |

**Designed but not built:** tool access as an evolvable surface (see section 7).

**Written but never run:** the LLM proposer. It makes network calls and was deliberately left off.

**Known limits**

- The heuristic proposer only finds what its fixed phrase list contains. It demonstrates the loop,
  not intelligence.
- The eval sets are small. A gain of 0.3 is a few scenarios.
- Some failures cannot be fixed through the manifest. Descriptions like "no fire involved" match a
  keyword by substring, and only a code change (negation or word boundaries) fixes that.
- The cross-user memory overwrite and the empty taxonomic extractions were observed on the local
  stack only. Check the hosted service before relying on either finding.

---

## 6. Operating it, and the gotchas that cost time

```bash
cd agents/agent46
cp env.example .env                     # then fill in LLM_API_KEY and VOYAGE_API_KEY
agentengine dev up                      # NOT `agentengine dev`: that only prints help
agentengine dev status                  # shows the mongo port and the playground URL

uv sync
uv run pytest -q
uv run python -m evals.run_eval --manifest baseline --split dev
uv run python -m harness_evolve.cli cycle
uv run python -m harness_evolve.cli status
uv run python -m harness_evolve.cli approve 2
uv run python -m harness_evolve.cli rollback 1
```

- **Mongo from the host needs `?directConnection=true`.** The Docker Mongo advertises an internal
  hostname, so a plain `mongodb://localhost:<port>` times out. The port is assigned per machine.
- **Invoking the agent needs a `user_id`.** Without one the runtime fails with
  `durable Memory identity is incomplete`.
- **Importing `main.py` outside the runtime needs `RUNNER_MODE=aer`.**
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

---

## 7. What to do next

1. **Tool access as an evolvable surface.** Add `tool_access` to the manifest as a per-intent
   allow-list that can only narrow what `agent.yaml` already places in a sandbox. Enforce it twice:
   when binding tool schemas to the model, and again before a tool runs, because a model can name
   a tool it was never offered. Any expansion needs approval, and `human_review` can never be
   removed. Scoring it needs whole-conversation scenarios. `build_agent(llm=...)` accepts a fake
   model, which is the likely route. The open question is how to score tool-use trajectories
   without paying for an LLM on every run.
2. **Run the LLM proposer** for one cycle against a fresh baseline and compare it with the
   heuristic one. Expect it to need a stricter parser than you think.
3. **Grow the eval sets** and add scenarios the proposer has never seen.
4. **Fix the held-out leak** by keeping held-out failure ids out of stored scores.
5. **Make the fallback loud.** Surface the active manifest version in a status check, and log at
   error level when the store is unreachable.
6. **Handle negation in `risk.py`** so "no fire involved" stops matching the keyword.
7. **Investigate the taxonomic extraction failures.** All five dead-lettered tasks failed with an
   empty model response, and `explain_insurance_term` depends on that memory.
8. **Package this as a skill** once a second use case has been built by hand. Do not write the
   skill from one example.

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
| Memory label separator is `::` | a plain `_` join lets two different (user, key) pairs collide |
| `uv.lock` stays git-ignored | the template's convention: CI resolves fresh under its cooldown policy |
