# Regression Court: production failures become tested policy

## Context
Agent46 already has a fixed evaluator, a closed patch language, a gate, and Raindrop events (done this session).
What is missing is the "collect" step in the HANDOFF loop. Dev scenarios are hand-written, so the harness can only
improve on failures someone imagined. Regression Court closes the loop: a failure seen in production becomes a
labelled scenario, the proposer fixes it, the unchanged gate judges the fix, and a human approves it.

Rule that keeps this safe: **Raindrop nominates, a human convicts, the evaluator decides.** Raindrop's signals are
LLM-judged, so they only nominate cases. A scenario becomes ground truth only after a human verdict.

## The use case, end to end (the demo story)
1. **Incident.** A customer files a $700 claim: "vandalism, car found stripped, no witnesses." The live agent (manifest v3)
   auto-approves it. Nobody notices at the time. This is a missed fraud.
2. **Detection.** Raindrop receives the turn. A custom signal ("claim approved despite fraud-like cues") and a
   disagreement check on `claim.recommendation` vs reviewer outcome nominate the case. It shows up as a candidate in the docket.
3. **Case file.** `court file` pulls the claim by `claim_id` from Mongo (description, amount, type, policy, `harness_version`,
   `recommendation`) and builds a draft risk scenario with `expected` unset.
4. **Verdict, in the web docket.** The adjuster opens the Court docket page, sees the case file (claim text, amount, what the agent
   decided, which manifest version, why it was nominated) and rules: expected outcome plus a "critical" checkbox, or dismiss.
   Only then does the case become a scenario. Cases with no verdict never enter any eval.
5. **Trial.** Verdicts land in `evals/scenarios/docket.json` and are merged into the **dev** split only. Held-out is untouched.
   The baseline and v3 now fail the new case, so the proposer has real headroom.
6. **Proposal.** The existing `harness_evolve` cycle runs. The proposer (heuristic, then LLM) emits whitelisted ops such as
   `add_keyword "stripped"`. Nothing else is allowed.
7. **Gate.** The existing four-part gate scores the candidate on dev plus untouched held-out: no new critical violation, no accuracy loss,
   no rise in over- or under-escalation.
8. **Approval.** `approve N` activates it. Invariants still run at propose, activate and load.
9. **Proof.** The same kind of claim is filed again. It is now sent to human review, the claim records the new `harness_version`,
   and Raindrop shows the issue rate for that failure type falling between v3 and v4. This is the live A/B.

A second detector, the spend circuit breaker, can plug in later as another candidate source (looping traces). It is out of scope for the first 5 hours.

## Work packages (about 6.5 hours as scoped; cut P's polish or A's extras first if the 5-hour limit is firm)
| # | Package | Est. |
|---|---|---|
| A | **Richer events.** Extend `risk_assessed` with amount, claim_type, description length/hash, policy id. Add `harness_version` and `recommendation` to `review_resumed`. Add intent and `session_id` to `tool_denied`. (`observability.py`, call sites in `main.py`, `tool_guard.py`) | 30m |
| B | **Docket store.** Mongo collection `court_cases` (`case_id, claim_id, source, status: nominated/filed/convicted/dismissed, draft scenario, verdict, verdict_by`). Reuse the `HarnessConfigStore` connection pattern (`harness_config.py:145`). | 45m |
| C | **Nominators.** (1) Reviewer-disagreement: `recommendation` vs `resolution` over-escalation. (2) Under-escalation: Raindrop signal export. Manual nomination by `claim_id` as a fallback so the demo never depends on Raindrop timing. New module `harness_evolve/court.py`. | 60m |
| D | **Case filing and verdict logic.** `court nominate / file / verdict / list` in `harness_evolve/cli.py`, shared with the web page. A verdict writes a scenario in the exact `evals/scenarios/dev.json` schema, tagged `source:court`. | 40m |
| D2 | **Web docket page.** Small FastAPI app in `harness_evolve/court_ui/` (host-side, never ships in the deployed agent, same as the rest of `harness_evolve`). Landing page is the docket: nominated cases, case file, verdict form, then buttons that run the cycle and show the candidate, its gate result and Approve or Reject. It calls the same functions as the CLI. Bound to localhost, no auth beyond that, reads Mongo with `EVOLVE_MONGODB_URI`. Assumption: "main page" means the docket is the landing page of this app, since the Atlas playground at :3000 is platform-owned and not editable. | 90m |
| E | **Scenario seam.** Merge `docket.json` into dev inside `evals/run_eval.load_scenarios`. Thread the same merge into the proposer's own `load_scenarios` (`proposer.py:45`) so both see the same cases. Held-out is never merged. | 30m |
| F | **Honesty check for risk scenarios** (the tool side has one, risk does not). Court-filed scenarios must pass schema validation, unique ids, no dev/held-out text overlap, and a rule-based reason for the label. Add to `tests/`. | 30m |
| G | **Close known gaps the story depends on.** Remove `failed_ids` for held-out from the stored scores (`loop.py:14`, HANDOFF item 4). Keep the risk and tool gates' critical-violation rules as they are unless a test forces otherwise. | 20m |
| P | **LLM proposer budget controls** (cache, call cap, dry-run, token cap) and one real run against a fresh baseline. | 60m |
| H | **Live proof and demo script.** Seed three incidents, run through the court, approve, re-file, compare in Raindrop by `harness_version`. | 45m |
| I | **Tests.** Each package ships tests. Negative controls: an unconvicted case never enters dev, held-out file is never read by the court, and an unknown op is still rejected. | in each |

Parallel plan: A+B in parallel, then C+D+E, then F+G, then H. Use Sonnet agents for B to F in a worktree. Haiku is enough for the
tests-only work. I review and run the integration steps myself.

## Reuse (do not rebuild)
- Gate and evaluation: `evals/run_eval.py` (`evaluate`, `compare`), `evals/tool_eval.py`
- Cycle, approve and rollback: `harness_evolve/loop.py` (`run_cycle`, `approve`)
- Proposers and patch language: `harness_evolve/proposer.py`, `harness_evolve/patch.py`
- Store pattern: `HarnessConfigStore` in `harness_config.py`
- Claim lookup fields: `policy_store.py` (`Claim` model, claim stores `harness_version`)
- Raindrop plumbing: `src/agent_agent46/observability.py` (already working, egress allowlisted)

## Risks and decisions
- **Missed fraud leaves no reviewer event.** Over-escalation is detectable from reviewer outcome; under-escalation is not.
  So under-escalation relies on the Raindrop signal or a human nomination, and the human verdict is the ground truth either way.
- **Overfitting to one case.** A keyword added for one claim could cause over-escalation. The gate's over-escalation check and the untouched
  held-out split are the defence. We also show this in the demo as a rejected candidate.
- **Egress and content.** Court cases contain claim text. Keep `RAINDROP_SEND_CONTENT` an explicit choice, and never send docket data back out.
- **LLM proposer, with a tight API budget.** HANDOFF says it was written but never run, so budget an hour of risk. To keep key usage low:
  one call per cycle (never per case), prompt carries only the failing scenarios and the manifest, `max_tokens` capped, the cheapest capable model
  via `agent_agent46.llm.MODEL`, the response cached by a hash of the prompt so UI reruns cost nothing, a `--dry-run` that prints the prompt and token
  estimate without calling, and a hard per-session call limit with a clear error. Fallback if it misbehaves: the heuristic proposer, stated openly.
- Out of scope: tool-access scenarios from denials (phase 2), the spend circuit breaker, auto-activation.

## Verification
1. `uv run pytest -q` passes (295 now, plus new tests). Negative controls fail when the guard is removed.
2. `evals.run_eval --manifest baseline --split dev` shows the new court case failing, then passing under the approved version. Held-out score is unchanged in the scenario set.
3. `harness_evolve.cli court list` and `cycle` and `status` work against the Docker Mongo.
4. Live: file the same kind of claim before and after `approve`. Read back `harness_version` on the claim and see the behaviour change.
5. Raindrop: events carry `harness_version`; the failure-type signal rate differs between v3 and v4.
