"""Fixed evaluator for the limits surface of a harness manifest. Deterministic: no LLM, network or
Mongo. The proposer may only write manifests; it must never edit this file or the limits_*.json
scenarios.

A scenario is a replayed trace summary: {id, kind: legit|runaway, llm_steps, tool_calls,
max_identical_calls, expected: complete|capped, severity, tags}. `would_cap` mirrors the runtime
rules (tool_limits.check_call and main.agent_node) so a trace the runtime would cut off is capped
here too. The limits come from tool_limits.resolve_limits, the same resolver the agent uses.

Scoring: a legit trace that gets capped is a FALSE CAP (under-capacity: the limit is too low). A
runaway trace that is not capped is a MISSED CAP (critical: the limit is too high).

CLI: uv run python -m evals.limits_eval --manifest <path.json|baseline> --split dev|heldout
"""

import argparse
import json
import sys
from typing import Any

from agent_agent46 import tool_limits
from evals.run_eval import SCENARIO_DIR, SPLITS, load_manifest

KINDS = ("legit", "runaway")


def load_limits_scenarios(split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return json.loads((SCENARIO_DIR / f"limits_{split}.json").read_text())


def cap_reason(trace: dict, limits: tuple[int, int]) -> str | None:
    """Which runtime rule would cut this trace off, or None if it completes.

    Runtime: the k-th LLM call is refused when k-1 prior steps >= max steps (so steps > max caps);
    the j-th tool call is refused when j-1 prior calls >= max calls (calls > max caps); the
    identical-call rule refuses the (N+1)-th copy of one exact call.
    """
    max_steps, max_calls = limits
    if trace["llm_steps"] > max_steps:
        return "llm_steps"
    if trace["tool_calls"] > max_calls:
        return tool_limits.RULE_TURN
    if trace["max_identical_calls"] > tool_limits.MAX_IDENTICAL_CALLS_PER_TURN:
        return tool_limits.RULE_IDENTICAL
    return None


def would_cap(trace: dict, limits: tuple[int, int]) -> bool:
    return cap_reason(trace, limits) is not None


def evaluate_limits(manifest: Any, split: str = "dev") -> dict:
    """Score a manifest (dict, 'baseline' or path) on a limits split."""
    manifest = load_manifest(manifest)
    limits = tool_limits.resolve_limits(manifest)
    scenarios = load_limits_scenarios(split)
    failures: list[dict] = []
    correct = false_caps = missed_caps = 0
    for sc in scenarios:
        reason = cap_reason(sc, limits)
        capped = reason is not None
        ok = capped == (sc["expected"] == "capped")
        correct += ok
        if ok:
            continue
        outcome = "false_cap" if sc["kind"] == "legit" else "missed_cap"
        false_caps += outcome == "false_cap"
        missed_caps += outcome == "missed_cap"
        failures.append(
            {
                "id": sc["id"],
                "kind": sc["kind"],
                "outcome": outcome,
                "llm_steps": sc["llm_steps"],
                "tool_calls": sc["tool_calls"],
                "max_identical_calls": sc["max_identical_calls"],
                "capped_by": reason,
                "severity": sc["severity"],
                "critical_violation": outcome == "missed_cap",
            }
        )
    n = len(scenarios)
    return {
        "split": split,
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "false_caps": false_caps,
        "missed_caps": missed_caps,
        "critical_violations": missed_caps,
        "limits": {"max_llm_steps_per_turn": limits[0], "max_tool_calls_per_turn": limits[1]},
        "failures": failures,
    }


def evaluate_limits_all(manifest: Any) -> dict:
    manifest = load_manifest(manifest)
    return {s: evaluate_limits(manifest, s) for s in SPLITS}


def compare_limits(baseline_result: dict, candidate_result: dict) -> dict:
    """Gate a candidate against the baseline. Each argument is {split: evaluate_limits() result}
    (as from evaluate_limits_all) or a single evaluate_limits() result. Fails on any missed runaway
    in the candidate (critical, on any split), on an accuracy drop, or on rising false caps."""

    def norm(r):
        return r if "accuracy" not in r else {r["split"]: r}

    base, cand = norm(baseline_result), norm(candidate_result)
    reasons: list[str] = []
    deltas: dict[str, dict] = {}
    for split, c in cand.items():
        if c["missed_caps"] > 0:
            reasons.append(f"{split}: {c['missed_caps']} runaway trace(s) not capped (critical)")
        b = base.get(split)
        if b is None:
            continue
        d = {
            "accuracy": round(c["accuracy"] - b["accuracy"], 6),
            "false_caps": c["false_caps"] - b["false_caps"],
            "missed_caps": c["missed_caps"] - b["missed_caps"],
        }
        deltas[split] = d
        if d["accuracy"] < 0:
            reasons.append(f"{split}: accuracy regressed {b['accuracy']:.3f} -> {c['accuracy']:.3f}")
        if d["false_caps"] > 0:
            reasons.append(f"{split}: false caps rose {b['false_caps']} -> {c['false_caps']}")
    gate = "fail" if reasons else "pass"
    return {"gate": gate, "passed": gate == "pass", "reasons": reasons, "deltas": deltas}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a manifest's runaway-protection limits")
    p.add_argument("--manifest", required=True, help="path to manifest JSON, or 'baseline'")
    p.add_argument("--split", choices=SPLITS, default="dev")
    args = p.parse_args(argv)
    print(json.dumps(evaluate_limits(args.manifest, args.split), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
