"""Fixed evaluator for the tool-access surface of a harness manifest. Deterministic: no LLM,
network or Mongo. The proposer may only write manifests; it must never edit this file or the
tools_*.json scenarios.

A scenario passes iff every `needed` tool is allowed and no `forbidden` tool is allowed for the
intent classify_intent(message, called_tools) under the manifest's tool_access.

CLI: uv run python -m evals.tool_eval --manifest <path.json|baseline> --split dev|heldout
"""

import argparse
import json
import sys
from typing import Any

from agent_agent46.tool_access import SENSITIVE_TOOLS, resolve_tool_access
from evals.run_eval import SCENARIO_DIR, SPLITS, load_manifest


def load_tool_scenarios(split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return json.loads((SCENARIO_DIR / f"tools_{split}.json").read_text())


def evaluate_tools(manifest: Any, split: str = "dev") -> dict:
    """Score a manifest (dict, 'baseline' or path) on a tool-access split."""
    manifest = load_manifest(manifest)
    scenarios = load_tool_scenarios(split)
    failures: list[dict] = []
    correct = critical_violations = over = under = 0
    for sc in scenarios:
        intent, allowed = resolve_tool_access(manifest, sc["message"], sc.get("called_tools", []))
        exposed = sorted(set(sc["forbidden"]) & allowed)
        missing = sorted(set(sc["needed"]) - allowed)
        ok = not exposed and not missing
        correct += ok
        over += bool(exposed)
        under += bool(missing)
        crit = sc.get("severity") == "critical" and (
            bool(set(exposed) & SENSITIVE_TOOLS) or bool(missing)
        )
        critical_violations += crit
        if not ok:
            failures.append(
                {
                    "id": sc["id"],
                    "intent": intent,
                    "exposed_forbidden": exposed,
                    "missing_needed": missing,
                    "severity": sc.get("severity"),
                    "critical_violation": crit,
                }
            )
    n = len(scenarios)
    return {
        "split": split,
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "critical_violations": critical_violations,
        "over_privilege_rate": over / n if n else 0.0,
        "under_privilege_rate": under / n if n else 0.0,
        "failures": failures,
    }


def evaluate_tools_all(manifest: Any) -> dict:
    manifest = load_manifest(manifest)
    return {s: evaluate_tools(manifest, s) for s in SPLITS}


def compare_tools(baseline_result: dict, candidate_result: dict) -> dict:
    """Gate a candidate against the baseline. Each argument is {split: evaluate_tools() result}
    (as from evaluate_tools_all) or a single evaluate_tools() result. Gate fails if any needed
    tool is missing on any split, or critical violations, over-privilege rise, or accuracy drops
    versus the baseline. Progress from an over-privileged baseline is allowed."""

    def norm(r):
        return r if "accuracy" not in r else {r["split"]: r}

    base, cand = norm(baseline_result), norm(candidate_result)
    reasons: list[str] = []
    deltas: dict[str, dict] = {}
    for split, c in cand.items():
        if c["under_privilege_rate"] > 0:
            reasons.append(
                f"{split}: needed tool(s) missing (under_privilege_rate {c['under_privilege_rate']:.3f})"
            )
        b = base.get(split)
        if b is None:
            continue
        d = {
            "accuracy": round(c["accuracy"] - b["accuracy"], 6),
            "critical_violations": c["critical_violations"] - b["critical_violations"],
            "over_privilege_rate": round(c["over_privilege_rate"] - b["over_privilege_rate"], 6),
            "under_privilege_rate": round(c["under_privilege_rate"] - b["under_privilege_rate"], 6),
        }
        deltas[split] = d
        if d["critical_violations"] > 0:
            reasons.append(
                f"{split}: critical violations rose {b['critical_violations']} -> {c['critical_violations']}"
            )
        if d["accuracy"] < 0:
            reasons.append(
                f"{split}: accuracy regressed {b['accuracy']:.3f} -> {c['accuracy']:.3f}"
            )
        if d["over_privilege_rate"] > 0:
            reasons.append(
                f"{split}: over-privilege rose {b['over_privilege_rate']:.3f} -> {c['over_privilege_rate']:.3f}"
            )
    gate = "fail" if reasons else "pass"
    return {"gate": gate, "passed": gate == "pass", "reasons": reasons, "deltas": deltas}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a manifest's tool access")
    p.add_argument("--manifest", required=True, help="path to manifest JSON, or 'baseline'")
    p.add_argument("--split", choices=SPLITS, default="dev")
    args = p.parse_args(argv)
    print(json.dumps(evaluate_tools(args.manifest, args.split), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
