"""Fixed evaluator for harness manifests. Deterministic: no LLM, network or Mongo.

The proposer may only write manifests; it must never edit this package or the scenarios.

CLI: uv run python -m evals.run_eval --manifest <path.json|baseline> --split dev|heldout
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

SCENARIO_DIR = Path(__file__).parent / "scenarios"
SPLITS = ("dev", "heldout")
AUTO, REVIEW, MANUAL = "auto_approve", "review_recommended", "manual_review_required"
RANK = {AUTO: 0, REVIEW: 1, MANUAL: 2}


def load_scenarios(split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return json.loads((SCENARIO_DIR / f"{split}.json").read_text())


def load_manifest(spec: Any) -> dict:
    """Accept a manifest dict, 'baseline', or a path to a JSON file."""
    from agent_agent46.harness_config import DEFAULT_MANIFEST

    if isinstance(spec, dict):
        return copy.deepcopy(spec)
    if spec == "baseline":
        return copy.deepcopy(DEFAULT_MANIFEST)
    return json.loads(Path(spec).read_text())


def _assess():
    from agent_agent46.risk import assess_claim_risk

    return assess_claim_risk


def evaluate(manifest: Any, split: str = "dev") -> dict:
    """Score a manifest (dict, 'baseline' or path) on a split."""
    manifest = load_manifest(manifest)
    assess = _assess()
    scenarios = load_scenarios(split)
    failures: list[dict] = []
    correct = critical_violations = 0
    auto_expected = auto_expected_missed = 0
    esc_expected = under = 0
    for sc in scenarios:
        prev = [{"claim_amount": 0}] * int(sc.get("previous_claims_count", 0))
        out = assess(copy.deepcopy(sc["claim"]), copy.deepcopy(sc["policy"]), prev, copy.deepcopy(manifest))
        got, exp = out["recommendation"], sc["expected"]["recommendation"]
        ok = got == exp
        correct += ok
        if exp == AUTO:
            auto_expected += 1
            auto_expected_missed += got != AUTO
        else:
            esc_expected += 1
            under += RANK.get(got, 0) < RANK[exp]
        crit = sc.get("severity") == "critical" and got == AUTO
        critical_violations += crit
        if not ok or crit:
            failures.append({"id": sc["id"], "expected": exp, "got": got, "severity": sc.get("severity"),
                             "tags": sc.get("tags", []), "critical_violation": crit})
    n = len(scenarios)
    return {
        "split": split,
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "critical_violations": critical_violations,
        "over_escalation_rate": auto_expected_missed / auto_expected if auto_expected else 0.0,
        "under_escalation_rate": under / esc_expected if esc_expected else 0.0,
        "failures": failures,
    }


def evaluate_all(manifest: Any) -> dict:
    return {s: evaluate(manifest, s) for s in SPLITS}


def compare(baseline_result: dict, candidate_result: dict) -> dict:
    """Gate a candidate against the baseline. Each argument is {split: evaluate() result}
    (as from evaluate_all) or a single evaluate() result. Gate fails on any critical
    violation, an accuracy regression, or a rise in over- or under-escalation on any split."""
    def norm(r):
        return r if "accuracy" not in r else {r["split"]: r}

    base, cand = norm(baseline_result), norm(candidate_result)
    reasons: list[str] = []
    deltas: dict[str, dict] = {}
    for split in cand:
        c = cand[split]
        b = base.get(split)
        if c["critical_violations"] > 0:
            reasons.append(f"{split}: {c['critical_violations']} critical violation(s)")
        if b is not None:
            d = round(c["accuracy"] - b["accuracy"], 6)
            deltas[split] = {"accuracy": d,
                             "over_escalation_rate": round(c["over_escalation_rate"] - b["over_escalation_rate"], 6),
                             "under_escalation_rate": round(c["under_escalation_rate"] - b["under_escalation_rate"], 6)}
            if d < 0:
                reasons.append(f"{split}: accuracy regressed {b['accuracy']:.3f} -> {c['accuracy']:.3f}")
            if deltas[split]["over_escalation_rate"] > 0:
                reasons.append(f"{split}: over-escalation rose {b['over_escalation_rate']:.3f} -> {c['over_escalation_rate']:.3f}")
            if deltas[split]["under_escalation_rate"] > 0:
                reasons.append(f"{split}: under-escalation rose {b['under_escalation_rate']:.3f} -> {c['under_escalation_rate']:.3f}")
    gate = "fail" if reasons else "pass"
    return {"gate": gate, "passed": gate == "pass", "reasons": reasons, "deltas": deltas}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a harness manifest")
    p.add_argument("--manifest", required=True, help="path to manifest JSON, or 'baseline'")
    p.add_argument("--split", choices=SPLITS, default="dev")
    args = p.parse_args(argv)
    result = evaluate(args.manifest, args.split)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
