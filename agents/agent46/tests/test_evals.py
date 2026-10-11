import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_agent46.harness_config import DEFAULT_MANIFEST, validate_manifest
from evals.run_eval import compare, evaluate, evaluate_all, load_scenarios


def test_scenario_files_wellformed():
    for split, lo in (("dev", 20), ("heldout", 8)):
        sc = load_scenarios(split)
        assert len(sc) >= lo
        assert len({s["id"] for s in sc}) == len(sc)
        for s in sc:
            assert {"claim", "policy", "previous_claims_count", "expected", "tags", "severity"} <= set(s)
            assert s["expected"]["recommendation"] in {
                "auto_approve", "review_recommended", "manual_review_required"}


def test_evaluate_shape_and_determinism():
    r = evaluate("baseline", "dev")
    for k in ("accuracy", "critical_violations", "over_escalation_rate", "under_escalation_rate", "failures", "n"):
        assert k in r
    assert 0 <= r["accuracy"] <= 1
    assert r["failures"], "baseline must leave headroom"
    assert r == evaluate(copy.deepcopy(DEFAULT_MANIFEST), "dev")


def test_baseline_manifest_valid_and_untouched():
    before = copy.deepcopy(DEFAULT_MANIFEST)
    evaluate(DEFAULT_MANIFEST, "heldout")
    assert DEFAULT_MANIFEST == before
    assert validate_manifest(DEFAULT_MANIFEST) == []


def test_keywords_can_improve_dev():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["keywords"]["high"] += ["arson", "submerged", "hit and run"]
    assert evaluate(m, "dev")["critical_violations"] < evaluate("baseline", "dev")["critical_violations"]


def _res(acc, crit=0):
    return {"split": "dev", "accuracy": acc, "critical_violations": crit,
            "over_escalation_rate": 0.0, "under_escalation_rate": 0.0}


def test_compare_pass_and_fail_cases():
    base = {"dev": _res(0.5), "heldout": _res(0.5)}
    assert compare(base, {"dev": _res(0.6), "heldout": _res(0.5)})["gate"] == "pass"
    v = compare(base, {"dev": _res(0.9), "heldout": _res(0.4)})
    assert v["gate"] == "fail" and any("heldout" in r for r in v["reasons"])
    v = compare(base, {"dev": _res(0.9, crit=1), "heldout": _res(0.9)})
    assert v["gate"] == "fail" and any("critical" in r for r in v["reasons"])


def test_compare_identical_passes_when_no_critical():
    r = evaluate_all("baseline")
    cand = {s: dict(v, critical_violations=0) for s, v in r.items()}
    assert compare(r, cand)["gate"] == "pass"
