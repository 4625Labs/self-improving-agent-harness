import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_agent46.tool_access import (
    ALL_TOOLS,
    INTENTS,
    REQUIRED_TOOLS,
    baseline_tool_access,
    classify_intent,
)
from evals.tool_eval import compare_tools, evaluate_tools, evaluate_tools_all, load_tool_scenarios

SPLITS = ("dev", "heldout")
ALL = [(s, sc) for s in SPLITS for sc in load_tool_scenarios(s)]


def reference_manifest():
    allow = {}
    for intent in INTENTS:
        needed = {t for _, sc in ALL if sc["expected_intent"] == intent for t in sc["needed"]}
        allow[intent] = sorted(
            ALL_TOOLS if intent == "general" else needed | REQUIRED_TOOLS[intent]
        )
    return {"tool_access": {"intents": {i: {"allow": a} for i, a in allow.items()}}}


def manifest_with(intent, remove):
    m = {"tool_access": baseline_tool_access()}
    a = m["tool_access"]["intents"][intent]["allow"]
    m["tool_access"]["intents"][intent]["allow"] = [t for t in a if t not in remove]
    return m


def test_baseline_has_exposed_tools_but_no_missing():
    for split in SPLITS:
        r = evaluate_tools("baseline", split)
        assert r["under_privilege_rate"] == 0
        assert r["over_privilege_rate"] > 0
        assert r["critical_violations"] > 0
        assert r["accuracy"] < 1


def test_reference_manifest_is_perfect():
    m = reference_manifest()
    for split in SPLITS:
        r = evaluate_tools(m, split)
        assert r["accuracy"] == 1.0, r["failures"]
        assert r["critical_violations"] == 0
        assert r["failures"] == []


def test_manifest_forms_equivalent():
    assert (
        evaluate_tools({"tool_access": baseline_tool_access()}, "dev")["accuracy"]
        == evaluate_tools("baseline")["accuracy"]
    )
    assert set(evaluate_tools_all("baseline")) == set(SPLITS)


def test_compare_progress_passes_and_over_narrowing_fails():
    base = evaluate_tools_all("baseline")
    good = evaluate_tools_all(manifest_with("info", {"resolve_claim"}))
    assert compare_tools(base, good)["passed"]
    ref = evaluate_tools_all(reference_manifest())
    assert compare_tools(base, ref)["gate"] == "pass"
    bad = evaluate_tools_all(manifest_with("info", {"lookup_policy"}))
    res = compare_tools(base, bad)
    assert res["gate"] == "fail" and not res["passed"]
    assert any("missing" in r for r in res["reasons"])


def test_compare_fails_on_regression_vs_better_baseline():
    ref = evaluate_tools_all(reference_manifest())
    worse = evaluate_tools_all("baseline")
    res = compare_tools(ref, worse)
    assert res["gate"] == "fail"
    assert any("over-privilege" in r for r in res["reasons"])
    assert any("accuracy" in r for r in res["reasons"])
    assert any("critical" in r for r in res["reasons"])


def test_compare_accepts_single_results():
    b, c = evaluate_tools("baseline"), evaluate_tools(reference_manifest())
    assert compare_tools(b, c)["passed"]
    assert "dev" in compare_tools(b, c)["deltas"]


def test_failure_shape():
    f = evaluate_tools("baseline")["failures"][0]
    assert set(f) == {
        "id",
        "intent",
        "exposed_forbidden",
        "missing_needed",
        "severity",
        "critical_violation",
    }


def test_load_unknown_split():
    with pytest.raises(ValueError):
        load_tool_scenarios("nope")


@pytest.mark.parametrize("split,sc", ALL, ids=[sc["id"] for _, sc in ALL])
def test_scenario_honesty(split, sc):
    assert set(sc) == {
        "id",
        "message",
        "called_tools",
        "expected_intent",
        "needed",
        "forbidden",
        "severity",
        "tags",
    }
    assert classify_intent(sc["message"], sc["called_tools"]) == sc["expected_intent"]
    assert set(sc["needed"]) <= ALL_TOOLS and set(sc["forbidden"]) <= ALL_TOOLS
    assert set(sc["called_tools"]) <= ALL_TOOLS
    assert not set(sc["forbidden"]) & REQUIRED_TOOLS[sc["expected_intent"]]
    assert sc["severity"] in ("critical", "normal")
    if sc["expected_intent"] == "general":
        assert sc["forbidden"] == []


def test_ids_unique_and_splits_distinct():
    ids = [sc["id"] for _, sc in ALL]
    assert len(ids) == len(set(ids))
    dev = {sc["message"] for sc in load_tool_scenarios("dev")}
    assert not dev & {sc["message"] for sc in load_tool_scenarios("heldout")}
    for s in SPLITS:
        assert {sc["expected_intent"] for sc in load_tool_scenarios(s)} == set(INTENTS)


def test_no_contradictions_per_intent():
    for intent in INTENTS:
        needed = set(REQUIRED_TOOLS[intent])
        forbidden = set()
        for _, sc in ALL:
            if sc["expected_intent"] == intent:
                needed |= set(sc["needed"])
                forbidden |= set(sc["forbidden"])
        assert not needed & forbidden, (intent, needed & forbidden)


def test_manifest_not_mutated():
    m = reference_manifest()
    snap = copy.deepcopy(m)
    evaluate_tools_all(m)
    assert m == snap
