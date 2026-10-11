"""The evolvable limits surface: manifest, resolver, patch ops, evaluator, gate, proposer, cycle."""

import copy
import json
import os
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_agent46 import observability, tool_limits as tl  # noqa: E402
from agent_agent46.harness_config import DEFAULT_MANIFEST, INVARIANTS, validate_manifest
from evals.limits_eval import (
    compare_limits,
    evaluate_limits,
    evaluate_limits_all,
    load_limits_scenarios,
    would_cap,
)
from harness_evolve.loop import run_cycle
from harness_evolve.patch import PatchError, apply_patch, classify_risk, validate_ops
from harness_evolve.proposer import (
    CompositeProposer,
    HeuristicProposer,
    LimitsHeuristicProposer,
    LLMProposer,
    ToolHeuristicProposer,
)
from tests.test_tool_guard import RecordingLLM, ai

SPLITS = ("dev", "heldout")
ALL = [(s, sc) for s in SPLITS for sc in load_limits_scenarios(s)]
STEPS, CALLS = "max_llm_steps_per_turn", "max_tool_calls_per_turn"


def with_limits(steps=10, calls=12):
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["limits"] = {STEPS: steps, CALLS: calls}
    return m


# ---------------------------------------------------------------- manifest / invariants


def test_baseline_manifest_has_todays_limits():
    assert DEFAULT_MANIFEST["limits"] == {STEPS: 10, CALLS: 12}
    assert tl.resolve_limits(DEFAULT_MANIFEST) == (10, 12)
    assert validate_manifest(DEFAULT_MANIFEST) == []


def test_stored_manifest_without_limits_is_baseline():
    old = {k: v for k, v in copy.deepcopy(DEFAULT_MANIFEST).items() if k != "limits"}
    assert validate_manifest(old) == []
    assert tl.resolve_limits(old) == (10, 12)
    assert evaluate_limits(old, "dev") == evaluate_limits("baseline", "dev")


@pytest.mark.parametrize(
    "limits",
    [
        {STEPS: 3, CALLS: 12},  # below min
        {STEPS: 21, CALLS: 12},  # above max
        {STEPS: 10, CALLS: 5},
        {STEPS: 10, CALLS: 31},
        {STEPS: 10.0, CALLS: 12},  # float
        {STEPS: True, CALLS: 12},  # bool
        {STEPS: "10", CALLS: 12},
        {STEPS: 10},  # missing field
        {STEPS: 10, CALLS: 12, "max_identical_calls_per_turn": 9},  # fixed in code, not evolvable
        [10, 12],
    ],
)
def test_out_of_bounds_or_malformed_limits_rejected(limits):
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["limits"] = limits
    assert validate_manifest(m)


def test_bounds_edges_accepted_and_unknown_fields_still_rejected():
    lo, hi = INVARIANTS["limits_bounds"][STEPS], INVARIANTS["limits_bounds"][CALLS]
    assert lo == (4, 20) and hi == (6, 30)
    assert validate_manifest(with_limits(4, 6)) == [] and validate_manifest(with_limits(20, 30)) == []
    m = with_limits()
    m["surprise"] = 1
    assert validate_manifest(m)


# ---------------------------------------------------------------- resolver


def test_resolver_values_and_fail_safe(caplog):
    assert tl.resolve_limits(with_limits(14, 18)) == (14, 18)
    for bad in (None, 5, "x", {"limits": 5}, {"limits": {STEPS: 99, CALLS: 12}}, {"limits": {}}):
        with caplog.at_level("ERROR"):
            caplog.clear()
            got = tl.resolve_limits(bad)
        if bad in (None, 5, "x"):  # no limits at all: baseline, nothing to complain about
            assert got == (10, 12)
        else:
            assert got == (10, 12)
            assert any(r.levelname == "ERROR" for r in caplog.records)


def test_resolver_never_raises_even_if_invariants_break(monkeypatch, caplog):
    from agent_agent46 import harness_config

    monkeypatch.setattr(harness_config, "INVARIANTS", {})  # KeyError inside the resolver
    with caplog.at_level("ERROR"):
        assert tl.resolve_limits(with_limits(14, 18)) == (10, 12)
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_check_call_uses_resolved_calls_limit():
    hist = [HumanMessage(content="hi")]
    for i in range(13):
        hist.append(AIMessage(content="", tool_calls=[{"name": "t", "args": {"x": str(i)}, "id": f"h{i}"}]))
    c = {"name": "t", "args": {"x": "new"}, "id": "n"}
    assert tl.check_call(hist[:13], c).startswith(tl.RULE_TURN)  # 12 prior: baseline cap
    assert tl.check_call(hist[:13], c, (10, 18)) is None  # raised
    assert tl.check_call(hist, c, (10, 18)) is None  # 13 prior still under 18
    assert tl.check_call(hist[:13], c, (10, 6)) is not None  # lowered


# ---------------------------------------------------------------- trace_from_messages


def test_trace_from_messages():
    def a(name, **args):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"{name}{len(args)}{id(args)}"}])

    msgs = [
        HumanMessage(content="old"),
        a("x", n=1),
        HumanMessage(content="new"),
        a("lookup", p=1),
        ToolMessage(content="ok", tool_call_id="z"),
        a("lookup", p=1),
        AIMessage(content="", tool_calls=[{"name": "other", "args": {}, "id": "o1"}, {"name": "other", "args": {}, "id": "o2"}]),
        AIMessage(content="done"),
    ]
    assert tl.trace_from_messages(msgs) == {"llm_steps": 4, "tool_calls": 4, "max_identical_calls": 2}
    assert tl.trace_from_messages([HumanMessage(content="hi"), AIMessage(content="hello")]) == {
        "llm_steps": 1, "tool_calls": 0, "max_identical_calls": 0,
    }


# ---------------------------------------------------------------- patch language


def test_set_limit_lowers_and_raises_within_bounds():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    low = apply_patch(m, [{"op": "set_limit", "name": STEPS, "value": 8}])
    assert low["limits"][STEPS] == 8 and m == DEFAULT_MANIFEST
    high = apply_patch(m, [{"op": "set_limit", "name": CALLS, "value": 18}])
    assert high["limits"][CALLS] == 18
    legacy = {k: v for k, v in m.items() if k != "limits"}
    assert apply_patch(legacy, [{"op": "set_limit", "name": STEPS, "value": 14}])["limits"] == {STEPS: 14, CALLS: 12}


@pytest.mark.parametrize(
    "op",
    [
        {"op": "set_limit", "name": STEPS, "value": 21},  # above bound
        {"op": "set_limit", "name": STEPS, "value": 3},
        {"op": "set_limit", "name": CALLS, "value": 31},
        {"op": "set_limit", "name": CALLS, "value": 12.5},
        {"op": "set_limit", "name": CALLS, "value": True},
        {"op": "set_limit", "name": "max_identical_calls_per_turn", "value": 5},  # unknown name
        {"op": "set_limit", "name": STEPS},  # missing field
        {"op": "set_limit", "name": STEPS, "value": 9, "extra": 1},
    ],
)
def test_set_limit_rejected(op):
    with pytest.raises(PatchError):
        apply_patch(DEFAULT_MANIFEST, [op])


def test_set_limit_risk_tiers():
    m = DEFAULT_MANIFEST
    lower = {"op": "set_limit", "name": STEPS, "value": 8}
    same = {"op": "set_limit", "name": STEPS, "value": 10}
    raise_ = {"op": "set_limit", "name": CALLS, "value": 18}
    assert classify_risk([lower], m) == "low"
    assert classify_risk([same], m) == "low"
    assert classify_risk([raise_], m) == "needs_approval"
    assert classify_risk([lower, raise_], m) == "needs_approval"
    assert classify_risk([lower], None) == "needs_approval"  # direction unknown: conservative
    # lower then raise back above the ORIGINAL is still a raise overall
    assert classify_risk([lower, {"op": "set_limit", "name": STEPS, "value": 12}], m) == "needs_approval"
    # relative to a manifest that already has a higher limit, a smaller value is a lowering
    assert classify_risk([{"op": "set_limit", "name": CALLS, "value": 15}], with_limits(10, 18)) == "low"
    validate_ops([lower])


def test_llm_parse_accepts_set_limit_and_prompt_mentions_it():
    raw = json.dumps({"ops": [{"op": "set_limit", "name": STEPS, "value": 12}], "rationale": "r"})
    assert LLMProposer.parse(raw)["ops"][0]["op"] == "set_limit"
    with pytest.raises(PatchError):
        LLMProposer.parse(json.dumps({"ops": [{"op": "set_limit", "name": STEPS}]}))
    dev = evaluate_limits("baseline", "dev")
    prompt = LLMProposer().build_prompt(DEFAULT_MANIFEST, [], {}, tool_failures=[], limits_failures=dev["failures"])
    assert "set_limit" in prompt and "false_cap" in prompt and STEPS in prompt


def test_llm_call_budget_unchanged_by_limits_section(tmp_path):
    calls = []

    def client(prompt, model, max_tokens):
        calls.append(prompt)
        return json.dumps({"ops": [], "rationale": "none"})

    p = LLMProposer(cache_dir=tmp_path, max_calls=1, client=client)
    dev = evaluate_limits("baseline", "dev")
    p.propose(DEFAULT_MANIFEST, [], {}, tool_failures=[], limits_failures=dev["failures"])
    p.propose(DEFAULT_MANIFEST, [], {}, tool_failures=[], limits_failures=dev["failures"])  # cached
    assert len(calls) == 1


# ---------------------------------------------------------------- evaluator


def test_would_cap_mirrors_runtime_rules():
    t = {"llm_steps": 10, "tool_calls": 12, "max_identical_calls": 3}
    assert not would_cap(t, (10, 12))  # exactly at the limits completes
    assert would_cap({**t, "llm_steps": 11}, (10, 12))
    assert would_cap({**t, "tool_calls": 13}, (10, 12))
    assert would_cap({**t, "max_identical_calls": 4}, (20, 30))  # identical rule is fixed in code
    assert not would_cap({**t, "llm_steps": 11}, (11, 12))


def test_would_cap_agrees_with_runtime_check_call():
    """Replay a trace through the real check_call and compare with the evaluator."""

    def runtime_capped(trace, limits):
        msgs = [HumanMessage(content="hi")]
        n_calls = trace["tool_calls"]
        ident = trace["max_identical_calls"]
        for j in range(n_calls):
            args = {"x": "same"} if j < ident else {"x": str(j)}
            c = {"name": "t", "args": args, "id": f"c{j}", "type": "tool_call"}
            if tl.check_call(msgs, c, limits):
                return True
            msgs.append(AIMessage(content="", tool_calls=[c]))
        return trace["llm_steps"] > limits[0]

    for _, sc in ALL:
        for limits in ((10, 12), (14, 18), (4, 6), (20, 30)):
            assert runtime_capped(sc, limits) == would_cap(sc, limits), (sc["id"], limits)


def test_baseline_has_false_caps_but_no_missed_caps():
    for split in SPLITS:
        r = evaluate_limits("baseline", split)
        assert r["accuracy"] < 1 and r["false_caps"] > 0
        assert r["missed_caps"] == 0 and r["critical_violations"] == 0
    assert evaluate_limits("baseline", "dev")["accuracy"] < 0.75


def test_reference_limits_are_perfect_on_both_splits():
    ref = with_limits(14, 18)
    for split in SPLITS:
        r = evaluate_limits(ref, split)
        assert r["accuracy"] == 1.0 and r["failures"] == []


def test_failure_shape_and_manifest_forms():
    f = evaluate_limits("baseline")["failures"][0]
    assert set(f) == {"id", "kind", "outcome", "llm_steps", "tool_calls", "max_identical_calls",
                      "capped_by", "severity", "critical_violation"}
    assert set(evaluate_limits_all("baseline")) == set(SPLITS)
    assert evaluate_limits(DEFAULT_MANIFEST, "dev") == evaluate_limits("baseline", "dev")


def test_load_unknown_split():
    with pytest.raises(ValueError):
        load_limits_scenarios("nope")


@pytest.mark.parametrize("split,sc", ALL, ids=[sc["id"] for _, sc in ALL])
def test_scenario_honesty(split, sc):
    assert set(sc) == {"id", "kind", "llm_steps", "tool_calls", "max_identical_calls", "expected", "severity", "tags"}
    assert sc["kind"] in ("legit", "runaway")
    assert sc["expected"] == ("complete" if sc["kind"] == "legit" else "capped")
    assert sc["severity"] in ("critical", "normal")
    assert isinstance(sc["tags"], list) and sc["tags"] and all(isinstance(t, str) for t in sc["tags"])
    for k in ("llm_steps", "tool_calls", "max_identical_calls"):
        assert isinstance(sc[k], int) and not isinstance(sc[k], bool) and sc[k] >= 0
    assert sc["llm_steps"] >= 1
    # the trace is physically possible: every step but the last made at least one call
    assert sc["tool_calls"] >= sc["llm_steps"] - 1
    assert sc["max_identical_calls"] <= sc["tool_calls"]
    assert (sc["max_identical_calls"] > 0) == (sc["tool_calls"] > 0)
    # a legit trace must be completable within the widest limits a human allows; a runaway must be
    # stopped by the reference limits (14 steps, 18 calls) that cover every legit trace
    widest = (INVARIANTS["limits_bounds"][STEPS][1], INVARIANTS["limits_bounds"][CALLS][1])
    assert would_cap(sc, (14, 18)) == (sc["kind"] == "runaway")
    if sc["kind"] == "legit":
        assert not would_cap(sc, widest)
    if sc["kind"] == "runaway":
        assert sc["severity"] == "critical"


def test_ids_unique_traces_distinct_and_split_shapes():
    ids = [sc["id"] for _, sc in ALL]
    assert len(ids) == len(set(ids))
    key = lambda sc: (sc["llm_steps"], sc["tool_calls"], sc["max_identical_calls"], sc["kind"])  # noqa: E731
    dev, held = load_limits_scenarios("dev"), load_limits_scenarios("heldout")
    assert len(dev) >= 20 and len(held) >= 10
    assert not {key(s) for s in dev} & {key(s) for s in held}
    assert len({key(s) for s in dev}) == len(dev) and len({key(s) for s in held}) == len(held)
    for split in (dev, held):
        assert {s["kind"] for s in split} == {"legit", "runaway"}
        # headroom: legit flows the baseline wrongly caps
        assert any(s["kind"] == "legit" and would_cap(s, (10, 12)) for s in split)


def test_legit_and_runaway_are_separable_by_dev_raise():
    """Legit traces need <=14 steps / <=18 calls; runaways need more, so a bounded raise exists."""
    for _, sc in ALL:
        if sc["kind"] == "legit":
            assert sc["llm_steps"] <= 14 and sc["tool_calls"] <= 18
        elif sc["max_identical_calls"] <= 3:
            assert sc["llm_steps"] >= 15 or sc["tool_calls"] >= 21


# ---------------------------------------------------------------- gate


def test_gate_passes_proper_raise():
    base = evaluate_limits_all("baseline")
    res = compare_limits(base, evaluate_limits_all(with_limits(14, 18)))
    assert res["gate"] == "pass" and res["passed"]
    assert res["deltas"]["dev"]["false_caps"] < 0


def test_negative_control_raise_past_a_runaway_fails_gate():
    base = evaluate_limits_all("baseline")
    too_high = evaluate_limits_all(with_limits(20, 30))  # lets the 16-step and 22-call runaways through
    res = compare_limits(base, too_high)
    assert res["gate"] == "fail" and not res["passed"]
    assert any("not capped" in r for r in res["reasons"])
    assert too_high["dev"]["missed_caps"] > 0
    only_steps = evaluate_limits_all(with_limits(16, 18))  # the 16-step slow loop slips through
    assert only_steps["dev"]["missed_caps"] > 0 and compare_limits(base, only_steps)["gate"] == "fail"
    only_calls = evaluate_limits_all(with_limits(10, 22))
    assert only_calls["dev"]["missed_caps"] > 0 and compare_limits(base, only_calls)["gate"] == "fail"


def test_gate_fails_on_accuracy_drop_and_rising_false_caps():
    ref = evaluate_limits_all(with_limits(14, 18))
    lowered = evaluate_limits_all(with_limits(6, 8))  # strangles legit flows
    res = compare_limits(ref, lowered)
    assert res["gate"] == "fail"
    assert any("accuracy" in r for r in res["reasons"]) and any("false caps" in r for r in res["reasons"])


def test_gate_accepts_single_results():
    b, c = evaluate_limits("baseline"), evaluate_limits(with_limits(14, 18))
    assert compare_limits(b, c)["passed"] and "dev" in compare_limits(b, c)["deltas"]


def test_lowering_a_limit_that_costs_nothing_passes():
    base = evaluate_limits_all(with_limits(14, 18))
    assert compare_limits(base, evaluate_limits_all(with_limits(14, 18)))["passed"]


# ---------------------------------------------------------------- proposer


def test_limits_proposer_smallest_covering_raise():
    dev = evaluate_limits("baseline", "dev")
    out = LimitsHeuristicProposer().propose(copy.deepcopy(DEFAULT_MANIFEST), dev["failures"], {})
    ops = {o["name"]: o["value"] for o in out["ops"]}
    assert ops == {STEPS: 14, CALLS: 18}
    patched = apply_patch(DEFAULT_MANIFEST, out["ops"])
    assert evaluate_limits(patched, "dev")["accuracy"] == 1.0
    assert classify_risk(out["ops"], DEFAULT_MANIFEST) == "needs_approval"
    # a second round proposes nothing
    again = LimitsHeuristicProposer().propose(patched, evaluate_limits(patched, "dev")["failures"], {})
    assert again["ops"] == []


def test_limits_proposer_never_exceeds_bound_and_ignores_runaways():
    fake = [
        {"id": "x", "kind": "legit", "outcome": "false_cap", "llm_steps": 25, "tool_calls": 40, "max_identical_calls": 1},
        {"id": "y", "kind": "runaway", "outcome": "missed_cap", "llm_steps": 5, "tool_calls": 5, "max_identical_calls": 1},
        {"id": "z", "kind": "legit", "outcome": "false_cap", "llm_steps": 11, "tool_calls": 12, "max_identical_calls": 1},
    ]
    out = LimitsHeuristicProposer().propose(copy.deepcopy(DEFAULT_MANIFEST), fake, {})
    assert out["ops"] == []  # 25 and 40 exceed the bounds; nothing else is falsely capped
    out = LimitsHeuristicProposer().propose(copy.deepcopy(DEFAULT_MANIFEST), fake[1:], {})
    assert out["ops"] == [{"op": "set_limit", "name": STEPS, "value": 11}]


def test_limits_proposer_loads_no_scenarios(monkeypatch):
    import evals.limits_eval as le

    seen = []
    real = le.load_limits_scenarios
    monkeypatch.setattr(le, "load_limits_scenarios", lambda s: seen.append(s) or real(s))
    LimitsHeuristicProposer().propose(copy.deepcopy(DEFAULT_MANIFEST), [], {})
    assert seen == []
    assert CompositeProposer([HeuristicProposer(), ToolHeuristicProposer(), LimitsHeuristicProposer()]).name == (
        "heuristic+tool-heuristic+limits-heuristic"
    )


# ---------------------------------------------------------------- cycle (in-memory fake store)


class FakeCollection:
    def __init__(self, store):
        self.store = store

    def update_one(self, filt, upd):
        self.store.docs[filt["version"]].update(upd["$set"])


class FakeStore:
    """Just the HarnessConfigStore surface run_cycle uses; never touches Mongo."""

    def __init__(self, manifest=None):
        self.docs = {1: {"version": 1, "parent": None, "status": "active", "scores": {},
                         "manifest": copy.deepcopy(manifest or DEFAULT_MANIFEST)}}
        self._collection = FakeCollection(self)

    def get_active(self):
        act = [d for d in self.docs.values() if d["status"] == "active"]
        return max(act, key=lambda d: d["version"]) if act else None

    def get(self, v):
        return self.docs.get(v)

    def propose(self, manifest, parent, reason, author):
        assert not validate_manifest(manifest)
        v = max(self.docs) + 1
        self.docs[v] = {"version": v, "parent": parent, "status": "candidate", "manifest": copy.deepcopy(manifest),
                        "reason": reason, "author": author, "scores": {}}
        return v

    def activate(self, v):
        for d in self.docs.values():
            if d["status"] == "active":
                d["status"] = "retired"
        self.docs[v]["status"] = "active"

    def set_status(self, v, status):
        self.docs[v]["status"] = status


class FixedProposer:
    surface = "limits"

    def __init__(self, ops):
        self.ops, self.name = ops, "fixed"

    def propose(self, manifest, failures, split_summary):
        return {"ops": self.ops, "rationale": "test"}


def test_limits_cycle_raise_gated_and_needs_approval():
    store = FakeStore()
    r = run_cycle(store, LimitsHeuristicProposer(), auto_activate_low_risk=True, surface="limits")
    assert r["gate"]["passed"] and r["limits_gate"]["passed"] and r["gate"]["limits"] == "pass"
    assert r["risk"] == "needs_approval" and r["status"] == "candidate"  # never auto-activated
    assert store.get_active()["version"] == 1
    scores = store.docs[r["version"]]["scores"]
    assert {"limits_baseline", "limits_candidate", "limits_gate"} <= set(scores)
    assert set(scores["limits_candidate"]) == set(SPLITS)
    assert scores["limits_candidate"]["dev"]["accuracy"] == 1.0
    assert "failed_ids" not in scores["limits_candidate"]["heldout"]
    assert r["limits_baseline"]["dev"]["accuracy"] < 1
    # risk and tools surfaces were not touched, so they are not re-litigated
    assert r["risk_gate"].get("untouched") and r["tools_gate"].get("untouched")


def test_limits_cycle_lowering_is_low_risk_and_auto_activates_when_gate_passes():
    store = FakeStore(with_limits(14, 18))
    r = run_cycle(store, FixedProposer([{"op": "set_limit", "name": STEPS, "value": 14}]),
                  auto_activate_low_risk=True, surface="limits")
    assert r["risk"] == "low" and r["gate"]["passed"] and r["status"] == "active"
    r2 = run_cycle(store, FixedProposer([{"op": "set_limit", "name": STEPS, "value": 6}]),
                   auto_activate_low_risk=True, surface="limits")
    assert r2["risk"] == "low" and not r2["gate"]["passed"] and r2["status"] == "rejected"  # strangles legit flows
    assert store.get_active()["version"] == r["version"]


def test_limits_cycle_over_raise_fails_gate_and_is_rejected():
    store = FakeStore()
    r = run_cycle(store, FixedProposer([{"op": "set_limit", "name": STEPS, "value": 20},
                                     {"op": "set_limit", "name": CALLS, "value": 30}]),
                  auto_activate_low_risk=True, surface="limits")
    assert not r["gate"]["passed"] and r["status"] == "rejected"
    assert any(x.startswith("limits:") and "not capped" in x for x in r["gate"]["reasons"])
    assert store.get_active()["version"] == 1


def test_out_of_bounds_patch_never_reaches_store():
    store = FakeStore()
    r = run_cycle(store, FixedProposer([{"op": "set_limit", "name": CALLS, "value": 31}]), surface="limits")
    assert r["status"] == "rejected" and r["version"] is None and len(store.docs) == 1


def test_all_surface_includes_limits_and_surface_filter_drops_foreign_ops():
    store = FakeStore()
    r = run_cycle(store, CompositeProposer([HeuristicProposer(), ToolHeuristicProposer(), LimitsHeuristicProposer()]),
                  surface="all")
    kinds = {o["op"] for o in r["ops"]}
    assert "set_limit" in kinds and "disable_tool" in kinds
    assert r["gate"]["passed"] and {"risk", "tools", "limits"} <= set(r["gate"])
    assert r["risk"] == "needs_approval"
    # --surface tools must not change limits even if the proposer wants to
    r2 = run_cycle(FakeStore(), CompositeProposer([ToolHeuristicProposer(), LimitsHeuristicProposer()]), surface="tools")
    assert all(o["op"] != "set_limit" for o in r2["ops"])


def test_cycle_on_legacy_manifest_without_limits():
    legacy = {k: v for k, v in copy.deepcopy(DEFAULT_MANIFEST).items() if k != "limits"}
    store = FakeStore(legacy)
    r = run_cycle(store, LimitsHeuristicProposer(), surface="limits")
    assert r["ops"] and r["gate"]["passed"]
    assert store.docs[r["version"]]["manifest"]["limits"] == {STEPS: 14, CALLS: 18}


def test_cli_accepts_limits_surface():
    from harness_evolve.cli import main

    with pytest.raises(SystemExit):  # argparse rejects an unknown surface ...
        main(["--surface", "nope", "status"])
    import harness_evolve.cli as cli
    import inspect

    assert "LimitsHeuristicProposer" in inspect.getsource(cli)


# ---------------------------------------------------------------- telemetry + real graph


def _tool_call(name, args, id):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


def _script(n_steps, calls_per_step):
    """n_steps - 1 tool-calling AIMessages (read-only tools, distinct args) then a final answer."""
    out, k = [], 0
    for s in range(n_steps - 1):
        calls = []
        for _ in range(calls_per_step[s]):
            calls.append(("explain_insurance_term", {"term": f"term-{k}"}, f"c{k}"))
            k += 1
        out.append(ai(*calls))
    out.append(AIMessage(content="done"))
    return out


@pytest.fixture
def main_mod(monkeypatch):
    os.environ.setdefault("RUNNER_MODE", "aer")
    monkeypatch.setenv("RUNNER_ALLOW_DIRECT_TOOL_EXECUTION", "true")  # read-only explain_insurance_term only
    main = pytest.importorskip("agent_agent46.main")
    seen = {"step": [], "tool": []}
    monkeypatch.setattr(observability, "step_cap_hit", lambda **kw: seen["step"].append(kw))
    monkeypatch.setattr(observability, "tool_limit_hit", lambda rule, tool, **kw: seen["tool"].append((rule, tool, kw)))

    def use(manifest, version=7):
        monkeypatch.setattr(main, "load_active_manifest", lambda store: (version, manifest))

    return main, use, seen


def run_flow(main, llm_script):
    llm = RecordingLLM(bound=[], script=llm_script)
    graph = main.build_agent(llm=llm)
    return graph.invoke({"messages": [HumanMessage(content="What is a deductible?")]}, {"configurable": {"thread_id": "t"}})


def measured(out):
    return tl.trace_from_messages(out["messages"])


def test_real_graph_trace_matches_legit_dev_scenario(main_mod):
    """Anchor: the real graph's measured trace for a 7-call, 8-step flow equals dev scenario ld04."""
    main, use, seen = main_mod
    use(copy.deepcopy(DEFAULT_MANIFEST))
    out = run_flow(main, _script(8, [1] * 7))
    trace = measured(out)
    sc = {s["id"]: s for s in load_limits_scenarios("dev")}["ld04"]
    assert sc["kind"] == "legit" and "real_trace" in sc["tags"]
    assert trace == {k: sc[k] for k in ("llm_steps", "tool_calls", "max_identical_calls")}
    assert out["messages"][-1].content == "done" and seen["step"] == [] and seen["tool"] == []
    assert not would_cap(trace, tl.resolve_limits(DEFAULT_MANIFEST))


def test_real_graph_long_flow_capped_at_baseline_and_completes_when_raised(main_mod):
    """ld10 (13 steps, 13 calls): the baseline graph hands off at step 11; the raised one completes."""
    main, use, seen = main_mod
    sc = {s["id"]: s for s in load_limits_scenarios("dev")}["ld10"]
    use(copy.deepcopy(DEFAULT_MANIFEST), version=3)
    out = run_flow(main, _script(13, [2] + [1] * 11))
    assert "human" in out["messages"][-1].content.lower()
    assert len(seen["step"]) == 1
    assert seen["step"][0] == {"steps": 10, "calls": 11, "limit": 10, "harness_version": 3}
    assert would_cap(sc, (10, 12))  # the evaluator agrees with what the real graph just did
    # raised manifest: same script completes, and the measured trace equals the scenario
    seen["step"].clear()
    use(with_limits(14, 18), version=8)
    out = run_flow(main, _script(13, [2] + [1] * 11))
    assert out["messages"][-1].content == "done" and seen["step"] == []
    assert measured(out) == {k: sc[k] for k in ("llm_steps", "tool_calls", "max_identical_calls")}
    assert not would_cap(sc, (14, 18))


def test_real_graph_tool_call_cap_follows_manifest(main_mod):
    """ld12-like: 9 steps with 13 calls. The 13th call is refused at baseline, runs when raised."""
    main, use, seen = main_mod
    per_step = [2, 2, 2, 2, 2, 1, 1, 1]  # 13 calls over 8 tool-calling steps, 9 steps in all
    use(copy.deepcopy(DEFAULT_MANIFEST), version=4)
    out = run_flow(main, _script(9, per_step))
    errs = [m for m in out["messages"] if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(errs) == 1 and "safety limit" in errs[0].content
    rule, tool, kw = seen["tool"][0]
    assert rule == tl.RULE_TURN and kw["limit"] == 12 and kw["harness_version"] == 4 and kw["steps"] == 8 and kw["calls"] == 13
    seen["tool"].clear()
    use(with_limits(10, 18), version=9)
    out = run_flow(main, _script(9, per_step))
    assert not [m for m in out["messages"] if isinstance(m, ToolMessage) and m.status == "error"]
    assert seen["tool"] == [] and measured(out)["tool_calls"] == 13


def test_real_graph_lowered_limit_caps_earlier(main_mod):
    main, use, seen = main_mod
    use(with_limits(5, 12))
    out = run_flow(main, _script(8, [1] * 7))
    assert "human" in out["messages"][-1].content.lower()
    assert seen["step"][0]["limit"] == 5


def test_real_graph_bad_stored_limits_fall_back_to_baseline(main_mod, caplog):
    main, use, seen = main_mod
    bad = copy.deepcopy(DEFAULT_MANIFEST)
    bad["limits"] = {STEPS: 999, CALLS: 999}  # would normally be filtered at load; the resolver still fails safe
    use(bad)
    with caplog.at_level("ERROR"):
        out = run_flow(main, _script(13, [2] + [1] * 11))
    assert "human" in out["messages"][-1].content.lower() and seen["step"][0]["limit"] == 10
    assert any(r.levelname == "ERROR" for r in caplog.records)
