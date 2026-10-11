import copy
import json
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_agent46.harness_config import DEFAULT_MANIFEST, HarnessConfigStore  # noqa: E402
from agent_agent46.tool_access import ALL_TOOLS, REQUIRED_TOOLS  # noqa: E402
from evals import tool_eval  # noqa: E402
from harness_evolve.loop import run_cycle  # noqa: E402
from harness_evolve.patch import PatchError, apply_patch, classify_risk  # noqa: E402
from harness_evolve.proposer import CompositeProposer, HeuristicProposer, LLMProposer, ToolHeuristicProposer  # noqa: E402

URI = "mongodb://localhost:53087/?directConnection=true"


def _optional_tool(intent):
    return sorted(ALL_TOOLS - REQUIRED_TOOLS[intent])[0]


def test_disable_and_enable_tool():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    t = _optional_tool("info")
    off = apply_patch(m, [{"op": "disable_tool", "intent": "info", "tool": t}])
    assert m == DEFAULT_MANIFEST
    assert t not in off["tool_access"]["intents"]["info"]["allow"]
    assert t in off["tool_access"]["intents"]["quote"]["allow"]
    on = apply_patch(off, [{"op": "enable_tool", "intent": "info", "tool": t}])
    assert t in on["tool_access"]["intents"]["info"]["allow"]


def test_patch_on_manifest_without_tool_access():
    m = {k: v for k, v in copy.deepcopy(DEFAULT_MANIFEST).items() if k != "tool_access"}
    t = _optional_tool("info")
    new = apply_patch(m, [{"op": "disable_tool", "intent": "info", "tool": t}])
    assert t not in new["tool_access"]["intents"]["info"]["allow"]


def test_patch_errors():
    t = _optional_tool("info")
    off = apply_patch(DEFAULT_MANIFEST, [{"op": "disable_tool", "intent": "info", "tool": t}])
    bad = [
        (DEFAULT_MANIFEST, {"op": "disable_tool", "intent": "nope", "tool": t}),
        (DEFAULT_MANIFEST, {"op": "disable_tool", "intent": "info", "tool": "rm_rf"}),
        (DEFAULT_MANIFEST, {"op": "enable_tool", "intent": "info", "tool": t}),  # already allowed
        (off, {"op": "disable_tool", "intent": "info", "tool": t}),  # already disabled
        (DEFAULT_MANIFEST, {"op": "disable_tool", "intent": "info"}),  # missing field
    ]
    for m, op in bad:
        with pytest.raises(PatchError):
            apply_patch(m, [op])


@pytest.mark.parametrize("intent", ["claims", "quote", "info", "general"])
def test_required_tools_cannot_be_disabled(intent):
    for tool in sorted(REQUIRED_TOOLS[intent])[:3]:
        with pytest.raises(PatchError):
            apply_patch(DEFAULT_MANIFEST, [{"op": "disable_tool", "intent": intent, "tool": tool}])


def test_classify_risk_tool_ops():
    t = _optional_tool("info")
    dis = {"op": "disable_tool", "intent": "info", "tool": t}
    en = {"op": "enable_tool", "intent": "info", "tool": t}
    assert classify_risk([dis]) == "low"
    assert classify_risk([en]) == "needs_approval"
    assert classify_risk([dis, en]) == "needs_approval"
    assert classify_risk([dis, {"op": "add_keyword", "level": "high", "keyword": "x"}]) == "low"


def test_tool_proposer_uses_only_dev_and_handles_missing_tool_access(monkeypatch):
    loaded = []
    real = tool_eval.load_tool_scenarios
    monkeypatch.setattr(tool_eval, "load_tool_scenarios", lambda s: loaded.append(s) or real(s))
    m = {k: v for k, v in copy.deepcopy(DEFAULT_MANIFEST).items() if k != "tool_access"}
    dev = tool_eval.evaluate_tools(m, "dev")
    out = ToolHeuristicProposer().propose(m, dev["failures"], {})
    assert loaded == ["dev"]
    assert out["ops"]
    assert all(o["op"] == "disable_tool" and o["tool"] not in REQUIRED_TOOLS[o["intent"]] for o in out["ops"])
    pairs = [(o["intent"], o["tool"]) for o in out["ops"]]
    assert len(pairs) == len(set(pairs))
    # applying the proposal and re-asking proposes nothing already disabled
    again = ToolHeuristicProposer().propose(apply_patch(m, out["ops"]), dev["failures"], {})
    assert not set(map(tuple, ((o["intent"], o["tool"]) for o in again["ops"]))) & set(pairs)


def test_heuristic_reads_no_scenario_files_via_composite(monkeypatch):
    seen = []
    real = tool_eval.load_tool_scenarios
    monkeypatch.setattr(tool_eval, "load_tool_scenarios", lambda s: seen.append(s) or real(s))
    p = ToolHeuristicProposer()
    p.propose(copy.deepcopy(DEFAULT_MANIFEST), [], {})
    assert seen == []
    assert CompositeProposer([HeuristicProposer(), p]).name == "heuristic+tool-heuristic"


def test_llm_parse_accepts_tool_ops_and_prompt_mentions_them():
    t = _optional_tool("info")
    raw = json.dumps({"ops": [{"op": "disable_tool", "intent": "info", "tool": t}], "rationale": "r"})
    assert LLMProposer.parse(raw)["ops"][0]["op"] == "disable_tool"
    with pytest.raises(PatchError):
        LLMProposer.parse(json.dumps({"ops": [{"op": "disable_tool", "intent": "info"}]}))
    prompt = LLMProposer().build_prompt(DEFAULT_MANIFEST, [], {}, tool_failures=[])
    assert "disable_tool" in prompt and "enable_tool" in prompt


@pytest.fixture
def store():
    s = HarnessConfigStore(URI, f"evolve_tools_test_{uuid.uuid4().hex[:8]}")
    try:
        s._client.admin.command("ping")
    except Exception:
        pytest.skip("local Mongo not available")
    s.seed_if_empty()
    yield s
    s._client.drop_database(s._collection.database.name)
    s._client.close()


class FixedProposer:
    surface = "tools"

    def __init__(self, ops):
        self.ops, self.name = ops, "fixed"

    def propose(self, manifest, failures, split_summary):
        return {"ops": self.ops, "rationale": "test"}


def test_tools_cycle_end_to_end(store):
    r = run_cycle(store, ToolHeuristicProposer(), auto_activate_low_risk=True, surface="tools")
    assert r["ops"] and all(o["op"] == "disable_tool" for o in r["ops"])
    assert r["gate"]["passed"], r["gate"]
    assert r["tools_gate"]["passed"] and r["risk_gate"]["passed"]
    scores = store.get(r["version"])["scores"]
    assert {"baseline", "candidate", "tools_baseline", "tools_candidate", "gate"} <= set(scores)
    assert set(scores["tools_candidate"]) == {"dev", "heldout"}
    assert scores["tools_candidate"]["dev"]["accuracy"] >= scores["tools_baseline"]["dev"]["accuracy"]
    # disabling a tool is low risk, so with auto_activate it activates
    assert r["risk"] == "low" and r["status"] == "active"


def test_needs_approval_tool_patch_not_auto_activated(store):
    t = _optional_tool("info")
    # one cycle narrows, activates by hand; second widens it again (needs_approval)
    r1 = run_cycle(store, FixedProposer([{"op": "disable_tool", "intent": "info", "tool": t}]),
                   auto_activate_low_risk=True, surface="tools")
    if r1["status"] != "active":
        pytest.skip("narrowing did not pass the gate; cannot test widening")
    r2 = run_cycle(store, FixedProposer([{"op": "enable_tool", "intent": "info", "tool": t}]),
                   auto_activate_low_risk=True, surface="tools")
    assert r2["risk"] == "needs_approval"
    assert r2["status"] in ("candidate", "rejected")
    assert store.get_active()["version"] == r1["version"]


def test_required_tool_patch_is_rejected(store):
    r = run_cycle(store, FixedProposer([{"op": "disable_tool", "intent": "claims", "tool": "resolve_claim"}]),
                  auto_activate_low_risk=True, surface="tools")
    assert r["status"] == "rejected" and r["version"] is None
    assert not r["gate"]["passed"]
    assert store.get_active()["version"] == 1
    assert len(store.list_versions()) == 1


def test_old_manifest_without_tool_access_is_baseline(store):
    old = {k: v for k, v in copy.deepcopy(DEFAULT_MANIFEST).items() if k != "tool_access"}
    v = store.propose(old, parent=1, reason="legacy", author="test")
    store.activate(v)
    r = run_cycle(store, ToolHeuristicProposer(), surface="tools")
    assert r["parent"] == v and r["ops"]
