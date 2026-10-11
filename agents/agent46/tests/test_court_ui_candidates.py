"""Candidates created outside the browser (CLI, demo, cycle) get the same summary card in Policy versions."""

import copy
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402
from test_court import FakeDb  # noqa: E402
from test_court_ui import TOKEN, FakeStore, _good  # noqa: E402

from agent_agent46.harness_config import DEFAULT_MANIFEST  # noqa: E402
from harness_evolve.court_ui.app import create_app, ops_from_manifests  # noqa: E402
from harness_evolve.patch import apply_patch  # noqa: E402

FIRST_RULE = "base-rule"  # the round-trip parent below carries this rule
DROP_TOOL = {"op": "disable_tool", "intent": "claims", "tool": "get_quote"}

ROUND_TRIP_CASES = [
    [{"op": "add_keyword", "level": "high", "keyword": "Arson"}],
    [{"op": "add_keyword", "level": "medium", "keyword": "flood"}],
    [{"op": "set_threshold", "name": "medium_score", "value": 0.55}],
    [{"op": "set_threshold", "name": "auto_approve_limit", "value": 100}],
    [{"op": "add_rule", "id": "r-new", "text": "Be careful"}],
    [{"op": "remove_rule", "id": FIRST_RULE}],
    [DROP_TOOL],
    [{"op": "set_limit", "name": "max_llm_steps_per_turn", "value": 4}],
    [{"op": "set_limit", "name": "max_tool_calls_per_turn", "value": 14}],
    [{"op": "add_keyword", "level": "high", "keyword": "arson"}, {"op": "set_threshold", "name": "medium_score", "value": 0.5},
     {"op": "add_rule", "id": "r2", "text": "x"}, DROP_TOOL,
     {"op": "set_limit", "name": "max_llm_steps_per_turn", "value": 5}],
]


@pytest.mark.parametrize("ops", ROUND_TRIP_CASES)
def test_ops_from_manifests_round_trips(ops):
    parent = apply_patch(copy.deepcopy(DEFAULT_MANIFEST), [{"op": "add_rule", "id": FIRST_RULE, "text": "old"}])
    child = apply_patch(parent, ops)
    got = ops_from_manifests(parent, child)
    assert got and got == ops_from_manifests(parent, child)
    assert apply_patch(parent, got) == child
    assert {o["op"] for o in got} == {o["op"] for o in ops}


def test_ops_from_manifests_enable_tool_and_missing_keys():
    base = copy.deepcopy(DEFAULT_MANIFEST)
    narrowed = apply_patch(base, [DROP_TOOL])
    got = ops_from_manifests(narrowed, base)
    assert got == [{"op": "enable_tool", "intent": "claims", "tool": "get_quote"}]
    assert apply_patch(narrowed, got) == base
    legacy = {k: v for k, v in base.items() if k not in ("limits", "tool_access")}
    assert ops_from_manifests(legacy, base) == [] and ops_from_manifests(base, legacy) == []
    lowered = apply_patch(legacy, [{"op": "set_limit", "name": "max_llm_steps_per_turn", "value": 5}])
    assert ops_from_manifests(legacy, lowered) == [{"op": "set_limit", "name": "max_llm_steps_per_turn", "value": 5}]
    with pytest.raises(ValueError):
        ops_from_manifests(None, base)


def _client(tmp_path, ops, status="candidate", scores="default", parent=1, raw_keyword=None):
    store = FakeStore()
    manifest = apply_patch(store.get(1)["manifest"], ops)
    if raw_keyword:  # a hand-edited stored manifest need not pass patch validation
        manifest["keywords"]["high"].append(raw_keyword)
    default_scores = {"baseline": {"dev": _good(0.8, 1), "heldout": _good(0.7, 0)},
                      "candidate": {"dev": _good(0.9, 0), "heldout": _good(0.75, 0)},
                      "gate": {"gate": "pass", "passed": True, "reasons": []}, "risk": "needs_approval"}
    store.docs.append({"version": 2, "parent": parent, "status": status, "manifest": manifest,
                       "reason": "because; reasons <b>x</b>", "author": "proposer:heuristic",
                       "scores": default_scores if scores == "default" else scores})
    app = create_app(FakeDb(), store, token=TOKEN, docket_path=tmp_path / "d.json")
    return TestClient(app, follow_redirects=True)


def _panel(client):
    return client.get("/").text.split('id="versions"')[1].split('id="audit"')[0]


def test_cli_candidate_renders_card_with_scores_and_forms(tmp_path):
    ops = [{"op": "set_threshold", "name": "medium_amount", "value": 6000}, DROP_TOOL]
    c = _client(tmp_path, ops, raw_keyword="<script>alert(1)</script>")
    html = c.get("/").text
    panel = html.split('id="versions"')[1].split('id="audit"')[0]
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)" in panel
    assert "Proposed change v2" in panel and "What would change" in panel
    assert "Raise the medium-risk claim amount" in panel and "Stop offering 1 tool" in panel
    assert 'class="badge b-warn">Loosens a safeguard<' in panel and "Needs your approval" in panel
    assert "Accuracy on known cases 80% to 90%" in panel and "unseen cases 70% to 75%" in panel
    assert "<td>2</td>" in panel  # compact table row kept
    forms = re.findall(r"<form .*?</form>", panel, re.S)
    assert len(forms) == 2 and any("/version/2/approve" in f for f in forms) and any("/version/2/reject" in f for f in forms)
    for f in forms:
        assert f'name="csrf" value="{TOKEN}"' in f and 'name="confirm"' in f
    assert "reasons &lt;b&gt;x&lt;/b&gt;" in panel and "<b>x</b>" not in panel


def test_lowering_has_no_loosens_badge_and_limit_raise_has(tmp_path):
    low = _panel(_client(tmp_path, [{"op": "set_threshold", "name": "medium_amount", "value": 2000},
                                    {"op": "set_limit", "name": "max_llm_steps_per_turn", "value": 5}]))
    assert "Lower the medium-risk claim amount" in low and "Cap at 5" in low
    assert "Loosens a safeguard" not in low.split("Proposed change v2")[1]
    high = _panel(_client(tmp_path, [{"op": "set_limit", "name": "max_tool_calls_per_turn", "value": 14}]))
    assert "Allow up to 14" in high and 'class="badge b-warn">Loosens a safeguard<' in high


def test_rejected_version_shows_gate_reasons_and_changes(tmp_path):
    sc = {"baseline": {"dev": _good(0.9), "heldout": _good(0.9)}, "candidate": {"dev": _good(0.5, 2), "heldout": _good(0.4, 1)},
          "gate": {"gate": "fail", "passed": False, "reasons": ["dev: 2 critical violation(s) <i>x</i>"]}, "risk": "low"}
    panel = _panel(_client(tmp_path, [{"op": "add_keyword", "level": "high", "keyword": "hail"}], status="rejected", scores=sc))
    assert "<summary>What changed</summary>" in panel and "hail" in panel
    assert "Safety check reasons" in panel and "2 critical violation(s) &lt;i&gt;x&lt;/i&gt;" in panel
    assert "Approve and go live" not in panel and "<i>x</i>" not in panel


def test_retired_version_has_what_changed_expander(tmp_path):
    panel = _panel(_client(tmp_path, [{"op": "set_threshold", "name": "medium_score", "value": 0.5}], status="retired"))
    assert "<summary>What changed</summary>" in panel and "review threshold" in panel
    assert "Safety check reasons" not in panel


@pytest.mark.parametrize("parent", [99, None])
def test_candidate_missing_parent_falls_back(tmp_path, parent):
    r = _client(tmp_path, [{"op": "add_keyword", "level": "high", "keyword": "hail"}], parent=parent).get("/")
    assert r.status_code == 200
    panel = r.text.split('id="versions"')[1].split('id="audit"')[0]
    assert "because; reasons" in panel and "not available" in panel


@pytest.mark.parametrize("sc", [None, "junk", {"gate": "x", "candidate": 5, "baseline": []}, {}])
def test_candidate_bad_scores_never_raises(tmp_path, sc):
    assert _client(tmp_path, [{"op": "add_keyword", "level": "high", "keyword": "hail"}], scores=sc).get("/").status_code == 200
