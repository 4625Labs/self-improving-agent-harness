import copy
import json
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_agent46.harness_config import DEFAULT_MANIFEST, HarnessConfigStore  # noqa: E402
from harness_evolve import proposer as proposer_mod  # noqa: E402
from harness_evolve.loop import approve, rollback, run_cycle  # noqa: E402
from harness_evolve.patch import PatchError, apply_patch, classify_risk  # noqa: E402
from harness_evolve.proposer import HeuristicProposer, LLMProposer  # noqa: E402

URI = "mongodb://localhost:53087/?directConnection=true"


def test_apply_patch_ops_and_immutability():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    ops = [
        {"op": "add_rule", "id": "r1", "text": "Be careful."},
        {"op": "add_keyword", "level": "high", "keyword": "Arson"},
        {"op": "set_threshold", "name": "medium_score", "value": 0.5},
    ]
    new = apply_patch(m, ops)
    assert m == DEFAULT_MANIFEST
    assert new["rules"] == [{"id": "r1", "text": "Be careful."}]
    assert "arson" in new["keywords"]["high"] and new["thresholds"]["medium_score"] == 0.5
    assert apply_patch(new, [{"op": "remove_rule", "id": "r1"}])["rules"] == []


@pytest.mark.parametrize("ops", [
    [{"op": "drop_table"}],
    [{"op": "add_rule", "id": "r", "text": "x", "extra": 1}],
    [{"op": "add_rule", "id": "r"}],
    [{"op": "add_keyword", "level": "low", "keyword": "x"}],
    [{"op": "set_threshold", "name": "nope", "value": 1}],
    [{"op": "set_threshold", "name": "auto_approve_limit", "value": 5000}],  # invariant
    [{"op": "remove_rule", "id": "missing"}],
    "not a list",
])
def test_apply_patch_rejects(ops):
    with pytest.raises(PatchError):
        apply_patch(DEFAULT_MANIFEST, ops)


def test_classify_risk():
    add = [{"op": "add_rule", "id": "a", "text": "t"}, {"op": "add_keyword", "level": "medium", "keyword": "k"}]
    assert classify_risk(add) == "low"
    assert classify_risk(add + [{"op": "set_threshold", "name": "medium_score", "value": 0.5}]) == "needs_approval"
    assert classify_risk([{"op": "remove_rule", "id": "a"}]) == "needs_approval"


def test_heuristic_never_reads_heldout(monkeypatch):
    seen = []
    real = proposer_mod.load_scenarios
    monkeypatch.setattr(proposer_mod, "load_scenarios", lambda s: seen.append(s) or real(s))
    from evals.run_eval import evaluate

    p = HeuristicProposer()
    dev = evaluate(DEFAULT_MANIFEST, "dev")
    out = p.propose(copy.deepcopy(DEFAULT_MANIFEST), dev["failures"], {})
    assert seen == ["dev"]
    assert out["ops"]
    kws = {o["keyword"] for o in out["ops"] if o["op"] == "add_keyword"}
    assert {"arson", "submerged", "hit and run"} <= kws


def test_llm_parse_is_strict_and_offline():
    assert LLMProposer.parse('x {"ops": [], "rationale": "r"} y')["ops"] == []
    with pytest.raises(PatchError):
        LLMProposer.parse('{"ops": [{"op": "exec", "cmd": "rm"}]}')
    with pytest.raises(PatchError):
        LLMProposer.parse("no json")


@pytest.fixture
def store():
    s = HarnessConfigStore(URI, f"evolve_test_{uuid.uuid4().hex[:8]}")
    try:
        s._client.admin.command("ping")
    except Exception:
        pytest.skip("local Mongo not available")
    s.seed_if_empty()
    yield s
    s._client.drop_database(s._collection.database.name)
    s._client.close()


class FixedProposer:
    def __init__(self, ops, name="fixed"):
        self.ops, self.name = ops, name

    def propose(self, manifest, failures, split_summary):
        return {"ops": self.ops, "rationale": "test"}


def test_cycle_heuristic_end_to_end(store):
    r = run_cycle(store, HeuristicProposer())
    assert r["version"] == 2 and r["parent"] == 1
    doc = store.get(2)
    assert doc["scores"]["gate"]["gate"] == r["gate"]["gate"]
    assert doc["author"] == "proposer:heuristic"
    # set_threshold in the patch => needs approval => never auto-activated
    assert r["risk"] == "needs_approval" and doc["status"] in ("candidate", "rejected")
    assert store.get_active()["version"] == 1


def test_gate_fail_is_rejected(store):
    # 'stolen'-style weakening is impossible, so break the gate via a regression:
    # keyword 'door' flags clean door-ding scenarios as high risk -> accuracy drops.
    r = run_cycle(store, FixedProposer([{"op": "add_keyword", "level": "high", "keyword": "door"}]),
                  auto_activate_low_risk=True)
    assert r["gate"]["gate"] == "fail"
    assert store.get(r["version"])["status"] == "rejected"
    assert store.get_active()["version"] == 1


def test_low_risk_auto_activate_and_rollback(store):
    ops = [{"op": "add_keyword", "level": "high", "keyword": "arson"}]
    r = run_cycle(store, FixedProposer(ops), auto_activate_low_risk=True)
    assert r["gate"]["passed"] and r["risk"] == "low" and r["status"] == "active"
    rollback(store, 1)
    assert store.get_active()["version"] == 1


def test_heldout_failed_ids_never_stored_or_reported(store):
    from evals.run_eval import evaluate

    held = evaluate("baseline", "heldout")
    held_ids = {f["id"] for f in held["failures"]}
    assert held_ids, "baseline must fail some held-out scenarios for this test to bite"
    r = run_cycle(store, FixedProposer([{"op": "add_keyword", "level": "high", "keyword": "arson"}]))
    stored = store.get(r["version"])["scores"]
    blob = json.dumps(stored, default=str) + json.dumps(r, default=str)
    for hid in held_ids:
        assert f'"{hid}"' not in blob
    assert "failed_ids" not in stored["candidate"]["heldout"]
    assert "failed_ids" not in r["baseline"]["heldout"]
    assert r["baseline"]["heldout"]["n"] == held["n"]
    assert "failed_ids" in r["baseline"]["dev"]


def test_needs_approval_not_auto_activated(store):
    ops = [{"op": "set_threshold", "name": "medium_score", "value": 0.55}]
    r = run_cycle(store, FixedProposer(ops), auto_activate_low_risk=True)
    assert r["risk"] == "needs_approval"
    assert store.get_active()["version"] == 1
    if r["gate"]["passed"]:
        assert r["status"] == "candidate"
        approve(store, r["version"])
        assert store.get_active()["version"] == r["version"]
    else:
        with pytest.raises(ValueError):
            approve(store, r["version"])
