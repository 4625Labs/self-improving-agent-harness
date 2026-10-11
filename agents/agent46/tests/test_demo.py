import copy
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_agent46.harness_config import DEFAULT_MANIFEST, HarnessConfigStore  # noqa: E402
from agent_agent46.risk import assess_claim_risk  # noqa: E402
from evals.run_eval import AUTO, RANK, REVIEW, evaluate_all  # noqa: E402
from harness_evolve import demo  # noqa: E402
from harness_evolve.audit import ConfirmationRequired  # noqa: E402
from harness_evolve.court import CourtDocket  # noqa: E402
from harness_evolve.patch import apply_patch  # noqa: E402


def _match(d, q):
    for k, v in q.items():
        if isinstance(v, dict) and "$in" in v:
            if d.get(k) not in v["$in"]:
                return False
        elif isinstance(v, dict) and "$ne" in v:
            if d.get(k) == v["$ne"]:
                return False
        elif d.get(k) != v:
            return False
    return True


class _Rows(list):
    def sort(self, key, direction=1):
        super().sort(key=lambda d: d[key], reverse=direction < 0)
        return self


class FakeColl:
    def __init__(self):
        self.docs = []

    def find(self, q=None, proj=None):
        return _Rows(dict(d) for d in self.docs if _match(d, q or {}))

    def find_one(self, q, proj=None, sort=None):
        r = self.find(q)
        if sort:
            key, direction = sort[0]
            r.sort(key, direction)
        return r[0] if r else None

    def count_documents(self, q):
        return len(self.find(q))

    def insert_one(self, d):
        self.docs.append(dict(d))

    def update_one(self, q, u):
        for d in self.docs:
            if _match(d, q):
                d.update(u["$set"])
                return

    def update_many(self, q, u):
        for d in self.docs:
            if _match(d, q):
                d.update(u["$set"])

    def delete_many(self, q):
        self.docs = [d for d in self.docs if not _match(d, q)]


class FakeDb(dict):
    def __missing__(self, k):
        self[k] = FakeColl()
        return self[k]


def make_env():
    db = FakeDb()
    store = object.__new__(HarnessConfigStore)
    store._collection = db["harness_manifests"]
    store.seed_if_empty()
    # real (non-demo) data that must survive a reset
    db["insurance_policies"].insert_one({"policy_number": "POL-REAL", "user_id": "u", "risk_score": 0.3})
    db["insurance_claims"].insert_one({"claim_id": "CLM-REAL", "policy_number": "POL-REAL", "claim_type": "collision",
                                       "claim_amount": 500, "description": "dent", "recommendation": AUTO,
                                       "resolution": "approved"})
    CourtDocket(db).nominate("CLM-REAL", "manual", "real case")
    return db, store


@pytest.fixture
def env():
    return make_env()


def quiet(*a, **k):
    pass


def run(db, store, scenario="parts-theft"):
    return demo.run(db, store, scenario, pace=0, out=quiet, sleep=quiet, actor="tester")


def case_statuses(db):
    return sorted(c["status"] for c in db["court_cases"].docs if c["claim_id"] in demo.CLAIM_IDS)


def baseline_rec(c):
    return assess_claim_risk({"claim_amount": c["amount"], "claim_type": c["type"], "description": c["desc"]},
                             {"risk_score": 0.3, "coverage_limit": 50000}, [],
                             copy.deepcopy(DEFAULT_MANIFEST))["recommendation"]


def test_seed_idempotent_three_nominated_two_filed(env):
    db, store = env
    for _ in range(2):
        demo.seed(db, store, out=quiet)
    assert case_statuses(db) == ["filed"] * 2 + ["nominated"] * 3
    sources = {c["source"] for c in db["court_cases"].docs
               if c["status"] == "nominated" and c["claim_id"] in demo.CLAIM_IDS}
    assert sources == {"manual", "raindrop_signal", "reviewer_disagreement"}
    assert len(db["insurance_claims"].find({"demo": True})) == len(demo.CLAIMS)
    assert len(db["insurance_policies"].find({"policy_number": demo.POLICY_NUMBER})) == 1
    filed = [c for c in db["court_cases"].docs if c["status"] == "filed"][0]
    assert filed["agent"]["harness_version"] == 1 and filed["resolution"] == "approved" and filed["draft"]
    assert any(r["action"] == "demo.seed" for r in db["audit_log"].docs)


def test_descriptions_read_like_incidents():
    for c in demo.CLAIMS:
        assert not any(w in c["desc"].lower() for w in ("test", "demo", "simulat"))


def test_baseline_under_escalates_scenario_claims():
    for sc in demo.SCENARIOS.values():
        for cid, expected, _ in sc["claims"]:
            got = baseline_rec(demo._BY_ID[cid])
            if expected == AUTO:
                assert got == AUTO  # control: correct today, must stay correct
            else:
                assert RANK[got] < RANK[expected], (cid, got)
    for c in demo.CLAIMS:  # every seeded claim was auto-approved or lightly reviewed
        assert baseline_rec(c) in (AUTO, REVIEW)


def test_seeded_recommendation_matches_real_risk_function(env):
    db, store = env
    demo.seed(db, store, out=quiet)
    for c in demo.CLAIMS:
        assert db["insurance_claims"].find_one({"claim_id": c["id"]})["recommendation"] == baseline_rec(c)


@pytest.mark.parametrize("name", ["parts-theft", "staged-accident", "cloned-vehicle"])
def test_run_leaves_one_candidate_and_never_activates(env, name):
    db, store = env
    demo.seed(db, store, out=quiet)
    res = run(db, store, name)
    versions = store.list_versions()
    assert [v["status"] for v in versions] == ["active", "candidate"]
    assert store.get_active()["version"] == 1
    assert res["gate"] == "pass" and res["status"] == "candidate"
    assert versions[1]["demo"] is True
    kws = {o["keyword"] for o in res["ops"] if o["op"] == "add_keyword"}
    expected = {"parts-theft": {"stripped", "cannibalized"}, "staged-accident": {"staged"},
                "cloned-vehicle": {"cloned"}}[name]
    assert expected <= kws
    actions = [r["action"] for r in db["audit_log"].docs]
    assert "manifest.approve" not in actions and "manifest.rollback" not in actions


def test_scenarios_propose_different_patches(tmp_path, monkeypatch):
    patches = {}
    for name in ("parts-theft", "staged-accident", "cloned-vehicle"):
        monkeypatch.setenv("EVOLVE_DOCKET_PATH", str(tmp_path / f"{name}.json"))
        db, store = make_env()
        demo.seed(db, store, out=quiet)
        patches[name] = frozenset(o["keyword"] for o in run(db, store, name)["ops"] if o["op"] == "add_keyword")
    assert len(set(patches.values())) == 3


def test_run_refuses_when_candidate_exists(env):
    db, store = env
    demo.seed(db, store, out=quiet)
    run(db, store, "parts-theft")
    n_cases = len(db["court_cases"].docs)
    with pytest.raises(RuntimeError, match="already awaiting approval"):
        run(db, store, "staged-accident")
    assert len(store.list_versions()) == 2 and len(db["court_cases"].docs) == n_cases


def test_run_needs_seed(env):
    db, store = env
    with pytest.raises(RuntimeError, match="demo seed"):
        run(db, store)


def test_verdict_is_labelled_simulated(env):
    db, store = env
    demo.seed(db, store, out=quiet)
    run(db, store, "parts-theft")
    rows = [r for r in db["audit_log"].docs if r["action"] == "court.verdict"]
    assert rows and all("simulated" in r["actor"].lower() for r in rows)
    sim = [r for r in db["audit_log"].docs if r["action"] == "demo.simulated_verdict"]
    assert sim and all(r["detail"]["simulated"] is True for r in sim)


def test_run_prints_next_step_and_url(env):
    db, store = env
    demo.seed(db, store, out=quiet)
    lines = []
    demo.run(db, store, "cloned-vehicle", pace=0, ui_url="http://x:1", out=lines.append, sleep=quiet)
    text = "\n".join(lines)
    assert "Open the Regression Court, find the proposed version under Policy versions, and approve it" in text
    assert "http://x:1" in text and "SIMULATED" in text


def test_trap_scenario_fails_gate(env):
    db, store = env
    demo.seed(db, store, out=quiet)
    res = run(db, store, "abandoned-lookalike")
    assert res["gate"] == "fail" and res["status"] == "rejected"
    assert [v["status"] for v in store.list_versions()] == ["active", "rejected"]
    assert store.get_active()["version"] == 1
    # the real evaluator names the cause: 'abandoned' catches the legitimate control claim on dev
    cand = apply_patch(DEFAULT_MANIFEST, res["ops"])
    assert "abandoned" in cand["keywords"]["medium"]
    assert (evaluate_all(cand)["dev"]["over_escalation_rate"]
            > evaluate_all(DEFAULT_MANIFEST)["dev"]["over_escalation_rate"])


def test_reset_removes_only_demo_data(env):
    db, store = env
    demo.seed(db, store, out=quiet)
    run(db, store, "parts-theft")
    docket = Path(os.environ["EVOLVE_DOCKET_PATH"])
    assert "court-" in docket.read_text()
    with pytest.raises(ConfirmationRequired):
        demo.reset(db, store, "tester", out=quiet)
    assert len(db["insurance_claims"].docs) == len(demo.CLAIMS) + 1
    demo.reset(db, store, "tester", confirmed=True, out=quiet)
    assert [c["claim_id"] for c in db["insurance_claims"].docs] == ["CLM-REAL"]
    assert [p["policy_number"] for p in db["insurance_policies"].docs] == ["POL-REAL"]
    assert [c["claim_id"] for c in db["court_cases"].docs] == ["CLM-REAL"]
    assert "court-" not in docket.read_text()
    assert [v["status"] for v in store.list_versions()] == ["active", "rejected"]
    assert store.get_active()["version"] == 1
    actions = [r["action"] for r in db["audit_log"].docs]
    assert "demo.reset" in actions and "court.retract" in actions and "manifest.reject" in actions


def test_cli_routes_demo(monkeypatch):
    from harness_evolve import cli

    seen = {}
    monkeypatch.setattr(demo, "demo_main", lambda a, rest: seen.update(rest=rest, yes=a.yes) or 0)
    assert cli.main(["--yes", "demo", "run", "--pace", "0", "--scenario", "cloned-vehicle"]) == 0
    assert seen == {"rest": ["run", "--pace", "0", "--scenario", "cloned-vehicle"], "yes": True}
