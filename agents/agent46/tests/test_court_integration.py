"""Court -> docket.json -> evaluator: the two halves were written separately, so test the seam."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_agent46.harness_config import DEFAULT_MANIFEST
from evals.run_eval import evaluate, load_scenarios
from harness_evolve.court import CourtDocket
from test_court import FakeDb


def _docket(tmp_path):
    db = FakeDb()
    db["insurance_policies"].insert_one({"policy_number": "POL9", "risk_score": 0.3, "coverage_limit": 20000})
    db["insurance_claims"].insert_one({
        "claim_id": "CLM9", "policy_number": "POL9", "user_id": "u", "claim_type": "auto",
        "claim_amount": 700.0, "description": "Vandalism: car found stripped overnight, no witnesses",
        "recommendation": "auto_approve", "risk_assessment": "low", "harness_version": 3, "resolution": "approved"})
    return CourtDocket(db, tmp_path / "docket.json")


def test_convicted_case_reaches_dev_only_and_is_scored(tmp_path, monkeypatch):
    court = _docket(tmp_path)
    monkeypatch.setenv("EVOLVE_DOCKET_PATH", str(court.docket_path))
    before = len(load_scenarios("dev"))
    case = court.file_case(court.nominate("CLM9", "manual", "missed fraud")["case_id"])
    # filed but not convicted: nothing changes
    assert not court.docket_path.exists() or len(load_scenarios("dev")) == before
    court.verdict(case["case_id"], "manual_review_required", True, "adjuster")
    dev = load_scenarios("dev")
    assert len(dev) == before + 1
    assert any(s["id"] == f"court-{case['case_id']}" for s in dev)
    assert all(not s["id"].startswith("court-") for s in load_scenarios("heldout"))
    res = evaluate(DEFAULT_MANIFEST, "dev")
    assert res["n"] == before + 1
