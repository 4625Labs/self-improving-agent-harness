import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402
from test_court import FakeDb  # noqa: E402
from test_court_ui import TOKEN, FakeStore  # noqa: E402

from harness_evolve.audit import ConfirmationRequired  # noqa: E402
from harness_evolve.court import CourtDocket  # noqa: E402
from harness_evolve.court_ui.app import create_app  # noqa: E402


def _db():
    db = FakeDb()
    db["insurance_policies"].insert_one({"policy_number": "POL1", "risk_score": 0.4, "coverage_limit": 50000})
    db["insurance_claims"].insert_one({
        "claim_id": "CLM1", "policy_number": "POL1", "user_id": "u", "claim_type": "auto", "claim_amount": 900.0,
        "description": "<b>dent</b>", "recommendation": "review_recommended", "risk_assessment": "medium",
        "harness_version": 3, "resolution": "approved"})
    return db


@pytest.fixture
def court(tmp_path):
    return CourtDocket(_db(), tmp_path / "docket.json")


def _convicted(court):
    c = court.file_case(court.nominate("CLM1", "manual", "x")["case_id"])
    court.verdict(c["case_id"], "review_recommended", False, "ann", confirmed=True)
    return c["case_id"]


def _audit(court):
    return [r for r in court._db["audit_log"].docs if r["action"] in ("court.overturn", "court.retract")]


def test_overturn_in_place_with_history(court):
    cid = _convicted(court)
    out = court.overturn(cid, "auto_approve", True, "bob", "wrong call", confirmed=True)
    data = json.loads(court.docket_path.read_text())
    assert len(data) == 1 and data[0]["id"] == f"court-{cid}"
    assert data[0]["expected"] == {"recommendation": "auto_approve"} and data[0]["severity"] == "critical"
    assert out["status"] == "convicted" and out["expected_recommendation"] == "auto_approve"
    h = out["history"]
    assert len(h) == 1 and h[0]["previous_expected"] == "review_recommended" and h[0]["previous_critical"] is False
    assert h[0]["by"] == "bob" and h[0]["reason"] == "wrong call" and h[0]["ts"]
    court.overturn(cid, "manual_review_required", False, "bob", "again", confirmed=True)
    assert len(json.loads(court.docket_path.read_text())) == 1
    assert len(court._get(cid)["history"]) == 2
    assert [r["outcome"] for r in _audit(court)] == ["ok", "ok"]


def test_refused_without_confirmation_or_reason(court):
    cid = _convicted(court)
    before = court.docket_path.read_text()
    with pytest.raises(ConfirmationRequired):
        court.overturn(cid, "auto_approve", True, "bob", "why", confirmed=False)
    with pytest.raises(ValueError):
        court.overturn(cid, "auto_approve", True, "bob", "  ", confirmed=True)
    with pytest.raises(ConfirmationRequired):
        court.retract(cid, "bob", "why")
    with pytest.raises(ValueError):
        court.retract(cid, "bob", "", confirmed=True)
    assert court.docket_path.read_text() == before
    case = court._get(cid)
    assert case["status"] == "convicted" and "history" not in case
    outs = [r["outcome"] for r in _audit(court)]
    assert len(outs) == 4 and outs[0].startswith("refused") and outs[1].startswith("error")


def test_non_convicted_rejected(court):
    c = court.file_case(court.nominate("CLM1", "manual", "x")["case_id"])
    with pytest.raises(ValueError):
        court.overturn(c["case_id"], "auto_approve", True, "bob", "r", confirmed=True)
    with pytest.raises(ValueError):
        court.retract(c["case_id"], "bob", "r", confirmed=True)
    assert not court.docket_path.exists()


def test_retract_removes_scenario(court, monkeypatch):
    cid = _convicted(court)
    monkeypatch.setenv("EVOLVE_DOCKET_PATH", str(court.docket_path))
    from evals.run_eval import load_scenarios
    assert f"court-{cid}" in [s["id"] for s in load_scenarios("dev")]
    out = court.retract(cid, "bob", "bad nomination", confirmed=True)
    assert out["status"] == "dismissed" and out["history"][0]["reason"] == "bad nomination"
    assert f"court-{cid}" not in [s["id"] for s in load_scenarios("dev")]
    with pytest.raises(ValueError):
        court.overturn(cid, "auto_approve", True, "bob", "r", confirmed=True)


def _ui(tmp_path):
    db = _db()
    c = TestClient(create_app(db, FakeStore(), token=TOKEN, docket_path=tmp_path / "docket.json"),
                   follow_redirects=True)
    court = CourtDocket(db, tmp_path / "docket.json")
    cid = _convicted(court)
    return c, court, cid


def test_ui_overturn_and_retract(tmp_path):
    c, court, cid = _ui(tmp_path)
    html = c.get("/").text
    assert f"/case/{cid}/overturn" in html and f"/case/{cid}/retract" in html
    d = {"csrf": TOKEN, "expected": "auto_approve", "critical": "1", "adjuster": "<i>Bob</i>",
         "reason": "<script>x</script>"}
    assert "confirmation" in c.post(f"/case/{cid}/overturn", data=d).text
    assert court._get(cid)["expected_recommendation"] == "review_recommended"
    assert "reason is required" in c.post(f"/case/{cid}/overturn", data={**d, "reason": "", "confirm": "1"}).text
    html = c.post(f"/case/{cid}/overturn", data={**d, "confirm": "1"}).text
    assert "verdict overturned" in html
    assert "<script>x</script>" not in html and "&lt;script&gt;x&lt;/script&gt;" in html
    assert "&lt;i&gt;Bob&lt;/i&gt;" in html and "History (1)" in html
    assert json.loads((tmp_path / "docket.json").read_text())[0]["expected"]["recommendation"] == "auto_approve"
    html = c.post(f"/case/{cid}/retract", data={"csrf": TOKEN, "adjuster": "Bob", "reason": "r", "confirm": "1"}).text
    assert "verdict retracted" in html
    assert json.loads((tmp_path / "docket.json").read_text()) == []
    assert court._get(cid)["status"] == "dismissed"


def test_ui_csrf_enforced(tmp_path):
    c, court, cid = _ui(tmp_path)
    for path in ("overturn", "retract"):
        r = c.post(f"/case/{cid}/{path}", data={"csrf": "bad", "adjuster": "a", "reason": "r", "confirm": "1",
                                                 "expected": "auto_approve"})
        assert r.status_code == 403
    assert court._get(cid)["status"] == "convicted" and "history" not in court._get(cid)
