import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness_evolve.court import CourtDocket


class FakeColl:
    def __init__(self):
        self.docs = []

    @staticmethod
    def _match(d, q):
        for k, v in q.items():
            if isinstance(v, dict) and "$in" in v:
                if d.get(k) not in v["$in"]:
                    return False
            elif d.get(k) != v:
                return False
        return True

    def find(self, q=None, proj=None):
        return [dict(d) for d in self.docs if self._match(d, q or {})]

    def find_one(self, q, proj=None):
        r = self.find(q)
        return r[0] if r else None

    def insert_one(self, d):
        self.docs.append(dict(d))

    def update_one(self, q, u):
        for d in self.docs:
            if self._match(d, q):
                d.update(u["$set"])
                return


class FakeDb(dict):
    def __missing__(self, k):
        self[k] = FakeColl()
        return self[k]


@pytest.fixture
def court(tmp_path):
    db = FakeDb()
    db["insurance_policies"].insert_one({"policy_number": "POL1", "risk_score": 0.4, "coverage_limit": 50000})
    for cid, rec, res in (("CLM1", "review_recommended", "approved"), ("CLM2", "auto_approve", "approved"),
                          ("CLM3", "manual_review_required", "denied")):
        db["insurance_claims"].insert_one({
            "claim_id": cid, "policy_number": "POL1", "user_id": "u", "claim_type": "auto",
            "claim_amount": 900.0, "description": "dent", "recommendation": rec,
            "risk_assessment": "medium", "harness_version": 3, "resolution": res})
    return CourtDocket(db, tmp_path / "docket.json")


def _shape_ok(sc):
    assert {"id", "claim", "policy", "previous_claims_count", "expected", "tags", "severity"} <= set(sc)
    assert {"claim_amount", "claim_type", "description"} <= set(sc["claim"])
    assert {"risk_score", "coverage_limit"} <= set(sc["policy"])
    assert sc["expected"]["recommendation"] in {"auto_approve", "review_recommended", "manual_review_required"}
    assert "source:court" in sc["tags"] and sc["id"].startswith("court-")


def test_nominate_idempotent(court):
    a = court.nominate("clm1", "manual", "x")
    assert court.nominate("CLM1", "manual", "y") == a
    assert len(court.list_cases()) == 1


def test_unconvicted_never_in_docket(court):
    c = court.nominate("CLM1", "manual", "x")
    court.file_case(c["case_id"])
    d = court.nominate("CLM3", "manual", "x")
    court.dismiss(d["case_id"], "me", "nope", confirmed=True)
    assert not court.docket_path.exists() or json.loads(court.docket_path.read_text()) == []


def test_verdict_rejected_unless_filed(court):
    c = court.nominate("CLM1", "manual", "x")
    with pytest.raises(ValueError):
        court.verdict(c["case_id"], "auto_approve", False, "me", confirmed=True)
    assert not court.docket_path.exists()


def test_verdict_invalid_expected(court):
    c = court.file_case(court.nominate("CLM1", "manual", "x")["case_id"])
    with pytest.raises(ValueError):
        court.verdict(c["case_id"], "bogus", False, "me", confirmed=True)


def test_file_case_draft(court):
    c = court.file_case(court.nominate("CLM1", "manual", "x")["case_id"])
    assert c["status"] == "filed" and "expected" not in c["draft"]
    assert c["draft"]["previous_claims_count"] == 2
    assert c["agent"] == {"recommendation": "review_recommended", "risk_assessment": "medium", "harness_version": 3}
    assert c["resolution"] == "approved"


def test_verdict_writes_valid_docket_no_duplicates(court):
    c = court.file_case(court.nominate("CLM1", "manual", "x")["case_id"])
    out = court.verdict(c["case_id"], "auto_approve", True, "me", confirmed=True)
    assert out["status"] == "convicted"
    with pytest.raises(ValueError):
        court.verdict(c["case_id"], "auto_approve", True, "me", confirmed=True)
    data = json.loads(court.docket_path.read_text())
    assert len(data) == 1 and data[0]["severity"] == "critical"
    for sc in data:
        _shape_ok(sc)
    dev = json.loads((Path(__file__).resolve().parent.parent / "evals/scenarios/dev.json").read_text())
    assert set(data[0]) == set(dev[0])


def test_scan_overescalation_only(court):
    new = court.nominate_reviewer_disagreements()
    assert [c["claim_id"] for c in new] == ["CLM1"]
    assert new[0]["source"] == "reviewer_disagreement"
    assert court.nominate_reviewer_disagreements() == []
    assert len(court.list_cases("nominated")) == 1
