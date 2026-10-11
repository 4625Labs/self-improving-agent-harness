import copy

import pytest

from agent_agent46.harness_config import DEFAULT_MANIFEST
from agent_agent46.risk import assess_claim_risk

M = DEFAULT_MANIFEST


def claim(amount=500, ctype="auto", desc="minor scratch"):
    return {"claim_amount": amount, "claim_type": ctype, "description": desc}


def pol(score=0.3, limit=50000):
    return {"risk_score": score, "coverage_limit": limit}


def run(c=None, p=None, prev=None, manifest=M):
    return assess_claim_risk(c or claim(), p or pol(), prev or [], manifest)


def test_high_by_amount():
    r = run(claim(amount=15000))
    assert r["risk_assessment"] == "high"
    assert r["confidence"] == 0.65
    assert r["recommendation"] == "manual_review_required"
    assert r["factors"] == ["High claim amount: $15000"]


def test_high_by_policy_score():
    r = run(p=pol(score=0.6))
    assert r["risk_assessment"] == "high"
    assert r["factors"] == ["High policy risk score: 0.6"]


def test_high_by_exceeding_coverage():
    r = run(claim(amount=2000), pol(limit=1000))
    assert r["risk_assessment"] == "high"
    assert r["factors"] == ["Claim amount ($2000) exceeds coverage limit ($1000)"]


@pytest.mark.parametrize("kw", DEFAULT_MANIFEST["keywords"]["high"])
def test_high_by_each_keyword(kw):
    r = run(claim(desc=f"Vehicle {kw.upper()} last night"))
    assert r["risk_assessment"] == "high"
    assert r["factors"] == ["High-risk keywords detected in claim description"]


def test_high_keyword_in_claim_type():
    assert run(claim(ctype="Theft"))["risk_assessment"] == "high"


def test_high_factor_order():
    r = run(claim(amount=20000, desc="fire"), pol(score=0.7, limit=10000))
    assert r["factors"] == [
        "Claim amount ($20000) exceeds coverage limit ($10000)",
        "High policy risk score: 0.7",
        "High claim amount: $20000",
        "High-risk keywords detected in claim description",
    ]


def test_medium_by_amount():
    r = run(claim(amount=5000))
    assert r["risk_assessment"] == "medium"
    assert r["confidence"] == 0.72
    assert r["recommendation"] == "review_recommended"
    assert r["reasoning"] == "Claim amount is elevated or patterns warrant review."
    assert r["factors"] == ["Elevated claim amount: $5000"]


def test_medium_by_score():
    r = run(p=pol(score=0.4))
    assert r["risk_assessment"] == "medium"
    assert r["factors"] == ["Elevated policy risk score: 0.4"]


def test_medium_by_previous_claims():
    assert run(prev=[{}])["risk_assessment"] == "low"
    r = run(prev=[{}, {}])
    assert r["risk_assessment"] == "medium"
    assert r["factors"] == ["Multiple previous claims: 2"]


@pytest.mark.parametrize("kw", DEFAULT_MANIFEST["keywords"]["medium"])
def test_medium_by_each_keyword(kw):
    r = run(claim(desc=f"There was {kw} to the unit"))
    assert r["risk_assessment"] == "medium"
    assert r["factors"] == []


def test_low_auto_approve():
    r = run(claim(amount=999.99))
    assert r["risk_assessment"] == "low"
    assert r["confidence"] == 0.85
    assert r["recommendation"] == "auto_approve"
    assert r["reasoning"] == "Claim amount is within normal range, no fraud indicators detected."
    assert r["factors"] == [
        "Claim amount ($999.99) is reasonable",
        "Policy risk score (0.3) is low",
        "No concerning patterns in claim description",
    ]


def test_low_at_limit_requires_review():
    r = run(claim(amount=1000))
    assert r["risk_assessment"] == "low"
    assert r["recommendation"] == "review_recommended"
    assert r["reasoning"] == "Low-risk analysis, but the claim amount requires human review."
    assert r["factors"][0] == "Claim amount ($1000) requires human review"


def test_policy_defaults():
    r = assess_claim_risk(claim(), {}, [], M)
    assert r["recommendation"] == "auto_approve"
    assert "Policy risk score (0.3) is low" in r["factors"]


def test_manifest_thresholds_change_outcome():
    m = copy.deepcopy(M)
    assert run(claim(amount=800))["recommendation"] == "auto_approve"
    m["thresholds"]["auto_approve_limit"] = 500
    assert run(claim(amount=800), manifest=m)["recommendation"] == "review_recommended"

    m = copy.deepcopy(M)
    m["thresholds"]["medium_amount"] = 2000
    assert run(claim(amount=3000), manifest=m)["risk_assessment"] == "medium"
    m["thresholds"]["high_amount"] = 2500
    assert run(claim(amount=3000), manifest=m)["risk_assessment"] == "high"

    m = copy.deepcopy(M)
    m["thresholds"]["high_score"] = 0.35
    assert run(p=pol(score=0.35), manifest=m)["risk_assessment"] == "high"
    m = copy.deepcopy(M)
    m["thresholds"]["medium_score"] = 0.2
    assert run(manifest=m)["risk_assessment"] == "medium"


def test_manifest_keywords_change_outcome():
    c = claim(desc="hail dents")
    assert run(c)["risk_assessment"] == "low"
    m = copy.deepcopy(M)
    m["keywords"]["medium"].append("hail")
    assert run(c, manifest=m)["risk_assessment"] == "medium"
    m["keywords"]["high"].append("hail")
    assert run(c, manifest=m)["risk_assessment"] == "high"
