"""Pure claim risk assessment: no I/O, driven entirely by the harness manifest."""

from typing import Any


def assess_claim_risk(
    claim: dict[str, Any],
    policy: dict[str, Any],
    previous_claims: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Classify a claim as high/medium/low risk using manifest thresholds and keywords.

    previous_claims must already exclude the claim being assessed.
    """
    t = manifest["thresholds"]
    high_keywords = manifest["keywords"]["high"]
    medium_keywords = manifest["keywords"]["medium"]

    claim_amount = claim.get("claim_amount", 0)
    claim_type = claim.get("claim_type", "")
    description = claim.get("description", "")
    policy_risk_score = policy.get("risk_score", 0.3)
    coverage_limit = policy.get("coverage_limit", 50000)

    text = f"{description} {claim_type}".lower()
    high_kw_hit = any(kw in text for kw in high_keywords)
    medium_kw_hit = any(kw in text for kw in medium_keywords)

    is_high_risk = (
        claim_amount >= t["high_amount"]
        or policy_risk_score >= t["high_score"]
        or claim_amount > coverage_limit
        or high_kw_hit
    )
    is_medium_risk = (
        claim_amount >= t["medium_amount"]
        or policy_risk_score >= t["medium_score"]
        or len(previous_claims) >= 2
        or medium_kw_hit
    )

    if is_high_risk:
        result: dict[str, Any] = {
            "risk_assessment": "high",
            "confidence": 0.65,
            "recommendation": "manual_review_required",
            "reasoning": "Multiple risk indicators detected. Human review strongly recommended.",
            "factors": [],
        }
        if claim_amount > coverage_limit:
            result["factors"].append(
                f"Claim amount (${claim_amount}) exceeds coverage limit (${coverage_limit})"
            )
        if policy_risk_score >= t["high_score"]:
            result["factors"].append(f"High policy risk score: {policy_risk_score}")
        if claim_amount >= t["high_amount"]:
            result["factors"].append(f"High claim amount: ${claim_amount}")
        if high_kw_hit:
            result["factors"].append("High-risk keywords detected in claim description")
    elif is_medium_risk:
        result = {
            "risk_assessment": "medium",
            "confidence": 0.72,
            "recommendation": "review_recommended",
            "reasoning": "Claim amount is elevated or patterns warrant review.",
            "factors": [],
        }
        if claim_amount >= t["medium_amount"]:
            result["factors"].append(f"Elevated claim amount: ${claim_amount}")
        if len(previous_claims) >= 2:
            result["factors"].append(f"Multiple previous claims: {len(previous_claims)}")
        if policy_risk_score >= t["medium_score"]:
            result["factors"].append(f"Elevated policy risk score: {policy_risk_score}")
    else:
        auto_approve = claim_amount < t["auto_approve_limit"]
        result = {
            "risk_assessment": "low",
            "confidence": 0.85,
            "recommendation": "auto_approve" if auto_approve else "review_recommended",
            "reasoning": (
                "Claim amount is within normal range, no fraud indicators detected."
                if auto_approve
                else "Low-risk analysis, but the claim amount requires human review."
            ),
            "factors": [
                (
                    f"Claim amount (${claim_amount}) is reasonable"
                    if auto_approve
                    else f"Claim amount (${claim_amount}) requires human review"
                ),
                f"Policy risk score ({policy_risk_score}) is low",
                "No concerning patterns in claim description",
            ],
        }
    return result
