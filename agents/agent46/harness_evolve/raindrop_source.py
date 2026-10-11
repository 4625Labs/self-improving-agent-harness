"""Turn exported Raindrop events into court nomination candidates. Pure and advisory (HANDOFF principle 17).

Nothing here reads a network, writes a scenario or convicts: a candidate is only a suggestion that a human
reviews in the court. Step-cap and tool-limit events are harness pressure, not claim failures, so they are
summarised for the limits surface instead of being nominated. review_resumed disagreements belong to the
reviewer-disagreement scan and are excluded here to avoid duplicate cases.
"""

from typing import Any, Iterable

SOURCE = "raindrop_signal"
DEFAULT_SIGNALS = ("claim_approved_despite_fraud_cues", "user_frustration", "task_failure")  # RULE_SIGNAL_ALLOWLIST
LIMIT_EVENTS = ("step_cap_hit", "tool_limit_hit")  # RULE_LIMIT_EVENTS: summarised, never nominated
REVIEW_EVENT = "review_resumed"  # RULE_REVIEW_EXCLUDED: owned by the reviewer-disagreement scan
AUTO_APPROVE = "auto_approve"  # RULE_AUTO_APPROVE_CONTEXT: noted in the reason when a risk_assessed says so


def _get(ev: dict, key: str) -> Any:
    p = ev.get("properties")
    v = p.get(key) if isinstance(p, dict) else None
    return ev.get(key) if v is None else v


def _name(ev: dict) -> str:
    n = ev.get("event") or ev.get("name") or ev.get("event_name")
    return n if isinstance(n, str) else ""


def _claim_id(ev: dict) -> str:
    c = _get(ev, "claim_id")
    return c.strip().upper() if isinstance(c, str) and c.strip() else ""


def analyse_events(events: Iterable[Any], *, min_count: int = 1, signals: Iterable[str] = DEFAULT_SIGNALS) -> dict:
    """Return {"candidates", "limit_pressure": {"rules", "harness_version"}, "skipped"}. Never raises."""
    allow = set(signals)
    skipped = 0
    found: dict[str, dict] = {}
    auto_approved: set[str] = set()
    limits: dict[str, int] = {}
    versions: set = set()
    try:
        seq = list(events)
    except TypeError:
        seq, skipped = [], 1
    for ev in seq:
        try:
            if not isinstance(ev, dict):
                skipped += 1
                continue
            name = _name(ev)
            if name in LIMIT_EVENTS:
                rule = _get(ev, "rule") if name == "tool_limit_hit" else "step_cap"
                rule = str(rule) if rule else name
                limits[rule] = limits.get(rule, 0) + 1
                v = _get(ev, "harness_version")
                if v is not None:
                    versions.add(v)
                continue
            cid = _claim_id(ev)
            if not cid:
                continue
            if name == "risk_assessed" and _get(ev, "recommendation") == AUTO_APPROVE:
                auto_approved.add(cid)
            if name == REVIEW_EVENT:
                continue
            sig = next((s for s in (name, _get(ev, "signal"), _get(ev, "signal_name")) if isinstance(s, str) and s in allow), None)
            if sig is None:
                continue
            slot = found.setdefault(cid, {"signals": [], "count": 0})
            slot["count"] += 1
            if sig not in slot["signals"]:
                slot["signals"].append(sig)
        except Exception:  # noqa: BLE001 - malformed input is skipped and counted, never raised
            skipped += 1
    out = []
    for cid, s in found.items():
        if s["count"] < min_count:
            continue
        reason = "Raindrop signal: " + ", ".join(s["signals"])
        if cid in auto_approved:
            reason += " (risk_assessed recommended auto_approve)"
        out.append({"claim_id": cid, "reason": reason, "evidence_count": s["count"]})
    hv = sorted(versions, key=str)
    return {"candidates": out, "skipped": skipped,
            "limit_pressure": {"rules": limits, "harness_version": hv[0] if len(hv) == 1 else (hv or None)}}


def candidates_from_events(events: Iterable[Any], *, min_count: int = 1, signals: Iterable[str] = DEFAULT_SIGNALS) -> list[dict]:
    """Nomination candidates {claim_id, reason, evidence_count}, deduped by claim_id."""
    return analyse_events(events, min_count=min_count, signals=signals)["candidates"]
