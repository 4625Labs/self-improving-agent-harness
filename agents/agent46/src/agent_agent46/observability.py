"""Raindrop observability: ships agent events so Raindrop can detect failures in production.

Every function is a no-op unless RAINDROP_WRITE_KEY is set, and no SDK error ever reaches the
agent. Each event carries the `harness_version` that decided it, so issues can be compared
across manifest versions.

Conversation text leaves the process only when RAINDROP_SEND_CONTENT=true. By default events
carry metadata and a placeholder, because customer messages can contain personal data.
"""

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_state: dict[str, Any] = {"client": None, "tried": False}


def _client() -> Any:
    """Lazily import and initialise the SDK. Returns None when disabled or unavailable."""
    if _state["tried"]:
        return _state["client"]
    _state["tried"] = True
    key = os.environ.get("RAINDROP_WRITE_KEY", "").strip()
    if not key:
        return None
    try:
        import raindrop.analytics as raindrop

        raindrop.init(key)
        _state["client"] = raindrop
    except Exception:
        logger.warning("Raindrop disabled: SDK import or init failed", exc_info=True)
    return _state["client"]


def _send_content() -> bool:
    return os.environ.get("RAINDROP_SEND_CONTENT", "").strip().lower() == "true"


def _text(value: Optional[str]) -> str:
    if not _send_content():
        return "[content withheld]" if value else ""
    return value or ""


def _str(value: Any) -> str:
    """str() that can never raise, so a bad value cannot break a customer turn."""
    try:
        return str(value)
    except Exception:
        return "<unprintable>"


def track_event(
    name: str,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    input_text: str = "",
    output_text: str = "",
    **properties: Any,
) -> None:
    """Send one event. Never raises."""
    try:
        client = _client()
        if client is None:
            return
        client.track_ai(
            user_id=user_id or "unknown",
            event=name,
            model="agent46",
            # Raindrop rejects events whose input and output are both empty.
            input=_text(input_text) or name,
            output=_text(output_text) or name,
            convo_id=session_id,
            properties={k: v for k, v in properties.items() if v is not None},
        )
    except Exception:
        logger.debug("Raindrop track_event failed", exc_info=True)


def track_turn(
    *,
    user_id: Optional[str],
    session_id: Optional[str],
    input_text: str,
    output_text: str,
    harness_version: Any,
    intent: Optional[str] = None,
) -> None:
    track_event(
        "agent_turn",
        user_id=user_id,
        session_id=session_id,
        input_text=input_text,
        output_text=output_text,
        harness_version=harness_version,
        intent=intent,
    )


def tool_denied(
    tool: Optional[str], *, intent: Optional[str] = None, session_id: Optional[str] = None
) -> None:
    track_event("tool_denied", tool=tool, intent=intent, session_id=session_id)


def access_failed_open() -> None:
    track_event("tool_access_failed_open")


def manifest_fallback(reason: str) -> None:
    track_event("manifest_fallback", reason=reason, harness_version=0)


def risk_assessed(
    *,
    claim_id: str,
    risk: Any,
    recommendation: Any,
    harness_version: Any,
    user_id: Optional[str],
    claim_amount: Optional[float] = None,
    claim_type: Optional[str] = None,
    policy_number: Optional[str] = None,
    description_len: Optional[int] = None,
    previous_claims_count: Optional[int] = None,
) -> None:
    """Record a risk assessment. Only the description's length is sent, never its text."""
    track_event(
        "risk_assessed",
        user_id=user_id,
        claim_id=claim_id,
        risk=_str(risk),
        recommendation=_str(recommendation),
        harness_version=harness_version,
        claim_amount=claim_amount,
        claim_type=claim_type,
        policy_number=policy_number,
        description_len=description_len,
        previous_claims_count=previous_claims_count,
    )


def review_suspended(*, claim_id: str, risk_level: Any, user_id: Optional[str]) -> None:
    track_event("review_suspended", user_id=user_id, claim_id=claim_id, risk_level=_str(risk_level))


def review_resumed(
    *,
    claim_id: str,
    decision: str,
    user_id: Optional[str],
    recommendation: Any = None,
    harness_version: Any = None,
) -> None:
    """Record a reviewer's decision, with the claim's stored recommendation when known."""
    track_event(
        "review_resumed",
        user_id=user_id,
        claim_id=claim_id,
        decision=decision,
        recommendation=_str(recommendation) if recommendation is not None else None,
        harness_version=harness_version,
    )
