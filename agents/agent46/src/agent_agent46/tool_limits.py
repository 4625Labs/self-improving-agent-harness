"""
Tool-call limits (HANDOFF principle 5: humans own the limits).

Two of them are evolvable inside human-owned bounds: MAX_LLM_STEPS_PER_TURN and
MAX_TOOL_CALLS_PER_TURN (manifest section "limits"; the constants below are the baseline and the
fail-safe). Everything else (identical calls, per-thread cap, argument sanity) lives in code on
purpose so the evolution loop cannot loosen it. `resolve_limits` is the ONE place the evolvable
values are read; the agent node, the guard and the evaluator all use it. Everything is derived from
the message history; no graph state is added. Pure and importable without the Agent Engine runtime.
"""

import logging
import math
from collections.abc import Sequence
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

MAX_TOOL_CALLS_PER_TURN = 12
MAX_IDENTICAL_CALLS_PER_TURN = 3
MAX_TOOL_CALLS_PER_THREAD = 60
MAX_STRING_ARG_CHARS = 5000
MAX_CLAIM_AMOUNT = 10_000_000
MAX_LLM_STEPS_PER_TURN = 10

logger = logging.getLogger(__name__)

LIMIT_NAMES = ("max_llm_steps_per_turn", "max_tool_calls_per_turn")


def baseline_limits() -> dict[str, int]:
    """The limits the agent shipped with."""
    return {
        "max_llm_steps_per_turn": MAX_LLM_STEPS_PER_TURN,
        "max_tool_calls_per_turn": MAX_TOOL_CALLS_PER_TURN,
    }


def validate_limits(limits: Any, bounds: dict[str, tuple[int, int]]) -> list[str]:
    """Return invariant violations for a manifest's limits value (empty list = valid)."""
    if not isinstance(limits, dict) or set(limits) != set(LIMIT_NAMES):
        return [f"limits must be an object with exactly the keys {list(LIMIT_NAMES)}"]
    errors: list[str] = []
    for name in LIMIT_NAMES:
        v = limits[name]
        lo, hi = bounds[name]
        if isinstance(v, bool) or not isinstance(v, int):
            errors.append(f"limits.{name} must be an integer")
        elif not lo <= v <= hi:
            errors.append(f"limits.{name}={v} outside allowed range [{lo}, {hi}]")
    return errors


def resolve_limits(manifest: Any) -> tuple[int, int]:
    """(max LLM steps per turn, max tool calls per turn) for a manifest. NEVER raises.

    A manifest with no "limits" (older manifests) means baseline. Any error, or a value outside the
    human-owned bounds, logs at ERROR and falls back to the baseline constants.
    """
    base = (MAX_LLM_STEPS_PER_TURN, MAX_TOOL_CALLS_PER_TURN)
    try:
        limits = manifest.get("limits") if isinstance(manifest, dict) else None
        if limits is None:
            return base
        from agent_agent46.harness_config import INVARIANTS

        errors = validate_limits(limits, INVARIANTS["limits_bounds"])
        if errors:
            raise ValueError("; ".join(errors))
        return limits["max_llm_steps_per_turn"], limits["max_tool_calls_per_turn"]
    except Exception:
        logger.error("Limit resolution failed; using baseline limits", exc_info=True)
        return base

# The per-thread cap never resets, so it must not strand a claim that is already in flight
# (HANDOFF principle 15). These finish a claim; the per-turn and identical-call rules still apply.
THREAD_CAP_EXEMPT = frozenset({"human_review", "resolve_claim", "send_notification", "check_claim_status"})

# Rule names (used in telemetry).
RULE_TURN = "calls_per_turn"
RULE_IDENTICAL = "identical_calls"
RULE_THREAD = "calls_per_thread"
RULE_ARGS = "argument_sanity"


def _turn_messages(history: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Messages since (excluding) the last HumanMessage."""
    for i in range(len(history) - 1, -1, -1):
        if isinstance(history[i], HumanMessage):
            return list(history[i + 1 :])
    return list(history)


def llm_steps_this_turn(history: Sequence[BaseMessage]) -> int:
    """Number of AIMessages since the last HumanMessage."""
    return sum(1 for m in _turn_messages(history) if isinstance(m, AIMessage))


def trace_from_messages(messages: Sequence[BaseMessage]) -> dict[str, int]:
    """Summarise the current turn: LLM steps (AIMessages since the last HumanMessage), tool calls
    and the highest count of any one identical call. The shape `evals/limits_eval` replays."""
    turn = _turn_messages(messages)
    calls = _calls(turn)
    counts: dict[tuple[str, str], int] = {}
    for c in calls:
        k = _key(c)
        counts[k] = counts.get(k, 0) + 1
    return {
        "llm_steps": llm_steps_this_turn(messages),
        "tool_calls": len(calls),
        "max_identical_calls": max(counts.values(), default=0),
    }


def _calls(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, AIMessage):
            out.extend(m.tool_calls or [])
    return out


def _key(call: dict[str, Any]) -> tuple[str, str]:
    args = call.get("args") or {}
    return str(call.get("name") or ""), repr(sorted(args.items(), key=lambda kv: str(kv[0])))


def _bad_arg(name: str, value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return f"argument '{name}' must be a finite number"
        if value < 0:
            return f"argument '{name}' must not be negative"
        if name == "claim_amount" and value > MAX_CLAIM_AMOUNT:
            return f"argument '{name}' exceeds the maximum allowed amount"
    elif isinstance(value, str):
        if len(value) > MAX_STRING_ARG_CHARS:
            return f"argument '{name}' is too long"
    elif isinstance(value, dict):
        for k, v in value.items():
            r = _bad_arg(str(k), v)
            if r:
                return r
    elif isinstance(value, (list, tuple)):
        for v in value:
            r = _bad_arg(name, v)
            if r:
                return r
    return None


def check_call(
    history: Sequence[BaseMessage], call: dict[str, Any], limits: Optional[tuple[int, int]] = None
) -> Optional[str]:
    """Return "<rule>: <message>" if `call` must be refused, else None.

    `history` must include every tool call already accepted (including earlier calls in the same
    AIMessage as `call`), but not `call` itself. `limits` is resolve_limits(manifest); None means
    the baseline.
    """
    max_calls = limits[1] if limits is not None else MAX_TOOL_CALLS_PER_TURN
    prior_all = _calls(history)
    prior_turn = _calls(_turn_messages(history))
    if len(prior_turn) >= max_calls:
        return f"{RULE_TURN}: too many tool calls in this request"
    k = _key(call)
    if sum(1 for c in prior_turn if _key(c) == k) >= MAX_IDENTICAL_CALLS_PER_TURN:
        return f"{RULE_IDENTICAL}: this exact call was already repeated too many times"
    if len(prior_all) >= MAX_TOOL_CALLS_PER_THREAD and call.get("name") not in THREAD_CAP_EXEMPT:
        return f"{RULE_THREAD}: too many tool calls in this conversation"
    for name, value in (call.get("args") or {}).items():
        r = _bad_arg(str(name), value)
        if r:
            return f"{RULE_ARGS}: {r}"
    return None
