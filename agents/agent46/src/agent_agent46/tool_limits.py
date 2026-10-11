"""
Code-owned tool-call limits (HANDOFF principle 5: humans own the limits).

These constants live in code on purpose: they are NOT part of the evolvable harness manifest, so
the evolution loop cannot loosen them. Everything is derived from the message history; no graph
state is added. Pure and importable without the Agent Engine runtime.
"""

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


def check_call(history: Sequence[BaseMessage], call: dict[str, Any]) -> Optional[str]:
    """Return "<rule>: <message>" if `call` must be refused, else None.

    `history` must include every tool call already accepted (including earlier calls in the same
    AIMessage as `call`), but not `call` itself.
    """
    prior_all = _calls(history)
    prior_turn = _calls(_turn_messages(history))
    if len(prior_turn) >= MAX_TOOL_CALLS_PER_TURN:
        return f"{RULE_TURN}: too many tool calls in this request"
    k = _key(call)
    if sum(1 for c in prior_turn if _key(c) == k) >= MAX_IDENTICAL_CALLS_PER_TURN:
        return f"{RULE_IDENTICAL}: this exact call was already repeated too many times"
    if len(prior_all) >= MAX_TOOL_CALLS_PER_THREAD:
        return f"{RULE_THREAD}: too many tool calls in this conversation"
    for name, value in (call.get("args") or {}).items():
        r = _bad_arg(str(name), value)
        if r:
            return f"{RULE_ARGS}: {r}"
    return None
