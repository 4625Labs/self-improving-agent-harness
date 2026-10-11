"""
Tool guard - enforces the manifest's tool access at two points (defence in depth).

1. Offer: the model is only shown (bound) the schemas the manifest allows for the current intent.
2. Execute: a wrapper around the ToolNode refuses any call to a tool that is not allowed, because a
   model can name a tool it was never offered.

Both points call resolve_allowed on the same conversation, so they agree. Importable without the
Agent Engine runtime (only langchain_core and tool_access are needed).

Failure policy: if resolving the allow-list raises, log at ERROR and fail OPEN to the full tool set
(the baseline behaviour) instead of crashing a customer turn. Exceptions raised while EXECUTING a
tool (including LangGraph's GraphInterrupt from interrupt()) always propagate untouched.
"""

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent_agent46 import observability
from agent_agent46.tool_access import ALL_TOOLS, resolve_tool_access

logger = logging.getLogger(__name__)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return " ".join(parts)
    return str(content)


def conversation_context(messages: Sequence[BaseMessage]) -> tuple[str, list[str]]:
    """Return (latest human message text, every tool name called in the history)."""
    latest_user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_text = _text(msg.content)
            break
    called: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls or []:
                name = call.get("name")
                if name:
                    called.append(name)
    return latest_user_text, called


def resolve_allowed(
    manifest: dict[str, Any], messages: Sequence[BaseMessage]
) -> tuple[str, frozenset[str]]:
    """Return (intent, allowed tool names). Fails open to the full tool set on any error."""
    try:
        text, called = conversation_context(messages)
        return resolve_tool_access(manifest, text, called)
    except Exception:
        logger.error("Tool access resolution failed; failing open to all tools", exc_info=True)
        observability.access_failed_open()
        return "general", ALL_TOOLS


def _schema_name(schema: Any) -> str | None:
    """Tool name of a LangChain tool, an OpenAI-style dict, or a callable/pydantic class."""
    if isinstance(schema, dict):
        fn = schema.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return fn["name"]
        name = schema.get("name")
        return name if isinstance(name, str) else None
    name = getattr(schema, "name", None) or getattr(schema, "__name__", None)
    return name if isinstance(name, str) else None


def filter_tool_schemas(schemas: Iterable[Any], allowed: Iterable[str]) -> list[Any]:
    """Keep only the schemas whose tool name is in `allowed`.

    app.get_tool_schemas() returns the original LangChain StructuredTool objects; OpenAI-style
    dicts are handled too. A schema whose name cannot be determined is dropped (least privilege).
    """
    allowed_set = frozenset(allowed)
    return [s for s in schemas if _schema_name(s) in allowed_set]


def _denial(call: dict[str, Any]) -> ToolMessage:
    name = call.get("name") or ""
    return ToolMessage(
        content=json.dumps({"error": f"Tool '{name}' is not available for this request"}),
        tool_call_id=call["id"],
        name=name,
        status="error",
    )


def guarded_tool_node(
    tool_node: Any,
    get_allowed: Callable[[list[BaseMessage]], Iterable[str]],
    *,
    intent_of: Callable[[list[BaseMessage]], Optional[str]] | None = None,
) -> Callable[..., Any]:
    """Wrap a ToolNode so calls to non-allowed tools are denied instead of executed.

    get_allowed receives the history BEFORE the last AIMessage (so the call being vetted does not
    influence its own intent classification, matching what the agent node saw when it chose it).
    With no denied calls the original state goes straight to the wrapped node, unchanged.

    Optional intent_of (same history argument) only labels the tool_denied telemetry event. It is
    called only when a call is denied, and its failures are ignored.
    """

    def node(state: dict[str, Any], config: RunnableConfig | None = None) -> Any:
        messages = list(state["messages"])
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return tool_node.invoke(state, config)

        allowed = frozenset(get_allowed(messages[:-1]))
        calls = list(last.tool_calls)
        denied_ids = {c["id"] for c in calls if c.get("name") not in allowed}
        if not denied_ids:
            return tool_node.invoke(state, config)

        intent: Optional[str] = None
        if intent_of is not None:
            try:
                intent = intent_of(messages[:-1])
            except Exception:
                logger.debug("Could not resolve intent for denied tool telemetry", exc_info=True)
        session_id = state.get("session_id") if isinstance(state, dict) else None
        for c in calls:
            if c["id"] in denied_ids:
                logger.warning(f"Denied call to tool not allowed for this request: {c.get('name')}")
                observability.tool_denied(
                    c.get("name"), intent=intent, session_id=session_id or None
                )

        kept = [c for c in calls if c["id"] not in denied_ids]
        by_id: dict[str, Any] = {}
        extra: list[Any] = []
        if kept:
            trimmed = last.model_copy(update={"tool_calls": kept})
            sub_state = {**state, "messages": [*messages[:-1], trimmed]}
            result = tool_node.invoke(sub_state, config)
            out = result.get("messages", []) if isinstance(result, dict) else result
            for item in out:
                tcid = getattr(item, "tool_call_id", None)
                if isinstance(item, ToolMessage) and tcid:
                    by_id[tcid] = item
                else:
                    extra.append(item)  # e.g. a Command returned by a tool
        ordered = [
            _denial(c) if c["id"] in denied_ids else by_id[c["id"]]
            for c in calls
            if c["id"] in denied_ids or c["id"] in by_id
        ]
        return {"messages": ordered + extra}

    return node
