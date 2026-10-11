import copy
import json
import os

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from agent_agent46.harness_config import DEFAULT_MANIFEST
from agent_agent46.tool_access import ALL_TOOLS
from agent_agent46.tool_guard import (
    conversation_context,
    filter_tool_schemas,
    guarded_tool_node,
    resolve_allowed,
)

NARROWED = {
    "resolve_claim", "create_policy", "file_claim", "human_review", "send_notification",
}


def info_narrowed_manifest():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    ta = {"intents": {i: {"allow": sorted(ALL_TOOLS)} for i in ("quote", "claims", "info", "general")}}
    ta["intents"]["info"]["allow"] = sorted(ALL_TOOLS - NARROWED)
    m["tool_access"] = ta
    return m


class Counter:
    def __init__(self):
        self.calls = {"safe": 0, "danger": 0, "ask": 0}


@pytest.fixture
def counter():
    return Counter()


@pytest.fixture
def graph(counter):
    """A one-node graph running the guarded wrapper around a real ToolNode (as in build_agent)."""

    def safe(x: str = "") -> str:
        """safe tool"""
        counter.calls["safe"] += 1
        return f"safe:{x}"

    def danger(x: str = "") -> str:
        """danger tool"""
        counter.calls["danger"] += 1
        return f"danger:{x}"

    def ask() -> str:
        """tool that suspends via interrupt()"""
        counter.calls["ask"] += 1
        return f"answer:{interrupt('need input')}"

    tools = [StructuredTool.from_function(f) for f in (safe, danger, ask)]
    node = guarded_tool_node(ToolNode(tools), lambda msgs: frozenset({"safe", "ask"}))
    b = StateGraph(MessagesState)
    b.add_node("tools", node)
    b.add_edge(START, "tools")
    b.add_edge("tools", END)
    return b.compile(checkpointer=MemorySaver())


CFG = {"configurable": {"thread_id": "t"}}


def ai(*calls):
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": a, "id": i, "type": "tool_call"} for n, a, i in calls],
    )


def run(graph, *calls):
    out = graph.invoke({"messages": [HumanMessage(content="hi"), ai(*calls)]}, CFG)
    return out, [m for m in out["messages"][2:]]


def test_allowed_executes(graph, counter):
    _, out = run(graph, ("safe", {"x": "1"}, "c1"))
    assert [m.tool_call_id for m in out] == ["c1"]
    assert out[0].content == "safe:1"
    assert counter.calls["safe"] == 1


def test_denied_not_executed(graph, counter):
    _, out = run(graph, ("danger", {"x": "1"}, "c9"))
    assert counter.calls["danger"] == 0
    assert len(out) == 1 and isinstance(out[0], ToolMessage)
    assert out[0].tool_call_id == "c9" and out[0].name == "danger"
    assert json.loads(out[0].content) == {"error": "Tool 'danger' is not available for this request"}


def test_mixed_preserves_order(graph, counter):
    _, out = run(
        graph,
        ("danger", {}, "a"),
        ("safe", {"x": "1"}, "b"),
        ("danger", {}, "c"),
        ("safe", {"x": "2"}, "d"),
    )
    assert [m.tool_call_id for m in out] == ["a", "b", "c", "d"]
    assert counter.calls["safe"] == 2 and counter.calls["danger"] == 0
    assert out[1].content == "safe:1" and out[3].content == "safe:2"
    assert "not available" in out[0].content and "not available" in out[2].content


def test_interrupt_propagates_and_resumes(graph, counter):
    out = graph.invoke(
        {"messages": [HumanMessage(content="hi"), ai(("danger", {}, "a"), ("ask", {}, "b"))]}, CFG
    )
    assert out["__interrupt__"][0].value == "need input"
    assert counter.calls["danger"] == 0
    out = graph.invoke(Command(resume="yes"), CFG)
    tms = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tms] == ["a", "b"]
    assert tms[1].content == "answer:yes"
    assert counter.calls["danger"] == 0


def test_tool_exception_propagates():
    def bad() -> str:
        """raises"""
        raise GraphInterrupt()

    node = guarded_tool_node(ToolNode([StructuredTool.from_function(bad)]), lambda m: {"bad"})
    b = StateGraph(MessagesState)
    b.add_node("tools", node)
    b.add_edge(START, "tools")
    g = b.compile()
    # ToolNode re-raises GraphInterrupt untouched; the wrapper adds nothing on top
    out = g.invoke({"messages": [HumanMessage(content="x"), ai(("bad", {}, "1"))]})
    assert out["messages"][-1].type == "ai"  # state unchanged: interrupted, no ToolMessage


def test_resolve_allowed_fails_open(caplog):
    with caplog.at_level("ERROR"):
        intent, allowed = resolve_allowed({"tool_access": "garbage"}, [HumanMessage(content="what is a deductible")])
    assert allowed == ALL_TOOLS
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_resolve_allowed_info_narrowed_and_ambiguous():
    m = info_narrowed_manifest()
    intent, allowed = resolve_allowed(m, [HumanMessage(content="What is a deductible?")])
    assert intent == "info"
    assert not (allowed & NARROWED) and allowed
    intent, allowed = resolve_allowed(m, [HumanMessage(content="hello there")])
    assert intent == "general" and allowed == ALL_TOOLS
    # ambiguous: info + claims keywords
    intent, allowed = resolve_allowed(m, [HumanMessage(content="what is a claim deductible")])
    assert intent == "general" and allowed == ALL_TOOLS


def test_conversation_context():
    msgs = [
        HumanMessage(content="first"),
        ai(("file_claim", {}, "1")),
        ToolMessage(content="ok", tool_call_id="1"),
        HumanMessage(content=[{"type": "text", "text": "latest"}]),
    ]
    assert conversation_context(msgs) == ("latest", ["file_claim"])


def test_filter_tool_schemas_objects_and_dicts():
    def a() -> str:
        """a"""
        return ""

    def b() -> str:
        """b"""
        return ""

    tools = [StructuredTool.from_function(a), StructuredTool.from_function(b)]
    assert [t.name for t in filter_tool_schemas(tools, {"a"})] == ["a"]
    dicts = [{"type": "function", "function": {"name": "a"}}, {"name": "b"}, {"foo": 1}]
    assert filter_tool_schemas(dicts, {"a", "b"}) == dicts[:2]


class RecordingLLM(BaseChatModel):
    bound: list = []
    script: list = []

    @property
    def _llm_type(self):
        return "recording"

    def bind_tools(self, tools, **kwargs):
        self.bound.append([getattr(t, "name", None) for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def test_build_agent_end_to_end(monkeypatch):
    os.environ.setdefault("RUNNER_MODE", "aer")
    main = pytest.importorskip("agent_agent46.main")
    monkeypatch.setattr(main, "load_active_manifest", lambda store: (99, info_narrowed_manifest()))
    llm = RecordingLLM(
        bound=[],
        script=[ai(("resolve_claim", {"claim_id": "CLM-1", "resolution": "approved"}, "t1")), AIMessage(content="done")],
    )
    graph = main.build_agent(llm=llm)
    out = graph.invoke({"messages": [HumanMessage(content="What is a deductible?")]})
    # offered tools were filtered on every bind
    assert llm.bound and all(not (set(b) & NARROWED) and "explain_insurance_term" in b for b in llm.bound)
    # model named a tool it was never offered: guard denied it without executing
    denial = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert len(denial) == 1 and denial[0].tool_call_id == "t1"
    assert "not available" in denial[0].content
    assert out["messages"][-1].content == "done"
    assert not any(isinstance(m, SystemMessage) for m in out["messages"][:0])
