import json
import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agent_agent46 import tool_limits as tl
from agent_agent46.tool_guard import guarded_tool_node
from tests.test_tool_guard import RecordingLLM, ai


def call(name="safe", args=None, id="c"):
    return {"name": name, "args": args or {}, "id": id, "type": "tool_call"}


def history(n, *, name="safe", distinct=True, human=True):
    """n prior tool calls in the current turn, each its own AIMessage."""
    msgs = [HumanMessage(content="hi")] if human else []
    for i in range(n):
        args = {"x": str(i) if distinct else "same"}
        msgs.append(AIMessage(content="", tool_calls=[call(name, args, f"h{i}")]))
        msgs.append(ToolMessage(content="ok", tool_call_id=f"h{i}"))
    return msgs


def test_benign_trace_passes():
    assert tl.check_call(history(3), call(args={"x": "new", "n": 5, "amount": 0})) is None


def test_per_turn_limit():
    assert tl.check_call(history(tl.MAX_TOOL_CALLS_PER_TURN - 1), call(args={"x": "n"})) is None
    r = tl.check_call(history(tl.MAX_TOOL_CALLS_PER_TURN), call(args={"x": "n"}))
    assert r and r.startswith(tl.RULE_TURN)


def test_per_turn_resets_on_new_human_message():
    h = history(tl.MAX_TOOL_CALLS_PER_TURN) + [HumanMessage(content="next")]
    assert tl.check_call(h, call(args={"x": "n"})) is None


def test_identical_loop_across_mixed_history():
    h = [HumanMessage(content="hi")]
    for i in range(tl.MAX_IDENTICAL_CALLS_PER_TURN):
        h.append(AIMessage(content="", tool_calls=[call("safe", {"x": "1"}, f"a{i}"), call("other", {}, f"b{i}")]))
        h.append(ToolMessage(content="ok", tool_call_id=f"a{i}"))
    assert tl.check_call(h, call("safe", {"x": "2"})) is None  # different args
    assert tl.check_call(h, call("other2", {"x": "1"})) is None  # different tool
    r = tl.check_call(h, call("safe", {"x": "1"}))
    assert r and r.startswith(tl.RULE_IDENTICAL)


def test_per_thread_limit():
    # many turns, each small, so only the thread cap can fire
    h = []
    for i in range(tl.MAX_TOOL_CALLS_PER_THREAD):
        h += [HumanMessage(content="q"), AIMessage(content="", tool_calls=[call("safe", {"x": str(i)}, f"t{i}")])]
    h.append(HumanMessage(content="again"))
    r = tl.check_call(h, call(args={"x": "z"}))
    assert r and r.startswith(tl.RULE_THREAD)
    assert tl.check_call(h[2:], call(args={"x": "z"})) is None


@pytest.mark.parametrize(
    "args",
    [
        {"n": -1},
        {"n": float("nan")},
        {"n": float("inf")},
        {"nested": {"deep": -0.5}},
        {"items": [1, -2]},
        {"s": "x" * (tl.MAX_STRING_ARG_CHARS + 1)},
        {"claim_amount": tl.MAX_CLAIM_AMOUNT + 1},
    ],
)
def test_argument_sanity_rejects(args):
    r = tl.check_call(history(0), call(args=args))
    assert r and r.startswith(tl.RULE_ARGS)


def test_argument_sanity_accepts_edges():
    ok = {"claim_amount": tl.MAX_CLAIM_AMOUNT, "s": "x" * tl.MAX_STRING_ARG_CHARS, "flag": True, "z": 0}
    assert tl.check_call(history(0), call(args=ok)) is None


def test_negative_control_huge_limit(monkeypatch):
    h = history(tl.MAX_TOOL_CALLS_PER_TURN)
    c = call(args={"x": "n"})
    assert tl.check_call(h, c) is not None
    monkeypatch.setattr(tl, "MAX_TOOL_CALLS_PER_TURN", 10**9)
    assert tl.check_call(h, c) is None


@pytest.fixture
def ran():
    return []


def make_graph(ran, monkeypatch=None):
    def safe(x: str = "") -> str:
        """safe tool"""
        ran.append(x)
        return f"safe:{x}"

    node = guarded_tool_node(ToolNode([StructuredTool.from_function(safe)]), lambda m: frozenset({"safe"}))
    b = StateGraph(MessagesState)
    b.add_node("tools", node)
    b.add_edge(START, "tools")
    b.add_edge("tools", END)
    return b.compile()


def test_partial_denial_other_calls_run(ran):
    g = make_graph(ran)
    msgs = history(0) + [ai(("safe", {"x": "1"}, "a"), ("safe", {"x": "x" * 6000}, "b"), ("safe", {"x": "3"}, "c"))]
    out = g.invoke({"messages": msgs})["messages"][2:]
    assert [m.tool_call_id for m in out] == ["a", "b", "c"]
    assert out[1].status == "error" and "safety limit" in out[1].content
    assert "x" * 50 not in out[1].content
    assert out[0].content == "safe:1" and out[2].content == "safe:3"
    assert ran == ["1", "3"]


def test_in_message_calls_count_toward_limits(ran):
    g = make_graph(ran)
    calls = [("safe", {"x": "same"}, f"i{i}") for i in range(tl.MAX_IDENTICAL_CALLS_PER_TURN + 1)]
    out = g.invoke({"messages": [HumanMessage(content="hi"), ai(*calls)]})["messages"][2:]
    assert [m.status for m in out] == ["success"] * tl.MAX_IDENTICAL_CALLS_PER_TURN + ["error"]
    assert len(ran) == tl.MAX_IDENTICAL_CALLS_PER_TURN


def test_check_call_raising_fails_open(ran, monkeypatch, caplog):
    def boom(h, c):
        raise RuntimeError("boom")

    monkeypatch.setattr(tl, "check_call", boom)
    g = make_graph(ran)
    with caplog.at_level("ERROR"):
        out = g.invoke({"messages": [HumanMessage(content="hi"), ai(("safe", {"x": "-"}, "a"))]})["messages"][2:]
    assert out[0].content == "safe:-"
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_limit_telemetry(ran, monkeypatch):
    from agent_agent46 import observability

    seen = []
    monkeypatch.setattr(observability, "tool_limit_hit", lambda rule, tool, **kw: seen.append((rule, tool)))
    make_graph(ran).invoke({"messages": [HumanMessage(content="hi"), ai(("safe", {"x": "9" * 6000}, "a"))]})
    assert seen == [(tl.RULE_ARGS, "safe")]


def test_step_cap_short_circuits_llm():
    os.environ.setdefault("RUNNER_MODE", "aer")
    main = pytest.importorskip("agent_agent46.main")
    llm = RecordingLLM(bound=[], script=[])  # any LLM call would raise IndexError on pop
    graph = main.build_agent(llm=llm)
    msgs = [HumanMessage(content="hello")]
    for i in range(tl.MAX_LLM_STEPS_PER_TURN):
        msgs.append(AIMessage(content="", tool_calls=[call("safe", {"x": str(i)}, f"s{i}")]))
        msgs.append(ToolMessage(content="ok", tool_call_id=f"s{i}"))
    out = graph.invoke({"messages": msgs})
    assert llm.bound == []
    last = out["messages"][-1]
    assert isinstance(last, AIMessage) and not last.tool_calls
    assert "human" in last.content.lower()


def test_thread_cap_never_strands_a_claim_in_flight():
    """HANDOFF principle 15: after the thread cap, claim-completing tools must still run."""
    from langchain_core.messages import AIMessage, HumanMessage
    from agent_agent46 import tool_limits as tl

    hist = [HumanMessage(content="old")]
    for i in range(tl.MAX_TOOL_CALLS_PER_THREAD):
        hist.append(AIMessage(content="", tool_calls=[{"id": f"c{i}", "name": "lookup_policy", "args": {"n": i}}]))
    hist.append(HumanMessage(content="finish my claim"))
    assert tl.check_call(hist, {"id": "x", "name": "lookup_policy", "args": {"n": 999}}).startswith(tl.RULE_THREAD)
    for name in ("human_review", "resolve_claim", "send_notification", "check_claim_status"):
        assert tl.check_call(hist, {"id": "y", "name": name, "args": {}}) is None
