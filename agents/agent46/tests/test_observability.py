import importlib

from agent_agent46 import observability


def _reset(monkeypatch, key=None, content=None):
    for k, v in (("RAINDROP_WRITE_KEY", key), ("RAINDROP_SEND_CONTENT", content)):
        monkeypatch.delenv(k, raising=False) if v is None else monkeypatch.setenv(k, v)
    importlib.reload(observability)


class FakeSdk:
    def __init__(self):
        self.calls = []

    def track_ai(self, **kw):
        self.calls.append(kw)


def test_noop_without_key(monkeypatch):
    _reset(monkeypatch)
    observability.tool_denied("file_claim")
    assert observability._client() is None


def test_event_carries_version_and_withholds_content(monkeypatch):
    _reset(monkeypatch, key="k")
    sdk = FakeSdk()
    observability._state.update(client=sdk, tried=True)
    observability.track_turn(user_id="u", session_id="s", input_text="my ssn", output_text="ok",
                             harness_version=3, intent="claims")
    call = sdk.calls[0]
    assert call["properties"]["harness_version"] == 3
    assert call["input"] == "[content withheld]"


def test_content_sent_only_when_opted_in(monkeypatch):
    _reset(monkeypatch, key="k", content="true")
    sdk = FakeSdk()
    observability._state.update(client=sdk, tried=True)
    observability.track_event("x", input_text="hello")
    assert sdk.calls[0]["input"] == "hello"


def test_sdk_errors_never_raise(monkeypatch):
    _reset(monkeypatch, key="k")

    class Boom:
        def track_ai(self, **kw):
            raise RuntimeError("down")

    observability._state.update(client=Boom(), tried=True)
    observability.tool_denied("x")


def test_risk_assessed_carries_claim_fields_and_length_not_text(monkeypatch):
    _reset(monkeypatch, key="k")
    sdk = FakeSdk()
    observability._state.update(client=sdk, tried=True)
    observability.risk_assessed(
        claim_id="CLM-1", risk="low", recommendation="auto_approve", harness_version=7,
        user_id="u", claim_amount=250.0, claim_type="glass", policy_number="POL-9",
        description_len=42, previous_claims_count=2,
    )
    props = sdk.calls[0]["properties"]
    assert props["claim_amount"] == 250.0
    assert props["claim_type"] == "glass"
    assert props["policy_number"] == "POL-9"
    assert props["description_len"] == 42
    assert props["previous_claims_count"] == 2
    assert props["harness_version"] == 7
    assert "description" not in props


def test_risk_assessed_omits_unknown_fields(monkeypatch):
    _reset(monkeypatch, key="k")
    sdk = FakeSdk()
    observability._state.update(client=sdk, tried=True)
    observability.risk_assessed(
        claim_id="CLM-1", risk="low", recommendation="auto_approve", harness_version=7, user_id="u",
    )
    props = sdk.calls[0]["properties"]
    assert "claim_amount" not in props
    assert "previous_claims_count" not in props


def test_review_resumed_carries_recommendation_and_version(monkeypatch):
    _reset(monkeypatch, key="k")
    sdk = FakeSdk()
    observability._state.update(client=sdk, tried=True)
    observability.review_resumed(
        claim_id="CLM-1", decision="approved", user_id="u",
        recommendation="manual_review_required", harness_version=5,
    )
    props = sdk.calls[0]["properties"]
    assert props["decision"] == "approved"
    assert props["recommendation"] == "manual_review_required"
    assert props["harness_version"] == 5


def test_tool_denied_carries_intent_and_session(monkeypatch):
    _reset(monkeypatch, key="k")
    sdk = FakeSdk()
    observability._state.update(client=sdk, tried=True)
    observability.tool_denied("file_claim", intent="claims", session_id="sess-1")
    call = sdk.calls[0]
    assert call["properties"]["tool"] == "file_claim"
    assert call["properties"]["intent"] == "claims"
    assert call["convo_id"] == "sess-1"


def test_changed_events_are_noop_without_key(monkeypatch):
    _reset(monkeypatch)
    observability.risk_assessed(
        claim_id="CLM-1", risk="low", recommendation="x", harness_version=1, user_id="u",
        claim_amount=1.0, description_len=3,
    )
    observability.review_resumed(claim_id="CLM-1", decision="denied", user_id="u",
                                 recommendation="x", harness_version=1)
    observability.tool_denied("file_claim", intent="claims", session_id="s")
    assert observability._client() is None


def test_risk_assessed_never_raises_on_unprintable_value(monkeypatch):
    _reset(monkeypatch, key="k")

    class Boom:
        def __str__(self):
            raise RuntimeError("no str")

    sdk = FakeSdk()
    observability._state.update(client=sdk, tried=True)
    observability.risk_assessed(
        claim_id="CLM-1", risk=Boom(), recommendation=Boom(), harness_version=1, user_id="u",
    )
    assert sdk.calls[0]["properties"]["risk"] == "<unprintable>"


def test_guarded_denial_reports_intent_and_session(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    from agent_agent46.tool_guard import guarded_tool_node

    _reset(monkeypatch, key="k")
    sdk = FakeSdk()
    observability._state.update(client=sdk, tried=True)

    class NoTools:
        def invoke(self, state, config=None):
            return {"messages": []}

    ai = AIMessage(content="", tool_calls=[{"name": "file_claim", "args": {}, "id": "t1"}])
    node = guarded_tool_node(
        NoTools(),
        lambda history: ["lookup_policy"],
        intent_of=lambda history: "policy",
    )
    node({"messages": [HumanMessage(content="hi"), ai], "session_id": "sess-9"})
    call = sdk.calls[0]
    assert call["properties"]["tool"] == "file_claim"
    assert call["properties"]["intent"] == "policy"
    assert call["convo_id"] == "sess-9"
