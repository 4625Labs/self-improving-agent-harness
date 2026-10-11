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
