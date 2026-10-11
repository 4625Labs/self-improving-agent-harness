"""LLM proposer budget controls. Everything uses a fake client: no network, no real LLM call."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_agent46.harness_config import DEFAULT_MANIFEST  # noqa: E402
from evals.run_eval import evaluate  # noqa: E402
from harness_evolve import proposer as proposer_mod  # noqa: E402
from harness_evolve.patch import PatchError  # noqa: E402
from harness_evolve.proposer import BudgetExceeded, LLMProposer  # noqa: E402

GOOD = json.dumps({"ops": [{"op": "add_keyword", "level": "high", "keyword": "arson"}], "rationale": "r"})


class Fake:
    def __init__(self, reply=GOOD):
        self.reply, self.calls = reply, []

    def __call__(self, prompt, model, max_tokens):
        self.calls.append((prompt, model, max_tokens))
        return self.reply


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    proposer_mod.reset_call_count()
    monkeypatch.delenv("EVOLVE_LLM_MAX_CALLS", raising=False)
    yield
    proposer_mod.reset_call_count()


def _dev():
    return evaluate(DEFAULT_MANIFEST, "dev")


def _mk(tmp_path, client, **kw):
    return LLMProposer(client=client, cache_dir=tmp_path / "cache", **kw)


def test_one_call_per_cycle_and_capped_prompt(tmp_path):
    dev = _dev()
    fails = dev["failures"]
    assert fails
    # Synthesize more failures than the cap, reusing real scenario ids.
    many = [copy.deepcopy(fails[i % len(fails)]) | {"critical_violation": False, "severity": "low"} for i in range(20)]
    many[5]["severity"] = "critical"
    many[5]["critical_violation"] = True
    fake = Fake()
    p = _mk(tmp_path, fake, max_failures=8)
    p.propose(copy.deepcopy(DEFAULT_MANIFEST), many, {})
    assert len(fake.calls) == 1
    prompt, model, max_tokens = fake.calls[0]
    assert "12 more failing dev scenarios omitted" in prompt
    assert prompt.count('"expected"') == 8
    assert max_tokens == 600
    from agent_agent46.llm import MODEL

    assert model == MODEL


def test_critical_first():
    fails = [{"id": "a", "expected": "manual_review_required", "got": "auto_approve", "severity": "low"},
             {"id": "z", "expected": "manual_review_required", "got": "auto_approve", "severity": "critical"}]
    shown, omitted = LLMProposer(max_failures=1)._cap(fails)
    assert [f["id"] for f in shown] == ["z"] and omitted == 1


def test_cache_hit_makes_zero_calls(tmp_path):
    fails = _dev()["failures"]
    f1, f2 = Fake(), Fake()
    _mk(tmp_path, f1).propose(DEFAULT_MANIFEST, fails, {})
    proposer_mod.reset_call_count()
    out = _mk(tmp_path, f2).propose(DEFAULT_MANIFEST, fails, {})
    assert len(f1.calls) == 1 and f2.calls == []
    assert out["ops"][0]["keyword"] == "arson"
    assert list((tmp_path / "cache").glob("*.json"))


def test_budget_exceeded(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLVE_LLM_MAX_CALLS", "2")
    fake = Fake()
    p = _mk(tmp_path, fake)
    for i in range(2):
        p.propose(DEFAULT_MANIFEST, [], {"n": i})  # distinct prompts, no cache hits
    with pytest.raises(BudgetExceeded, match="EVOLVE_LLM_MAX_CALLS"):
        p.propose(DEFAULT_MANIFEST, [], {"n": 99})
    assert len(fake.calls) == 2


def test_budget_default_is_three(tmp_path):
    fake = Fake()
    p = _mk(tmp_path, fake)
    for i in range(3):
        p.propose(DEFAULT_MANIFEST, [], {"n": i})
    with pytest.raises(BudgetExceeded):
        p.propose(DEFAULT_MANIFEST, [], {"n": 3})


def test_bad_reply_raises_without_retry_or_cache(tmp_path):
    fake = Fake("not json at all")
    p = _mk(tmp_path, fake)
    with pytest.raises(PatchError):
        p.propose(DEFAULT_MANIFEST, [], {})
    assert len(fake.calls) == 1
    # Only the daily ledger may exist; no cache entry is written for an unparseable reply.
    cache = tmp_path / "cache"
    assert {p.name for p in cache.glob("*.json")} <= {"ledger.json"}


def test_parse_fenced_json():
    raw = 'Here:\n```json\n{"ops": [], "rationale": "x"}\n```\nthanks'
    assert LLMProposer.parse(raw) == {"ops": [], "rationale": "x"}
    with pytest.raises(PatchError):
        LLMProposer.parse('{"ops": [], "extra": 1}')


def test_dry_run_makes_no_call_and_estimates(tmp_path):
    fake = Fake()
    info = _mk(tmp_path, fake).dry_run(DEFAULT_MANIFEST, _dev()["failures"], {})
    assert fake.calls == [] and info["estimated_prompt_tokens"] > 0 and info["max_output_tokens"] == 600


def test_cli_dry_run_never_calls(monkeypatch, capsys):
    from harness_evolve import cli

    def boom(*a, **k):
        raise AssertionError("LLM call attempted")

    monkeypatch.setattr(proposer_mod, "_default_client", boom)
    monkeypatch.setattr(proposer_mod.LLMProposer, "__init__", _wrap_init(boom))
    assert cli.main(["--proposer", "llm", "--dry-run", "cycle"]) == 0
    assert "estimated_prompt_tokens" in capsys.readouterr().out


def _wrap_init(boom):
    orig = proposer_mod.LLMProposer.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self._client = boom

    return init


def test_llm_never_reads_heldout(tmp_path, monkeypatch):
    seen = []
    real = proposer_mod.load_scenarios
    monkeypatch.setattr(proposer_mod, "load_scenarios", lambda s: seen.append(s) or real(s))
    _mk(tmp_path, Fake()).propose(copy.deepcopy(DEFAULT_MANIFEST), _dev()["failures"], {})
    assert seen and set(seen) == {"dev"}
