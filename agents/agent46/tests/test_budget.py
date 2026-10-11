"""Persistent daily LLM budget. Fake clients and tmp_path cache dirs only: no network, no real LLM call."""

import copy
import json
import logging
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_agent46.harness_config import DEFAULT_MANIFEST  # noqa: E402
from harness_evolve import budget as budget_mod  # noqa: E402
from harness_evolve import cli  # noqa: E402
from harness_evolve import proposer as proposer_mod  # noqa: E402
from harness_evolve.budget import BudgetExceeded, DailyBudget  # noqa: E402
from harness_evolve.proposer import LLMProposer  # noqa: E402

GOOD = json.dumps({"ops": [{"op": "add_keyword", "level": "high", "keyword": "arson"}], "rationale": "r"})
DAY1 = "2026-10-10"
DAY2 = "2026-10-11"


class Fake:
    def __init__(self, reply=GOOD, exc=None):
        self.reply, self.exc, self.calls = reply, exc, []

    def __call__(self, prompt, model, max_tokens):
        self.calls.append(prompt)
        if self.exc is not None:
            raise self.exc
        return self.reply


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    proposer_mod.reset_call_count()
    for k in ("EVOLVE_LLM_MAX_CALLS", "EVOLVE_LLM_MAX_CALLS_PER_DAY", "EVOLVE_LLM_MAX_TOKENS_PER_DAY"):
        monkeypatch.delenv(k, raising=False)
    yield
    proposer_mod.reset_call_count()


def _budget(tmp_path, **kw):
    return DailyBudget(tmp_path / "cache", **kw)


def _ledger(tmp_path) -> dict:
    return json.loads((tmp_path / "cache" / "ledger.json").read_text())["days"]


def _prop(tmp_path, client, **kw):
    return LLMProposer(client=client, cache_dir=tmp_path / "cache", **kw)


# --- defaults and env --------------------------------------------------------

def test_defaults_and_env_overrides(tmp_path, monkeypatch):
    b = _budget(tmp_path)
    assert (b.max_calls, b.max_tokens) == (10, 50000)
    monkeypatch.setenv("EVOLVE_LLM_MAX_CALLS_PER_DAY", "4")
    monkeypatch.setenv("EVOLVE_LLM_MAX_TOKENS_PER_DAY", "900")
    b = _budget(tmp_path)
    assert (b.max_calls, b.max_tokens) == (4, 900)


def test_bad_env_value_is_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLVE_LLM_MAX_CALLS_PER_DAY", "lots")
    with pytest.raises(ValueError, match="EVOLVE_LLM_MAX_CALLS_PER_DAY"):
        _budget(tmp_path)


# --- daily call limit / token limit -----------------------------------------

def test_daily_call_limit(tmp_path):
    b = _budget(tmp_path, max_calls=2, max_tokens=10**9)
    b.check_and_reserve(1, day=DAY1)
    b.check_and_reserve(1, day=DAY1)
    with pytest.raises(BudgetExceeded, match="EVOLVE_LLM_MAX_CALLS_PER_DAY"):
        b.check_and_reserve(1, day=DAY1)
    assert b.remaining(DAY1)["calls_remaining"] == 0
    assert _ledger(tmp_path)[DAY1]["calls"] == 2  # refused reservation changed nothing


def test_daily_token_limit(tmp_path):
    b = _budget(tmp_path, max_calls=100, max_tokens=1000)
    b.check_and_reserve(600, day=DAY1)
    with pytest.raises(BudgetExceeded, match="EVOLVE_LLM_MAX_TOKENS_PER_DAY"):
        b.check_and_reserve(401, day=DAY1)
    b.check_and_reserve(400, day=DAY1)  # exactly at the limit is allowed
    assert b.remaining(DAY1)["tokens_remaining"] == 0
    assert _ledger(tmp_path)[DAY1] == {"calls": 2, "tokens": 1000}


def test_record_replaces_estimate_with_actual(tmp_path):
    b = _budget(tmp_path, max_calls=5, max_tokens=1000)
    r = b.check_and_reserve(800, day=DAY1)
    b.record(120, r)
    assert _ledger(tmp_path)[DAY1] == {"calls": 1, "tokens": 120}
    assert b.remaining(DAY1)["tokens_remaining"] == 880


def test_never_negative(tmp_path):
    b = _budget(tmp_path, max_calls=5, max_tokens=1000)
    r = b.check_and_reserve(10, day=DAY1)
    b.record(-500, r)  # nonsense input must not drive counters below zero
    b.release(r)
    b.release(r)
    assert _ledger(tmp_path)[DAY1] == {"calls": 0, "tokens": 0}
    out = b.remaining(DAY1)
    assert out["calls_used"] == 0 and out["calls_remaining"] == 5 and out["tokens_remaining"] == 1000


def test_release_gives_back_call_and_tokens(tmp_path):
    b = _budget(tmp_path, max_calls=1, max_tokens=1000)
    r = b.check_and_reserve(900, day=DAY1)
    b.release(r)
    b.check_and_reserve(900, day=DAY1)  # budget is available again


# --- new day reset ------------------------------------------------------------

def test_new_day_resets(tmp_path):
    b = _budget(tmp_path, max_calls=1, max_tokens=1000)
    b.check_and_reserve(1000, day=DAY1)
    with pytest.raises(BudgetExceeded):
        b.check_and_reserve(1, day=DAY1)
    b.check_and_reserve(1000, day=DAY2)  # fresh UTC date, fresh limits
    assert b.remaining(DAY2)["calls_used"] == 1
    assert b.remaining(DAY1)["calls_used"] == 1  # yesterday's record is kept, not reset in place


def test_remaining_shape_and_default_date(tmp_path):
    out = _budget(tmp_path, max_calls=3, max_tokens=50).remaining()
    assert set(out) == {"date", "calls_used", "calls_remaining", "tokens_used", "tokens_remaining",
                        "max_calls_per_day", "max_tokens_per_day"}
    assert out["date"] == budget_mod.utc_today()
    assert not (tmp_path / "cache").exists()  # reading never creates the ledger


def test_old_days_are_pruned(tmp_path):
    b = _budget(tmp_path, max_calls=100, max_tokens=10**9)
    for i in range(40):
        b.check_and_reserve(1, day=f"2026-01-{i + 1:02d}" if i < 31 else f"2026-02-{i - 30:02d}")
    assert len(_ledger(tmp_path)) == budget_mod.KEEP_DAYS


# --- corrupt / missing ledger ----------------------------------------------

def test_missing_ledger_starts_fresh(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        out = _budget(tmp_path).remaining(DAY1)
    assert out["calls_used"] == 0 and not caplog.records  # missing is normal, not a warning


@pytest.mark.parametrize("content", ["{not json", "[]", '{"days": 3}', '{"days": {"x": {"calls": "a"}}}', ""])
def test_corrupt_ledger_starts_fresh_with_warning(tmp_path, caplog, content):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "ledger.json").write_text(content)
    b = _budget(tmp_path, max_calls=2, max_tokens=10**9)
    with caplog.at_level(logging.WARNING, logger="harness_evolve.budget"):
        assert b.remaining(DAY1)["calls_used"] == 0
        b.check_and_reserve(5, day=DAY1)  # and it heals on the next write
    assert any("starting fresh" in r.message for r in caplog.records)
    assert _ledger(tmp_path)[DAY1] == {"calls": 1, "tokens": 5}


def test_corrupt_ledger_does_not_crash_proposer(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "ledger.json").write_text("garbage")
    fake = Fake()
    out = _prop(tmp_path, fake).propose(copy.deepcopy(DEFAULT_MANIFEST), [], {})
    assert out["ops"][0]["keyword"] == "arson" and len(fake.calls) == 1


# --- atomic writes and concurrency -----------------------------------------

def test_write_is_atomic_no_temp_files_left(tmp_path, monkeypatch):
    b = _budget(tmp_path, max_calls=10, max_tokens=10**9)
    b.check_and_reserve(1, day=DAY1)
    assert sorted(p.name for p in (tmp_path / "cache").iterdir() if p.name.startswith(".ledger-")) == []

    def boom(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr(budget_mod.os, "replace", boom)
    with pytest.raises(OSError):
        b.check_and_reserve(1, day=DAY1)
    monkeypatch.undo()
    assert _ledger(tmp_path)[DAY1]["calls"] == 1  # failed write left the old ledger intact
    assert not [p for p in (tmp_path / "cache").iterdir() if p.name.endswith(".tmp")]


def test_concurrent_reservations_respect_limit(tmp_path):
    b = _budget(tmp_path, max_calls=10, max_tokens=10**9)
    ok, refused = [], []

    def worker():
        try:
            b.check_and_reserve(1, day=DAY1)
            ok.append(1)
        except BudgetExceeded:
            refused.append(1)

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ok) == 10 and len(refused) == 30
    assert _ledger(tmp_path)[DAY1]["calls"] == 10  # file is valid JSON with the exact count


# --- LLMProposer integration -----------------------------------------------

def test_cache_hit_is_free_and_skips_ledger(tmp_path):
    fake = Fake()
    p = _prop(tmp_path, fake, max_calls=10)
    p.propose(copy.deepcopy(DEFAULT_MANIFEST), [], {"n": 1})
    before = (tmp_path / "cache" / "ledger.json").read_text()
    # Same prompt again: served from cache; ledger must be byte-identical and no call made.
    proposer_mod.reset_call_count()
    second = Fake()
    out = _prop(tmp_path, second, max_calls=10).propose(copy.deepcopy(DEFAULT_MANIFEST), [], {"n": 1})
    assert second.calls == [] and out["ops"]
    assert (tmp_path / "cache" / "ledger.json").read_text() == before


def test_cache_hit_does_not_consume_daily_budget(tmp_path):
    fake = Fake()
    b = _budget(tmp_path, max_calls=1, max_tokens=10**9)
    p = _prop(tmp_path, fake, budget=b)
    p.propose(copy.deepcopy(DEFAULT_MANIFEST), [], {"n": 1})
    assert b.remaining()["calls_remaining"] == 0
    for _ in range(3):  # cached prompt keeps working after the daily call limit is spent
        assert p.propose(copy.deepcopy(DEFAULT_MANIFEST), [], {"n": 1})["ops"]
    assert len(fake.calls) == 1


def test_real_call_reserves_then_records_actual(tmp_path):
    fake = Fake()
    b = _budget(tmp_path, max_calls=5, max_tokens=50000)
    p = _prop(tmp_path, fake, budget=b, max_tokens=600)
    p.propose(copy.deepcopy(DEFAULT_MANIFEST), [], {})
    entry = _ledger(tmp_path)[budget_mod.utc_today()]
    assert entry["calls"] == 1
    prompt_tokens = proposer_mod.estimate_tokens(fake.calls[0])
    assert entry["tokens"] == prompt_tokens + proposer_mod.estimate_tokens(GOOD)  # actual, not the 600 reservation


def test_reservation_refused_before_call_when_daily_tokens_exhausted(tmp_path):
    fake = Fake()
    b = _budget(tmp_path, max_calls=10, max_tokens=100)  # reservation (prompt + 600) cannot fit
    p = _prop(tmp_path, fake, budget=b, max_tokens=600)
    with pytest.raises(BudgetExceeded, match="EVOLVE_LLM_MAX_TOKENS_PER_DAY"):
        p.propose(copy.deepcopy(DEFAULT_MANIFEST), [], {})
    assert fake.calls == []
    assert proposer_mod._CALLS == 0


def test_daily_call_limit_stops_proposer(tmp_path):
    fake = Fake()
    p = _prop(tmp_path, fake, budget=_budget(tmp_path, max_calls=2, max_tokens=10**9), max_calls=99)
    for i in range(2):
        p.propose(copy.deepcopy(DEFAULT_MANIFEST), [], {"n": i})
    with pytest.raises(BudgetExceeded, match="EVOLVE_LLM_MAX_CALLS_PER_DAY"):
        p.propose(copy.deepcopy(DEFAULT_MANIFEST), [], {"n": 2})
    assert len(fake.calls) == 2


def test_failed_call_releases_budget(tmp_path):
    fake = Fake(exc=RuntimeError("network down"))
    b = _budget(tmp_path, max_calls=1, max_tokens=50000)
    p = _prop(tmp_path, fake, budget=b)
    with pytest.raises(RuntimeError, match="network down"):
        p.propose(copy.deepcopy(DEFAULT_MANIFEST), [], {})
    assert b.remaining()["calls_used"] == 0 and b.remaining()["tokens_used"] == 0
    assert proposer_mod._CALLS == 0
    fake.exc = None  # the failed call did not eat the single daily call
    assert p.propose(copy.deepcopy(DEFAULT_MANIFEST), [], {})["ops"]


def test_unparseable_reply_keeps_call_and_actual_tokens(tmp_path):
    fake = Fake(reply="not json at all")
    b = _budget(tmp_path, max_calls=5, max_tokens=50000)
    p = _prop(tmp_path, fake, budget=b)
    with pytest.raises(Exception):
        p.propose(copy.deepcopy(DEFAULT_MANIFEST), [], {})
    # A reply came back, so the call was really made and its tokens are spent.
    out = b.remaining()
    assert out["calls_used"] == 1 and out["tokens_used"] > 0


# --- CLI ----------------------------------------------------------------------

def test_cli_budget_prints_without_mongo(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", tmp_path / "cache")

    def no_store(*a, **k):
        raise AssertionError("store reached")

    monkeypatch.setattr(cli, "HarnessConfigStore", no_store)
    assert cli.main(["budget"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["calls_remaining"] == 10 and out["tokens_remaining"] == 50000
    assert out["max_calls_per_day"] == 10 and out["max_tokens_per_day"] == 50000


def test_cli_budget_reflects_spend(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    _budget(tmp_path, max_calls=10, max_tokens=50000).check_and_reserve(250, day=budget_mod.utc_today())
    assert cli.main(["budget"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["calls_remaining"] == 9 and out["tokens_remaining"] == 49750
