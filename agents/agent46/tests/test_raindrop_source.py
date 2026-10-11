import json
import sys
from pathlib import Path

import pymongo
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness_evolve import cli
from harness_evolve import raindrop_source as rs
from harness_evolve.raindrop_source import analyse_events
from harness_evolve.raindrop_source import candidates_from_events as cfe
from tests.test_court import FakeDb

SIG = "claim_approved_despite_fraud_cues"


def ev(name, **p):
    return {"event": name, "properties": p}


def test_signal_event_name_is_candidate():
    r = cfe([ev("risk_assessed", claim_id="c1", recommendation="auto_approve"), ev(SIG, claim_id="c1")])
    assert r == [{"claim_id": "C1", "reason": f"Raindrop signal: {SIG} (risk_assessed recommended auto_approve)",
                  "evidence_count": 1}]


def test_signal_property_and_allowlist():
    r = cfe([ev("x", claim_id="a", signal="user_frustration"), ev("x", claim_id="b", signal_name="task_failure"),
             ev("x", claim_id="c", signal="other")])
    assert [c["claim_id"] for c in r] == ["A", "B"]
    assert cfe([ev("x", claim_id="c", signal="other")], signals=["other"])
    assert set(rs.DEFAULT_SIGNALS) == {SIG, "user_frustration", "task_failure"}


def test_dedupe_and_min_count():
    evs = [ev(SIG, claim_id="a"), ev(SIG, claim_id="A"), ev("task_failure", claim_id="b")]
    r = cfe(evs)
    assert len(r) == 2 and r[0]["evidence_count"] == 2
    assert [c["claim_id"] for c in cfe(evs, min_count=2)] == ["A"]


def test_review_resumed_disagreement_excluded():
    assert cfe([ev("review_resumed", claim_id="a", decision="approved", recommendation="manual_review_required"),
                ev("review_resumed", claim_id="b", decision="approved", recommendation="review_recommended", signal=SIG)]) == []


def test_limit_events_summarised_not_nominated():
    r = analyse_events([ev("step_cap_hit", claim_id="a", harness_version=3, steps=9, limit=8),
                        ev("tool_limit_hit", rule="per_tool", claim_id="a", harness_version=3),
                        ev("tool_limit_hit", rule="per_tool", harness_version=3)])
    assert r["candidates"] == []
    assert r["limit_pressure"] == {"rules": {"step_cap": 1, "per_tool": 2}, "harness_version": 3}


def test_malformed_skipped_and_never_raises():
    r = analyse_events([None, 5, "x", ev(SIG), ev(SIG, claim_id=7), ev(SIG, claim_id="ok")])
    assert [c["claim_id"] for c in r["candidates"]] == ["OK"] and r["skipped"] == 3
    assert cfe([{"event": 3, "properties": "bad"}]) == []
    assert cfe(None) == []


@pytest.fixture
def fake(monkeypatch, tmp_path):
    db = FakeDb()
    monkeypatch.setattr(pymongo, "MongoClient", lambda *a, **k: {"agent_agent46_local": db})
    monkeypatch.delenv("EVOLVE_ALLOW_REMOTE", raising=False)
    monkeypatch.setenv("EVOLVE_DOCKET_PATH", str(tmp_path / "docket.json"))
    f = tmp_path / "events.json"
    f.write_text(json.dumps([ev(SIG, claim_id="c1"), ev("step_cap_hit", harness_version=1)]))
    return db, f, tmp_path


def _cases(db):
    return [d for k, c in db.items() if k != "audit_log" for d in c.docs]


def test_cli_dry_run_does_not_nominate(fake, capsys):
    db, f, tmp = fake
    assert cli.main(["court", "ingest-events", str(f), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "C1" in out and "limit_pressure" in out
    assert _cases(db) == [] and "audit_log" not in db
    assert not (tmp / "docket.json").exists()


def test_cli_real_run_nominates_once_with_source_and_audit(fake):
    db, f, tmp = fake
    for _ in range(2):
        assert cli.main(["court", "ingest-events", str(f)]) == 0
    cases = _cases(db)
    assert len(cases) == 1 and cases[0]["source"] == "raindrop_signal" and cases[0]["claim_id"] == "C1"
    row = db["audit_log"].docs[0]
    assert row["action"] == "court.ingest" and row["outcome"] == "ok" and row["detail"]["nominated"] == 1
    assert not (tmp / "docket.json").exists()


def test_cli_jsonl_input(fake):
    db, f, tmp = fake
    f.write_text("\n".join(json.dumps(e) for e in [ev(SIG, claim_id="z9"), ev("task_failure", claim_id="z8")]) + "\nnot json\n")
    assert cli.main(["court", "ingest-events", str(f)]) == 0
    assert len(_cases(db)) == 2


def test_cli_remote_refused_without_allow_remote(fake, capsys):
    db, f, _ = fake
    remote = ["--uri", "mongodb://db.example.com:27017/"]
    assert cli.main(remote + ["court", "ingest-events", str(f)]) == 2
    assert "non-local" in capsys.readouterr().err
    assert _cases(db) == []
    assert cli.main(remote + ["court", "ingest-events", str(f), "--dry-run"]) == 0
