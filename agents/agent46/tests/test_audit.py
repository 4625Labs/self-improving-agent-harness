import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402
from test_court import FakeDb  # noqa: E402
from test_court_ui import TOKEN, FakeStore  # noqa: E402

from harness_evolve import audit, cli  # noqa: E402
from harness_evolve.audit import ConfirmationRequired, RemoteRefused, check_remote  # noqa: E402
from harness_evolve.court import CourtDocket  # noqa: E402
from harness_evolve.court_ui.app import create_app  # noqa: E402
from harness_evolve.loop import approve, reject, rollback  # noqa: E402


def _rows(db):
    return db["audit_log"].docs


def _court(path):
    path.mkdir(exist_ok=True)
    db = FakeDb()
    db["insurance_policies"].insert_one({"policy_number": "P", "risk_score": 0.4, "coverage_limit": 1})
    db["insurance_claims"].insert_one({"claim_id": "C1", "policy_number": "P", "claim_type": "auto",
                                       "claim_amount": 1.0, "description": "d", "recommendation": "review_recommended"})
    c = CourtDocket(db, path / "docket.json")
    cid = c.file_case(c.nominate("C1", "manual", "x")["case_id"])["case_id"]
    return db, c, cid


def test_verdict_refused_without_confirmation_audits_and_changes_nothing(tmp_path):
    db, c, cid = _court(tmp_path / "a")
    with pytest.raises(ConfirmationRequired):
        c.verdict(cid, "auto_approve", False, "ann")
    assert not c.docket_path.exists() and c._get(cid)["status"] == "filed"
    r = _rows(db)[-1]
    assert r["action"] == "court.verdict" and r["actor"] == "ann" and r["target"] == cid
    assert r["outcome"].startswith("refused") and r["host"] and r["ts"]


def test_confirmed_verdict_and_dismiss_change_state_and_audit(tmp_path):
    db, c, cid = _court(tmp_path / "a")
    c.verdict(cid, "auto_approve", True, "ann", confirmed=True)
    assert c._get(cid)["status"] == "convicted"
    assert _rows(db)[-1]["outcome"] == "ok" and _rows(db)[-1]["detail"]["critical"] is True
    db2, c2, cid2 = _court(tmp_path / "b")
    with pytest.raises(ConfirmationRequired):
        c2.dismiss(cid2, "bob")
    assert c2._get(cid2)["status"] == "filed"
    c2.dismiss(cid2, "bob", "n", confirmed=True)
    assert c2._get(cid2)["status"] == "dismissed"
    assert [r["outcome"] for r in _rows(db2)][-2:] == ["refused: confirmation required", "ok"]


def _candidate_store():
    s = FakeStore()
    s.propose(s.get(1)["manifest"], 1, "r", "a")
    s.get(2)["scores"] = {"gate": {"passed": True}}
    return s


def test_manifest_actions_refused_then_confirmed():
    db, s = FakeDb(), _candidate_store()
    for fn in (approve, reject, rollback):
        with pytest.raises(ConfirmationRequired):
            fn(s, 2, actor="ann", audit_db=db)
    assert s.get(2)["status"] == "candidate" and s.get_active()["version"] == 1
    assert [r["outcome"].split(":")[0] for r in _rows(db)] == ["refused"] * 3
    approve(s, 2, confirmed=True, actor="ann", audit_db=db)
    assert s.get_active()["version"] == 2 and _rows(db)[-1]["action"] == "manifest.approve"
    rollback(s, 1, confirmed=True, actor="ann", audit_db=db)
    assert s.get_active()["version"] == 1
    with pytest.raises(ValueError):
        approve(s, 99, confirmed=True, actor="ann", audit_db=db)
    assert _rows(db)[-1]["outcome"].startswith("error")


def test_audit_failure_never_breaks_action():
    class Boom:
        def __getitem__(self, k):
            raise RuntimeError("mongo down")

    s = _candidate_store()
    approve(s, 2, confirmed=True, actor="ann", audit_db=Boom())
    assert s.get_active()["version"] == 2
    audit.record(Boom(), "x", "a", "t", {}, "ok")  # does not raise
    assert audit.recent(Boom()) == []


def test_recent_newest_first_and_limited():
    db = FakeDb()
    for i in range(15):
        audit.record(db, "a", "me", i, {}, "ok")
    out = audit.recent(db, 10)
    assert len(out) == 10 and out[0]["target"] == 14 and "_id" not in out[0]


@pytest.mark.parametrize("uri,ok", [
    ("mongodb://localhost:53087/?directConnection=true", True), ("mongodb://127.0.0.1/", True),
    ("mongodb://[::1]:27017/", True), ("mongodb://u:p@host.docker.internal:1/", True),
    ("mongodb://db.example.com:27017/", False), ("mongodb+srv://u:p@cluster.mongodb.net/", False),
    ("mongodb://localhost,evil.example.com/", False)])
def test_remote_guard(uri, ok, monkeypatch):
    monkeypatch.delenv("EVOLVE_ALLOW_REMOTE", raising=False)
    if ok:
        check_remote(uri)
    else:
        with pytest.raises(RemoteRefused):
            check_remote(uri)
        check_remote(uri, allow_remote=True)
        monkeypatch.setenv("EVOLVE_ALLOW_REMOTE", "1")
        check_remote(uri)


def test_cli_remote_refused_but_readonly_works(monkeypatch, capsys):
    monkeypatch.delenv("EVOLVE_ALLOW_REMOTE", raising=False)
    remote = ["--uri", "mongodb://db.example.com:27017/"]
    for argv in (["approve", "2", "--yes"], ["cycle"], ["court", "dismiss", "x", "--by", "a", "--yes"],
                 ["court", "scan"]):
        assert cli.main(remote + argv) == 2, argv
        assert "non-local" in capsys.readouterr().err
    assert cli.main(remote + ["budget"]) == 0


def test_cli_confirmation_flags(monkeypatch, capsys):
    s = _candidate_store()
    db = FakeDb()
    s.audit_db = db
    monkeypatch.setattr(cli, "HarnessConfigStore", lambda uri, d: type("S", (), {
        "seed_if_empty": lambda self: None, "__getattr__": lambda self, k: getattr(s, k)})())
    assert cli.main(["approve", "2", "--by", "ann"]) == 2
    assert "needs explicit confirmation" in capsys.readouterr().err
    assert s.get_active()["version"] == 1 and _rows(db)[-1]["actor"] == "ann"
    assert cli.main(["approve", "2", "--yes", "--by", "ann"]) == 0
    assert s.get_active()["version"] == 2 and _rows(db)[-1]["outcome"] == "ok"


def _ui(tmp_path):
    db, c, cid = _court(tmp_path)
    store = _candidate_store()
    app = create_app(db, store, token=TOKEN, docket_path=tmp_path / "docket.json")
    return TestClient(app, follow_redirects=True), db, store, cid


def _post(c, url, **data):
    return c.post(url, data={"csrf": TOKEN, **data})


def test_ui_post_without_checkbox_refused_in_page(tmp_path):
    c, db, store, cid = _ui(tmp_path / "u")
    for url, data in ((f"/case/{cid}/verdict", dict(action="verdict", expected="auto_approve", adjuster="Ann")),
                      (f"/case/{cid}/verdict", dict(action="dismiss", adjuster="Ann")),
                      ("/version/2/approve", dict(reviewer="Rev")), ("/version/2/reject", dict(reviewer="Rev")),
                      ("/version/1/rollback", dict(reviewer="Rev"))):
        assert "needs explicit confirmation" in _post(c, url, **data).text
    assert store.get(2)["status"] == "candidate" and not (tmp_path / "u" / "docket.json").exists()
    assert [r["outcome"].split(":")[0] for r in _rows(db)] == ["refused"] * 5
    assert {r["actor"] for r in _rows(db)} == {"Ann", "Rev"}
    html = _post(c, "/version/2/approve", reviewer="Rev", confirm="1").text
    assert "v2 is now active" in html and store.get_active()["version"] == 2
    assert _rows(db)[-1]["outcome"] == "ok"
    assert "I understand this changes live state" in c.get("/").text


def test_ui_audit_rows_escaped_in_collapsed_details(tmp_path):
    c, db, store, cid = _ui(tmp_path / "u")
    audit.record(db, "x", "<script>alert(1)</script>", "t", {"k": "<b>"}, "ok")
    html = c.get("/").text
    assert "<script>alert(1)" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<details><summary>Activity log" in html
