import copy
import logging
import sys
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_agent46.harness_config import DEFAULT_MANIFEST  # noqa: E402

from harness_evolve import atlas_sync, cli  # noqa: E402
from harness_evolve.atlas_sync import mask_uri, plan  # noqa: E402

PASSWORD = "S3cretPw0rd"
URI = f"mongodb+srv://atlasuser:{PASSWORD}@cluster0.abcde.mongodb.net/?retryWrites=true"
ARGS = ["--to-uri", URI, "--to-db", "t", "--from-uri", "mongodb://localhost:1/", "--from-db", "s"]
APPLY = ["--apply", "--yes", "--by", "ann", "--allow-remote"]


class FakeColl:
    def __init__(self, name, log):
        self.name, self.docs, self.log = name, [], log

    def find(self, flt=None, proj=None):
        self.log.add(self.name)
        return [copy.deepcopy(d) for d in self.docs]

    def insert_one(self, d):
        self.log.add(self.name)
        self.docs.append(copy.deepcopy(d))

    def update_one(self, flt, upd, upsert=False):
        self.log.add(self.name)
        for d in self.docs:
            if all(d.get(k) == v for k, v in flt.items()):
                d.update(copy.deepcopy(upd["$set"]))
                return
        if upsert:
            self.docs.append({**flt, **copy.deepcopy(upd["$set"])})


class FakeDb:
    def __init__(self):
        self.log, self.colls = set(), {}

    def __getitem__(self, name):
        return self.colls.setdefault(name, FakeColl(name, self.log))


def man(**over):
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m.update(over)
    return m


def doc(v, status, manifest=None, **kw):
    return {"version": v, "parent": v - 1 or None, "status": status, "manifest": manifest or man(),
            "reason": "r", "author": "a", "scores": {}, **kw}


def dbs(src, tgt):
    s, t = FakeDb(), FakeDb()
    s["harness_manifests"].docs = [copy.deepcopy(d) for d in src]
    t["harness_manifests"].docs = [copy.deepcopy(d) for d in tgt]
    return s, t


def alt(n=1):
    m = man()
    m["rules"] = [{"id": f"r{i}", "text": f"rule {i}"} for i in range(n)]
    return m


SRC = [doc(1, "retired"), doc(2, "candidate", alt()), doc(3, "active", alt(2))]
TGT = [doc(1, "active")]


def run_main(s, t, extra, capsys):
    rc = atlas_sync.main(ARGS + extra, source=s, target=t, env_path=Path("/nonexistent/.env"))
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def statuses(t):
    return {d["version"]: d["status"] for d in t["harness_manifests"].docs}


def test_dry_run_writes_nothing(capsys):
    s, t = dbs(SRC, TGT)
    rc, out, _ = run_main(s, t, [], capsys)
    assert rc == 0 and "dry run" in out and "insert" in out and "retire" in out
    assert statuses(t) == {1: "active"} and "audit_log" not in t.colls


def test_apply_missing_flags_refused(capsys):
    for drop in ("--yes", "--allow-remote", "--by"):
        s, t = dbs(SRC, TGT)
        args = list(APPLY)
        i = args.index(drop)
        del args[i:i + (2 if drop == "--by" else 1)]
        rc, out, _ = run_main(s, t, args, capsys)
        assert rc == 2 and "refused" in out
        assert statuses(t) == {1: "active"}
        assert t["audit_log"].docs[-1]["outcome"].startswith("refused")


def test_apply_ok_single_active_audit_and_verification(capsys):
    s, t = dbs(SRC, TGT)
    rc, out, _ = run_main(s, t, APPLY, capsys)
    assert rc == 0 and "verified" in out
    assert statuses(t) == {1: "retired", 2: "candidate", 3: "active"}
    row = t["audit_log"].docs[-1]
    assert row["action"] == "atlas_sync.manifests" and row["actor"] == "ann" and row["outcome"] == "ok"
    assert t["harness_manifests"].docs[0]["manifest"] == TGT[0]["manifest"]  # v1 only retired


def test_idempotent_second_run(capsys):
    s, t = dbs(SRC, TGT)
    run_main(s, t, APPLY, capsys)
    before = copy.deepcopy(t["harness_manifests"].docs)
    rc, out, _ = run_main(s, t, APPLY, capsys)
    assert rc == 0 and "no changes" in out and t["harness_manifests"].docs == before


def test_unknown_target_version_refused(capsys):
    s, t = dbs(SRC, TGT + [doc(9, "candidate")])
    rc, out, _ = run_main(s, t, APPLY, capsys)
    assert rc == 2 and "unknown history" in out and 3 not in statuses(t)


def test_conflicting_version_refused_with_diff(capsys):
    s, t = dbs(SRC, TGT + [doc(2, "candidate", man(thresholds={"x": 1}))])
    rc, out, _ = run_main(s, t, APPLY, capsys)
    assert rc == 2 and "conflict at v2" in out and "thresholds" in out and "rules" in out
    assert statuses(t) == {1: "active", 2: "candidate"}


def test_invalid_manifest_refused(capsys):
    bad = man()
    bad["smuggled"] = 1
    s, t = dbs([doc(1, "retired"), doc(2, "active", bad)], TGT)
    rc, out, _ = run_main(s, t, APPLY + ["--force"], capsys)
    assert rc == 2 and "invalid" in out and statuses(t) == {1: "active"}


def test_v1_retired_only_when_source_active_above_one():
    p = plan([doc(1, "active"), doc(2, "candidate", alt())], TGT)
    assert not p["refusals"] and p["final_active"] == 1
    assert [r["action"] for r in p["actions"]] == ["unchanged", "insert"]
    p = plan(SRC, TGT)
    assert [r["action"] for r in p["actions"]] == ["retire", "insert", "insert"]


def test_write_order_never_two_active():
    src = SRC + [doc(4, "candidate", alt(3))]
    p = plan(src, [doc(1, "active")])
    order = [(r["version"], r["action"]) for r in p["write_order"]]
    assert order[0] == (1, "retire") and order[-1] == (3, "insert")
    # replay: never more than one active at any step
    state = {1: "active"}
    for r in p["write_order"]:
        state[r["version"]] = "retired" if r["action"] == "retire" else r["doc"]["status"]
        assert sum(x == "active" for x in state.values()) <= 1


def test_plan_end_state_one_active_even_with_stale_target_active(capsys):
    s, t = dbs(SRC, [doc(1, "active"), doc(2, "active", alt())])
    rc, _, _ = run_main(s, t, APPLY, capsys)
    assert rc == 0 and sorted(v for v, st in statuses(t).items() if st == "active") == [3]


def test_no_other_collection_touched(capsys):
    s, t = dbs(SRC, TGT)
    s["insurance_claims"].docs = [{"claim_id": "C"}]
    t["insurance_claims"].docs = [{"claim_id": "T"}]
    s.log.clear(), t.log.clear()
    run_main(s, t, APPLY, capsys)
    assert s.log == {"harness_manifests"}
    assert t.log == {"harness_manifests", "audit_log"}
    assert t["insurance_claims"].docs == [{"claim_id": "T"}]


def test_uri_never_in_output_logs_or_exceptions(capsys, caplog):
    class Boom(FakeDb):
        def __getitem__(self, name):
            raise RuntimeError(f"cannot connect to {URI} user atlasuser pw {PASSWORD}")

    s, t = dbs(SRC, TGT)
    with caplog.at_level(logging.DEBUG):
        run_main(s, t, [], capsys)
        rc, out, err = run_main(s, t, APPLY, capsys)
        rc2, out2, err2 = run_main(s, Boom(), APPLY, capsys)
        with pytest.raises(atlas_sync.SyncError) as ei:
            atlas_sync.run(s, Boom(), to_uri=URI, from_uri="mongodb://localhost:1/")
    tb = "".join(traceback.format_exception(ei.value))
    assert rc2 == 1
    blob = out + err + out2 + err2 + caplog.text + str(ei.value) + tb
    assert PASSWORD not in blob and "atlasuser" not in blob and URI not in blob
    assert "cluster0.abcde.mongodb.net" in out
    audit_text = str(t["audit_log"].docs)
    assert PASSWORD not in audit_text and "atlasuser" not in audit_text
    assert PASSWORD not in mask_uri(URI) and "cluster0.abcde.mongodb.net" in mask_uri(URI)


def test_cli_routes_and_missing_target_config(capsys, monkeypatch):
    monkeypatch.delenv("ATLAS_EVOLVE_URI", raising=False)
    monkeypatch.delenv("ATLAS_EVOLVE_DB", raising=False)
    monkeypatch.setattr(atlas_sync, "ENV_FILE", Path("/nonexistent/.env"))
    monkeypatch.setattr(atlas_sync, "_load_env_names", lambda path=None: {})
    assert cli.main(["atlas-sync"]) == 2
    assert "not configured" in capsys.readouterr().err


def test_verification_failure_is_loud(capsys, monkeypatch):
    s, t = dbs(SRC, TGT)
    monkeypatch.setattr(atlas_sync, "_apply", lambda target, p: None)
    rc, out, err = run_main(s, t, APPLY, capsys)
    assert rc == 1 and "VERIFICATION FAILED" in err
