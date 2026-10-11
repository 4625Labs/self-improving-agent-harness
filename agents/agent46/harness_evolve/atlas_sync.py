"""Copy the harness manifest history from a source Mongo (local dev) to a target Mongo (Atlas), safely.

Touches ONLY the `harness_manifests` collection, plus one `audit_log` row in the target. Dry-run is the
default and writes nothing. Writing needs --apply AND --yes AND --by NAME AND --allow-remote.
No URI or password is ever printed, logged, audited or put in an exception message (see mask_uri / _scrub).

    uv run python -m harness_evolve.cli atlas-sync                      # dry run (prints the plan)
    uv run python -m harness_evolve.cli atlas-sync --apply --yes --by NAME --allow-remote
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_agent46.harness_config import MANIFEST_COLLECTION, validate_manifest  # noqa: E402

from .audit import RemoteRefused, check_remote, record, uri_hosts  # noqa: E402

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
SOURCE_URI_ENV, SOURCE_DEFAULT_DB = "EVOLVE_MONGODB_URI", "agent_agent46_local"
TARGET_URI_ENV, TARGET_DB_ENV = "ATLAS_EVOLVE_URI", "ATLAS_EVOLVE_DB"
DEFAULT_SOURCE_URI = "mongodb://localhost:53087/?directConnection=true"
AUDIT_ACTION = "atlas_sync.manifests"


class SyncError(RuntimeError):
    """A sanitised, user-facing failure (never contains a URI or credentials)."""


class SyncVerificationError(SyncError):
    """After writing, the target does not look the way the plan said it would."""


def mask_uri(uri: str) -> str:
    """Scheme and host names only, e.g. 'mongodb+srv://***@host'. Never user, password or options."""
    if not uri:
        return "(unset)"
    try:
        scheme = uri.split("://", 1)[0] if "://" in uri else "mongodb"
        cred = "***@" if "@" in uri.split("://", 1)[-1].split("/", 1)[0] else ""
        return f"{scheme}://{cred}{','.join(uri_hosts(uri)) or '?'}"
    except Exception:  # noqa: BLE001
        return "(unparseable uri)"


def _secrets(*uris: str) -> list[str]:
    out: set[str] = set()
    for u in uris:
        if not u:
            continue
        out.add(u)
        try:
            rest = u.split("://", 1)[-1].split("/", 1)[0]
            if "@" in rest:
                info = rest.rsplit("@", 1)[0]
                out.update({info, unquote(info)})
                out.update(p for p in (info.split(":", 1) + unquote(info).split(":", 1)) if len(p) >= 3)
            sp = urlsplit(u)
            out.update(x for x in (sp.password, sp.username) if x and len(x) >= 3)
        except Exception:  # noqa: BLE001
            pass
    return sorted(out, key=len, reverse=True)


def _scrub(text: str, *uris: str) -> str:
    for s in _secrets(*uris):
        text = text.replace(s, "***")
    return text


def _sanitised(e: Exception, *uris: str) -> SyncError:
    return SyncError(f"{type(e).__name__}: {_scrub(str(e), *uris)}")


def _strip(doc: dict) -> dict:
    return {k: copy.deepcopy(v) for k, v in doc.items() if k != "_id"}


def _read(db) -> list[dict]:
    return sorted((_strip(d) for d in db[MANIFEST_COLLECTION].find({})), key=lambda d: d.get("version", 0))


def _diff_keys(a: dict, b: dict) -> list[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def plan(source_docs: list[dict], target_docs: list[dict], force: bool = False) -> dict:
    """Pure. Returns {actions, refusals, final_active, write_order, counts, forced}.

    actions: one row per version {version, source_status, target_status, action, doc?} with action in
    insert / replace / unchanged / retire / refuse. write_order is the safe order (retire first, non-active
    upserts next, the active version last). `force` overrides only unknown-history and conflict refusals."""
    src = {d.get("version"): d for d in source_docs}
    tgt = {d.get("version"): d for d in target_docs}
    hard: list[str] = []  # never overridable
    soft: list[str] = []  # overridable with force
    actives = sorted(v for v, d in src.items() if d.get("status") == "active")
    s_active = actives[0] if len(actives) == 1 else None
    if s_active is None:
        hard.append(f"source must have exactly one active version (found {actives or 'none'})")
    if len(src) != len(source_docs) or len(tgt) != len(target_docs):
        hard.append("duplicate version numbers found")
    if any(not isinstance(v, int) for v in list(src) + list(tgt)):
        hard.append("a manifest document has a missing or non-integer version")
    if not hard and 1 not in tgt:
        hard.append("target has no version 1 baseline (run the deployed agent once so it seeds v1)")
    if hard:
        return {"actions": [], "refusals": hard, "final_active": s_active, "write_order": [], "counts": {},
                "forced": []}

    for v in sorted(v for v in src if v >= 2):
        m = src[v].get("manifest")
        errs = validate_manifest(m) if isinstance(m, dict) else ["manifest is not an object"]
        if errs:
            hard.append(f"source v{v} manifest is invalid: {'; '.join(errs)}")
    for v in sorted(tgt):
        if v != 1 and v not in src:
            soft.append(f"target has version {v}, which is neither 1 nor in the source (unknown history)")
        elif v >= 2 and tgt[v].get("manifest") != src[v].get("manifest"):
            keys = _diff_keys(tgt[v].get("manifest") or {}, src[v].get("manifest") or {})
            soft.append(f"conflict at v{v}: target manifest differs from source in top-level keys {keys}")
    if s_active == 1 and tgt[1].get("status") != "active":
        hard.append("source active is v1 but target v1 is not active; fix the target by hand")

    actions: dict[int, dict] = {}
    for v in sorted(set(src) | set(tgt)):
        s, t = src.get(v), tgt.get(v)
        row = {"version": v, "source_status": s.get("status") if s else None,
               "target_status": t.get("status") if t else None, "action": "unchanged"}
        if v == 1:
            if s_active > 1 and t.get("status") == "active":
                row["action"] = "retire"
        elif s is None:  # target-only (unknown) version
            if not force:
                row["action"] = "refuse"
            elif t.get("status") == "active":
                row["action"] = "retire"
        else:
            if t is None:
                row["action"] = "insert"
            elif t.get("manifest") != s.get("manifest") and not force:
                row["action"] = "refuse"
            elif _strip(t) != _strip(s):
                row["action"] = "replace"
            if row["action"] in ("insert", "replace"):
                row["doc"] = _strip(s)
        actions[v] = row

    if not hard:  # end state must have exactly one active version
        final = {v: (src[v]["status"] if v >= 2 and v in src else tgt[v]["status"]) for v in set(src) | set(tgt)}
        for v, r in actions.items():
            if r["action"] == "retire":
                final[v] = "retired"
        if sorted(v for v, st in final.items() if st == "active") != [s_active]:
            hard.append(f"end state would not have exactly one active version (source active v{s_active})")
    refusals = hard + ([] if force else soft)
    writes = [r for r in actions.values() if r["action"] in ("insert", "replace", "retire")]
    order = sorted(writes, key=lambda r: (r["action"] != "retire", (r.get("doc") or {}).get("status") == "active",
                                          r["version"]))
    counts: dict[str, int] = {}
    for r in actions.values():
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    return {"actions": [actions[v] for v in sorted(actions)], "refusals": refusals, "final_active": s_active,
            "write_order": [] if refusals else order, "counts": counts, "forced": soft if force else []}


def render(p: dict, applied: bool) -> str:
    lines = ["version  source      target-before  action", "-------  ----------  -------------  ------"]
    for r in p["actions"]:
        lines.append(f"{r['version']:<7}  {str(r['source_status'] or '-'):<10}  "
                     f"{str(r['target_status'] or '-'):<13}  {r['action']}")
    c = p["counts"]
    lines.append(f"final active version: {p['final_active']}")
    lines.append("counts: " + (", ".join(f"{k}={n}" for k, n in sorted(c.items())) or "none"))
    lines += [f"REFUSED: {r}" for r in p["refusals"]] + [f"forced past: {r}" for r in p["forced"]]
    if not p["refusals"]:
        changes = sum(c.get(k, 0) for k in ("insert", "replace", "retire"))
        lines.append(("applied" if applied else "dry run (nothing written)") + ("" if changes else ": no changes"))
    return "\n".join(lines)


def _apply(target, p: dict) -> None:
    coll = target[MANIFEST_COLLECTION]
    for r in p["write_order"]:  # retire first, non-active upserts next, the active version last
        if r["action"] == "retire":
            coll.update_one({"version": r["version"]}, {"$set": {"status": "retired"}})
        else:
            coll.update_one({"version": r["version"]}, {"$set": r["doc"]}, upsert=True)


def _verify(target, p: dict) -> str:
    docs = _read(target)
    versions = [d["version"] for d in docs]
    active = [d["version"] for d in docs if d.get("status") == "active"]
    missing = {r["version"] for r in p["actions"] if r["action"] != "refuse"} - set(versions)
    if active != [p["final_active"]] or missing:
        raise SyncVerificationError(
            f"VERIFICATION FAILED: target versions {versions}, active {active}, expected active [{p['final_active']}]")
    return f"verified: versions {versions}, single active version {active[0]}"


def _reachable(db) -> bool:
    try:
        if hasattr(db, "command"):
            db.command("ping")
        return True
    except Exception:  # noqa: BLE001
        return False


def run(source, target, *, to_uri: str, apply: bool = False, yes: bool = False, by: str | None = None,
        allow_remote: bool = False, force: bool = False, from_uri: str = "", out=print) -> int:
    """Core, with injected dbs. Returns an exit code (0 ok, 2 refused); raises SyncError on failure."""
    uris = (to_uri, from_uri)
    try:
        src_docs, tgt_docs = _read(source), _read(target)
    except Exception as e:  # noqa: BLE001
        raise _sanitised(e, *uris) from None
    p = plan(src_docs, tgt_docs, force=force)
    out(f"source: {mask_uri(from_uri)}   target: {mask_uri(to_uri)}   collection: {MANIFEST_COLLECTION}")
    out(render(p, applied=False))
    detail = {"counts": p["counts"], "final_active": p["final_active"], "target_hosts": uri_hosts(to_uri)}

    def refuse(msg: str) -> int:
        out(f"refused: {msg}")
        if _reachable(target):
            record(target, AUDIT_ACTION, by or "unknown", MANIFEST_COLLECTION, detail, f"refused: {msg}")
        return 2

    if not apply:
        return 2 if p["refusals"] else 0
    missing = [f for f, ok in (("--yes", yes), ("--by NAME", bool(by)), ("--allow-remote", allow_remote)) if not ok]
    if missing:
        return refuse(f"--apply also needs {', '.join(missing)}; nothing was written")
    try:
        check_remote(to_uri, allow_remote)
    except RemoteRefused as e:
        return refuse(_scrub(str(e), *uris))
    if p["refusals"]:
        return refuse("plan has refusals (see above); nothing was written")
    try:
        _apply(target, p)
        line = _verify(target, p)
    except SyncVerificationError as e:
        record(target, AUDIT_ACTION, by, MANIFEST_COLLECTION, detail, f"error: {e}")
        raise
    except Exception as e:  # noqa: BLE001
        err = _sanitised(e, *uris)
        record(target, AUDIT_ACTION, by, MANIFEST_COLLECTION, detail, f"error: {err}")
        raise err from None
    record(target, AUDIT_ACTION, by, MANIFEST_COLLECTION, detail | {"forced": p["forced"]}, "ok")
    out(render(p, applied=True))
    out(line)
    return 0


def _load_env_names(path: Path = ENV_FILE) -> dict[str, str]:
    """Read ONLY the two Atlas names from .env; nothing is exported to os.environ."""
    try:
        from dotenv import dotenv_values

        vals = dotenv_values(path) if path.exists() else {}
    except Exception:  # noqa: BLE001
        vals = {}
    return {k: vals[k] for k in (TARGET_URI_ENV, TARGET_DB_ENV) if vals.get(k)}


def main(argv=None, source=None, target=None, env_path: Path = ENV_FILE, out=print) -> int:
    ap = argparse.ArgumentParser(prog="harness_evolve atlas-sync", description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="write (also needs --yes, --by, --allow-remote)")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--by", help="actor recorded in the audit row")
    ap.add_argument("--allow-remote", action="store_true")
    ap.add_argument("--force", action="store_true", help="override unknown-history and conflict refusals")
    ap.add_argument("--from-uri", default=os.environ.get(SOURCE_URI_ENV) or DEFAULT_SOURCE_URI)
    ap.add_argument("--from-db", default=SOURCE_DEFAULT_DB)
    ap.add_argument("--to-uri")
    ap.add_argument("--to-db")
    a = ap.parse_args(argv)
    env = _load_env_names(env_path)
    to_uri = a.to_uri or os.environ.get(TARGET_URI_ENV) or env.get(TARGET_URI_ENV) or ""
    to_db = a.to_db or os.environ.get(TARGET_DB_ENV) or env.get(TARGET_DB_ENV) or ""
    if not to_uri or not to_db:
        print(f"refused: target not configured; set {TARGET_URI_ENV} and {TARGET_DB_ENV} (agents/agent46/.env) "
              "or pass --to-uri/--to-db", file=sys.stderr)
        return 2
    uris = (to_uri, a.from_uri)
    try:
        if source is None or target is None:
            from pymongo import MongoClient

            source = source or MongoClient(a.from_uri, serverSelectionTimeoutMS=3000)[a.from_db]
            target = target or MongoClient(to_uri, serverSelectionTimeoutMS=3000)[to_db]
        return run(source, target, to_uri=to_uri, apply=a.apply, yes=a.yes, by=a.by,
                   allow_remote=a.allow_remote, force=a.force, from_uri=a.from_uri, out=out)
    except SyncError as e:
        print(f"error: {_scrub(str(e), *uris)}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"error: {_sanitised(e, *uris)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
