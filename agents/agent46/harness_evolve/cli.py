"""uv run python -m harness_evolve.cli [--uri U] [--db D] [--proposer heuristic|llm] cycle|status|approve V --yes|rollback V --yes|reject V --yes|court ...|budget|audit [--n N]

Mutating commands need --allow-remote (or EVOLVE_ALLOW_REMOTE=1) for a non-local Mongo host; destructive ones
(approve, rollback, reject, court verdict, court dismiss) also need --yes. Every attempt is written to audit_log."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_agent46.harness_config import HarnessConfigStore  # noqa: E402

from .audit import ConfirmationRequired, RemoteRefused, check_remote, os_user, recent  # noqa: E402
from .budget import DailyBudget  # noqa: E402
from .loop import approve, reject, rollback, run_cycle  # noqa: E402
from .proposer import DEFAULT_CACHE_DIR, CompositeProposer, HeuristicProposer, LimitsHeuristicProposer, LLMProposer, ToolHeuristicProposer  # noqa: E402

# Local Docker Mongo port is assigned per machine: check `agentengine dev status` and override
# with --uri or EVOLVE_MONGODB_URI. directConnection=true is required from the host.
DEFAULT_URI = "mongodb://localhost:53087/?directConnection=true"
DEFAULT_DB = "agent_agent46_local"


def _load_events(path: str) -> list:
    text = Path(path).read_text()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except ValueError:
        out = []
        for line in text.splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    out.append(None)  # counted as skipped
        return out


def _ingest_events(a, c, by) -> int:
    """Advisory: only nominates (source raindrop_signal); a human still decides every verdict."""
    from pymongo import MongoClient

    from .audit import record
    from .court import CourtDocket
    from .raindrop_source import SOURCE, analyse_events

    res = analyse_events(_load_events(c.file), min_count=c.min_count)
    print(json.dumps(res, indent=2, default=str))
    if c.dry_run or a.dry_run:
        return 0
    check_remote(a.uri, a.allow_remote)
    db = MongoClient(a.uri, serverSelectionTimeoutMS=2000)[a.db]
    court, nominated = CourtDocket(db), 0
    detail = {"candidates": len(res["candidates"]), "skipped": res["skipped"], "min_count": c.min_count}
    try:
        for cand in res["candidates"]:
            court.nominate(cand["claim_id"], SOURCE, cand["reason"])
            nominated += 1
    except Exception as e:
        record(db, "court.ingest", by, c.file, detail | {"nominated": nominated}, f"error: {e}")
        raise
    record(db, "court.ingest", by, c.file, detail | {"nominated": nominated}, "ok")
    return 0


def _court_main(a, rest) -> int:
    from pymongo import MongoClient

    from .court import CourtDocket

    cp = argparse.ArgumentParser(prog="harness_evolve court")
    sub = cp.add_subparsers(dest="action", required=True)
    n = sub.add_parser("nominate")
    n.add_argument("claim_id")
    n.add_argument("--reason", default="")
    sub.add_parser("scan")
    f = sub.add_parser("file")
    f.add_argument("case_id")
    v = sub.add_parser("verdict")
    v.add_argument("case_id")
    v.add_argument("expected")
    v.add_argument("--critical", action="store_true")
    v.add_argument("--by")
    d = sub.add_parser("dismiss")
    d.add_argument("case_id")
    d.add_argument("--by")
    d.add_argument("--note", default="")
    o = sub.add_parser("overturn")
    o.add_argument("case_id")
    o.add_argument("expected")
    o.add_argument("--critical", action="store_true")
    o.add_argument("--by")
    o.add_argument("--reason", default="")
    r = sub.add_parser("retract")
    r.add_argument("case_id")
    r.add_argument("--by")
    r.add_argument("--reason", default="")
    ls = sub.add_parser("list")
    ls.add_argument("--status")
    ie = sub.add_parser("ingest-events")
    ie.add_argument("file")
    ie.add_argument("--min-count", type=int, default=1)
    ie.add_argument("--dry-run", action="store_true")
    c = cp.parse_args(rest)
    by = getattr(c, "by", None) or a.by or os_user()
    if c.action in ("verdict", "dismiss", "overturn", "retract") and not (getattr(c, "by", None) or a.by):
        print(f"note: --by not given; recording actor as OS user '{by}'", file=sys.stderr)
    if c.action == "ingest-events":
        return _ingest_events(a, c, by)
    if c.action != "list":
        check_remote(a.uri, a.allow_remote)
    court = CourtDocket(MongoClient(a.uri, serverSelectionTimeoutMS=2000)[a.db])
    if c.action == "nominate":
        out = court.nominate(c.claim_id, "manual", c.reason)
    elif c.action == "scan":
        out = court.nominate_reviewer_disagreements()
    elif c.action == "file":
        out = court.file_case(c.case_id)
    elif c.action == "verdict":
        out = court.verdict(c.case_id, c.expected, c.critical, by, confirmed=a.yes)
    elif c.action == "dismiss":
        out = court.dismiss(c.case_id, by, c.note, confirmed=a.yes)
    elif c.action == "overturn":
        out = court.overturn(c.case_id, c.expected, c.critical, by, c.reason, confirmed=a.yes)
    elif c.action == "retract":
        out = court.retract(c.case_id, by, c.reason, confirmed=a.yes)
    else:
        out = court.list_cases(c.status)
    print(json.dumps(out, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="harness_evolve")
    p.add_argument("--uri", default=os.environ.get("EVOLVE_MONGODB_URI") or os.environ.get("MONGODB_URI") or DEFAULT_URI)
    p.add_argument("--db", default=os.environ.get("MONGODB_DATABASE", DEFAULT_DB))
    p.add_argument("--proposer", choices=["heuristic", "llm"], default="heuristic")
    p.add_argument("--surface", choices=["risk", "tools", "limits", "all"], default="all",
                   help="which harness surface a cycle may change (both are always evaluated and gated)")
    p.add_argument("--dry-run", action="store_true",
                   help="with `cycle --proposer llm`: print the prompt and a token estimate, make no call, exit")
    p.add_argument("--max-tokens", type=int, default=600, help="LLM output cap (default 600)")
    p.add_argument("--max-failures", type=int, default=8, help="max failing scenarios in the LLM prompt (default 8)")
    p.add_argument("--auto-activate-low-risk", action="store_true")
    p.add_argument("--yes", action="store_true",
                   help="confirm a destructive action (approve, rollback, reject, court verdict/dismiss)")
    p.add_argument("--by", help="actor recorded in the audit log (default: OS user)")
    p.add_argument("--allow-remote", action="store_true",
                   help="allow mutating commands against a non-local Mongo host (or set EVOLVE_ALLOW_REMOTE=1)")
    p.add_argument("--n", type=int, default=20, help="rows for the `audit` command (default 20)")
    p.add_argument("command", choices=["cycle", "status", "approve", "rollback", "reject", "court", "budget", "audit"])
    p.add_argument("version", nargs="?")  # int for approve/rollback/reject; the court action word for `court`
    a, rest = p.parse_known_args(argv)
    try:
        return _run(p, a, rest)
    except (ConfirmationRequired, RemoteRefused) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


def _run(p, a, rest) -> int:
    if a.command == "court":
        return _court_main(a, ([a.version] if a.version else []) + rest)
    if a.version is not None:
        try:
            a.version = int(a.version)
        except ValueError:
            p.error(f"invalid version: {a.version!r}")
    if rest:
        p.error(f"unrecognized arguments: {' '.join(rest)}")
    if a.command == "budget":  # before the store is reached: no Mongo needed
        try:
            out = DailyBudget(DEFAULT_CACHE_DIR).remaining()
        except ValueError as e:
            p.error(str(e))
        print(json.dumps(out, indent=2))
        return 0
    if a.dry_run:
        if a.command != "cycle" or a.proposer != "llm":
            p.error("--dry-run only applies to `cycle --proposer llm`")
        from agent_agent46.harness_config import DEFAULT_MANIFEST
        from evals.run_eval import evaluate
        from evals.tool_eval import evaluate_tools

        proposer = LLMProposer(max_tokens=a.max_tokens, max_failures=a.max_failures)
        risk, tools = evaluate(DEFAULT_MANIFEST, "dev"), evaluate_tools(DEFAULT_MANIFEST, "dev")
        info = proposer.dry_run(DEFAULT_MANIFEST, risk["failures"], {k: v for k, v in risk.items() if k != "failures"},
                                tool_failures=tools["failures"])
        print(info.pop("prompt"))
        print(json.dumps(info, indent=2))
        return 0
    if a.command == "audit":  # read-only
        from pymongo import MongoClient

        print(json.dumps(recent(MongoClient(a.uri, serverSelectionTimeoutMS=2000)[a.db], a.n), indent=2, default=str))
        return 0
    if a.command in ("cycle", "approve", "rollback", "reject"):
        check_remote(a.uri, a.allow_remote)
    store = HarnessConfigStore(a.uri, a.db)
    store.seed_if_empty()
    if a.command == "cycle":
        if a.proposer == "llm":
            proposer = LLMProposer(max_tokens=a.max_tokens, max_failures=a.max_failures)
        elif a.surface == "risk":
            proposer = HeuristicProposer()
        elif a.surface == "tools":
            proposer = ToolHeuristicProposer()
        elif a.surface == "limits":
            proposer = LimitsHeuristicProposer()
        else:
            proposer = CompositeProposer([HeuristicProposer(), ToolHeuristicProposer(), LimitsHeuristicProposer()])
        out = run_cycle(store, proposer, auto_activate_low_risk=a.auto_activate_low_risk, surface=a.surface)
    elif a.command == "status":
        out = []
        for d in store.list_versions():
            sc = d.get("scores") or {}
            out.append({k: d[k] for k in ("version", "parent", "status", "author", "reason")} | {
                "gate": (sc.get("gate") or {}).get("gate"),
                "risk_gate": (sc.get("risk_gate") or sc.get("gate") or {}).get("gate"),
                "tools_gate": (sc.get("tools_gate") or {}).get("gate"),
            })
    else:
        if a.version is None:
            p.error(f"{a.command} needs a version")
        fn = {"approve": approve, "rollback": rollback, "reject": reject}[a.command]
        fn(store, a.version, confirmed=a.yes, actor=a.by or os_user())
        out = {"ok": True, "active": store.get_active()["version"]}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
