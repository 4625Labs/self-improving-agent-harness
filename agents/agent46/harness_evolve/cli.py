"""uv run python -m harness_evolve.cli [--uri U] [--db D] [--proposer heuristic|llm] cycle|status|approve V|rollback V"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_agent46.harness_config import HarnessConfigStore  # noqa: E402

from .loop import approve, rollback, run_cycle  # noqa: E402
from .proposer import CompositeProposer, HeuristicProposer, LLMProposer, ToolHeuristicProposer  # noqa: E402

# Local Docker Mongo port is assigned per machine: check `agentengine dev status` and override
# with --uri or EVOLVE_MONGODB_URI. directConnection=true is required from the host.
DEFAULT_URI = "mongodb://localhost:53087/?directConnection=true"
DEFAULT_DB = "agent_agent46_local"


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
    v.add_argument("--by", required=True)
    d = sub.add_parser("dismiss")
    d.add_argument("case_id")
    d.add_argument("--by", required=True)
    d.add_argument("--note", default="")
    ls = sub.add_parser("list")
    ls.add_argument("--status")
    c = cp.parse_args(rest)
    court = CourtDocket(MongoClient(a.uri, serverSelectionTimeoutMS=2000)[a.db])
    if c.action == "nominate":
        out = court.nominate(c.claim_id, "manual", c.reason)
    elif c.action == "scan":
        out = court.nominate_reviewer_disagreements()
    elif c.action == "file":
        out = court.file_case(c.case_id)
    elif c.action == "verdict":
        out = court.verdict(c.case_id, c.expected, c.critical, c.by)
    elif c.action == "dismiss":
        out = court.dismiss(c.case_id, c.by, c.note)
    else:
        out = court.list_cases(c.status)
    print(json.dumps(out, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="harness_evolve")
    p.add_argument("--uri", default=os.environ.get("EVOLVE_MONGODB_URI") or os.environ.get("MONGODB_URI") or DEFAULT_URI)
    p.add_argument("--db", default=os.environ.get("MONGODB_DATABASE", DEFAULT_DB))
    p.add_argument("--proposer", choices=["heuristic", "llm"], default="heuristic")
    p.add_argument("--surface", choices=["risk", "tools", "all"], default="all",
                   help="which harness surface a cycle may change (both are always evaluated and gated)")
    p.add_argument("--dry-run", action="store_true",
                   help="with `cycle --proposer llm`: print the prompt and a token estimate, make no call, exit")
    p.add_argument("--max-tokens", type=int, default=600, help="LLM output cap (default 600)")
    p.add_argument("--max-failures", type=int, default=8, help="max failing scenarios in the LLM prompt (default 8)")
    p.add_argument("--auto-activate-low-risk", action="store_true")
    p.add_argument("command", choices=["cycle", "status", "approve", "rollback", "court"])
    p.add_argument("version", nargs="?", type=int)
    a, rest = p.parse_known_args(argv)
    if a.command == "court":
        return _court_main(a, rest)
    if rest:
        p.error(f"unrecognized arguments: {' '.join(rest)}")
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
    store = HarnessConfigStore(a.uri, a.db)
    store.seed_if_empty()
    if a.command == "cycle":
        if a.proposer == "llm":
            proposer = LLMProposer(max_tokens=a.max_tokens, max_failures=a.max_failures)
        elif a.surface == "risk":
            proposer = HeuristicProposer()
        elif a.surface == "tools":
            proposer = ToolHeuristicProposer()
        else:
            proposer = CompositeProposer([HeuristicProposer(), ToolHeuristicProposer()])
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
        (approve if a.command == "approve" else rollback)(store, a.version)
        out = {"ok": True, "active": store.get_active()["version"]}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
