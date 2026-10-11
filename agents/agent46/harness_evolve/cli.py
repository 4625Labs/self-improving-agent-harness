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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="harness_evolve")
    p.add_argument("--uri", default=os.environ.get("EVOLVE_MONGODB_URI") or os.environ.get("MONGODB_URI") or DEFAULT_URI)
    p.add_argument("--db", default=os.environ.get("MONGODB_DATABASE", DEFAULT_DB))
    p.add_argument("--proposer", choices=["heuristic", "llm"], default="heuristic")
    p.add_argument("--surface", choices=["risk", "tools", "all"], default="all",
                   help="which harness surface a cycle may change (both are always evaluated and gated)")
    p.add_argument("--auto-activate-low-risk", action="store_true")
    p.add_argument("command", choices=["cycle", "status", "approve", "rollback"])
    p.add_argument("version", nargs="?", type=int)
    a = p.parse_args(argv)
    store = HarnessConfigStore(a.uri, a.db)
    store.seed_if_empty()
    if a.command == "cycle":
        if a.proposer == "llm":
            proposer = LLMProposer()
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
