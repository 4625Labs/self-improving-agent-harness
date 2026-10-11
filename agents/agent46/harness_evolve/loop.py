"""One self-improvement cycle: propose -> patch -> store candidate -> evaluate -> gate."""

import copy
from typing import Any

from agent_agent46.harness_config import HarnessConfigStore
from evals.run_eval import compare, evaluate_all
from evals.tool_eval import compare_tools, evaluate_tools_all

from .patch import PatchError, apply_patch, classify_risk
from .proposer import RISK_OPS, SURFACES, TOOL_OPS


def _summary(res: dict) -> dict:
    out = {k: v for k, v in res.items() if k != "failures"}
    if res.get("split") == "heldout":
        return out  # held-out ids must never leave the evaluator: counts and metrics only
    return out | {"failed_ids": [f["id"] for f in res["failures"]]}


def record_scores(store: HarnessConfigStore, version: int, scores: dict[str, Any]) -> None:
    store._collection.update_one({"version": version}, {"$set": {"scores": scores}})


def _collect_proposals(proposer, surface: str, manifest: dict, risk_dev: dict, tools_dev: dict) -> dict:
    """Call each member proposer with DEV data of its own surface only (never heldout).

    A proposer with no `surface` is treated as a risk proposer. Ops outside the requested
    surface are dropped, so `--surface tools` can never change the risk settings.
    """
    allowed = {"risk": RISK_OPS, "tools": TOOL_OPS, "all": RISK_OPS + TOOL_OPS}[surface]
    ops: list[dict[str, Any]] = []
    notes: list[str] = []
    for m in getattr(proposer, "members", [proposer]):
        ms = getattr(m, "surface", "risk")
        if surface != "all" and ms not in (surface, "all"):
            continue
        if ms == "tools":
            out = m.propose(copy.deepcopy(manifest), copy.deepcopy(tools_dev["failures"]), _summary(tools_dev))
        elif ms == "all":
            out = m.propose(copy.deepcopy(manifest), copy.deepcopy(risk_dev["failures"]), _summary(risk_dev),
                            tool_failures=copy.deepcopy(tools_dev["failures"]))
        else:
            out = m.propose(copy.deepcopy(manifest), copy.deepcopy(risk_dev["failures"]), _summary(risk_dev))
        ops += [o for o in out["ops"] if o.get("op") in allowed or surface == "all"]
        if out.get("rationale"):
            notes.append(out["rationale"])
    return {"ops": ops, "rationale": "; ".join(notes)}


def _untouched(surface: str, baseline: dict, result: dict) -> dict:
    same = all(baseline[s]["failures"] == result[s]["failures"] and baseline[s]["accuracy"] == result[s]["accuracy"]
               for s in baseline)
    return {"gate": "pass" if same else "fail", "passed": same, "deltas": {}, "untouched": True,
            "reasons": [] if same else [f"{surface} surface not patched but its results changed"]}


def combine_gates(risk_gate: dict, tools_gate: dict) -> dict:
    """The combined gate passes only if BOTH surface gates pass."""
    reasons = [f"risk: {r}" for r in risk_gate["reasons"]] + [f"tools: {r}" for r in tools_gate["reasons"]]
    passed = bool(risk_gate["passed"] and tools_gate["passed"])
    return {"gate": "pass" if passed else "fail", "passed": passed, "reasons": reasons,
            "risk": risk_gate["gate"], "tools": tools_gate["gate"]}


def run_cycle(store: HarnessConfigStore, proposer, auto_activate_low_risk: bool = False, surface: str = "all") -> dict:
    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}")
    active = store.get_active()
    if active is None:
        raise RuntimeError("no active manifest; seed the store first")
    base_manifest, parent = active["manifest"], active["version"]  # may lack tool_access: means baseline
    baseline = evaluate_all(base_manifest)
    tools_baseline = evaluate_tools_all(base_manifest)
    proposal = _collect_proposals(proposer, surface, base_manifest, baseline["dev"], tools_baseline["dev"])
    ops, rationale = proposal["ops"], proposal["rationale"]
    report: dict[str, Any] = {
        "parent": parent, "proposer": proposer.name, "surface": surface, "ops": ops, "rationale": rationale,
        "baseline": {s: _summary(r) for s, r in baseline.items()},
        "tools_baseline": {s: _summary(r) for s, r in tools_baseline.items()},
    }
    if not ops:
        return report | {"version": None, "status": "no_change", "gate": None, "risk": None}

    try:
        candidate = apply_patch(base_manifest, ops)
    except PatchError as e:
        # Invariant-breaking or malformed patches never reach the store.
        return report | {"version": None, "status": "rejected", "risk": None, "error": str(e),
                         "gate": {"gate": "fail", "passed": False, "reasons": [f"patch rejected: {e}"]}}
    risk = classify_risk(ops)
    version = store.propose(candidate, parent=parent, reason=rationale, author=f"proposer:{proposer.name}")
    result = evaluate_all(candidate)
    tools_result = evaluate_tools_all(candidate)
    # A surface the patch did not touch cannot have changed, so its gate is not re-litigated:
    # compare() fails on ANY critical violation, including ones the baseline already has, which
    # would block every patch to the other surface. Both surfaces are still evaluated and stored.
    risk_touched = any(o["op"] in RISK_OPS for o in ops)
    tools_touched = any(o["op"] in TOOL_OPS for o in ops)
    risk_gate = compare(baseline, result) if risk_touched else _untouched("risk", baseline, result)
    tools_gate = compare_tools(tools_baseline, tools_result) if tools_touched else _untouched("tools", tools_baseline, tools_result)
    gate = combine_gates(risk_gate, tools_gate)
    record_scores(store, version, {
        "baseline": {s: _summary(r) for s, r in baseline.items()},
        "candidate": {s: _summary(r) for s, r in result.items()},
        "tools_baseline": {s: _summary(r) for s, r in tools_baseline.items()},
        "tools_candidate": {s: _summary(r) for s, r in tools_result.items()},
        "risk_gate": risk_gate, "tools_gate": tools_gate,
        "gate": gate, "risk": risk,
    })
    if not gate["passed"]:
        store.set_status(version, "rejected")
        status = "rejected"
    elif auto_activate_low_risk and risk == "low":
        store.activate(version)
        status = "active"
    else:
        status = "candidate"
    return report | {
        "version": version, "status": status, "risk": risk, "gate": gate,
        "risk_gate": risk_gate, "tools_gate": tools_gate,
        "candidate": {s: _summary(r) for s, r in result.items()},
        "tools_candidate": {s: _summary(r) for s, r in tools_result.items()},
    }


def approve(store: HarnessConfigStore, version: int) -> None:
    """Human approval: activate a candidate whose gate passed."""
    doc = store.get(version)
    if not doc:
        raise ValueError(f"no version {version}")
    if doc["status"] != "candidate":
        raise ValueError(f"version {version} is {doc['status']}, not a candidate")
    if not (doc.get("scores", {}).get("gate") or {}).get("passed"):
        raise ValueError(f"version {version} has no passing gate result")
    store.activate(version)


def rollback(store: HarnessConfigStore, to_version: int) -> None:
    """Re-activate an older version (retires the current one)."""
    doc = store.get(to_version)
    if not doc:
        raise ValueError(f"no version {to_version}")
    if doc["status"] == "rejected":
        raise ValueError(f"version {to_version} was rejected")
    store.activate(to_version)
