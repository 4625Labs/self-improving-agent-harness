"""One self-improvement cycle: propose -> patch -> store candidate -> evaluate -> gate."""

import copy
from typing import Any

from agent_agent46.harness_config import HarnessConfigStore
from evals.run_eval import compare, evaluate_all
from evals.limits_eval import compare_limits, evaluate_limits_all
from evals.tool_eval import compare_tools, evaluate_tools_all

from .audit import run_confirmed
from .patch import PatchError, apply_patch, classify_risk
from .proposer import LIMIT_OPS, RISK_OPS, SURFACES, TOOL_OPS


def _summary(res: dict) -> dict:
    out = {k: v for k, v in res.items() if k != "failures"}
    if res.get("split") == "heldout":
        return out  # held-out ids must never leave the evaluator: counts and metrics only
    return out | {"failed_ids": [f["id"] for f in res["failures"]]}


def record_scores(store: HarnessConfigStore, version: int, scores: dict[str, Any]) -> None:
    store._collection.update_one({"version": version}, {"$set": {"scores": scores}})


def _collect_proposals(proposer, surface: str, manifest: dict, risk_dev: dict, tools_dev: dict,
                       limits_dev: dict | None = None) -> dict:
    """Call each member proposer with DEV data of its own surface only (never heldout).

    A proposer with no `surface` is treated as a risk proposer. Ops outside the requested
    surface are dropped, so `--surface tools` can never change the risk settings.
    """
    allowed = {"risk": RISK_OPS, "tools": TOOL_OPS, "limits": LIMIT_OPS, "all": RISK_OPS + TOOL_OPS + LIMIT_OPS}[surface]
    limits_dev = limits_dev or {"failures": []}
    ops: list[dict[str, Any]] = []
    notes: list[str] = []
    for m in getattr(proposer, "members", [proposer]):
        ms = getattr(m, "surface", "risk")
        if surface != "all" and ms not in (surface, "all"):
            continue
        if ms == "tools":
            out = m.propose(copy.deepcopy(manifest), copy.deepcopy(tools_dev["failures"]), _summary(tools_dev))
        elif ms == "limits":
            out = m.propose(copy.deepcopy(manifest), copy.deepcopy(limits_dev["failures"]), _summary(limits_dev))
        elif ms == "all":
            out = m.propose(copy.deepcopy(manifest), copy.deepcopy(risk_dev["failures"]), _summary(risk_dev),
                            tool_failures=copy.deepcopy(tools_dev["failures"]),
                            limits_failures=copy.deepcopy(limits_dev["failures"]))
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


def combine_gates(risk_gate: dict, tools_gate: dict, limits_gate: dict | None = None) -> dict:
    """The combined gate passes only if EVERY surface gate passes."""
    reasons = [f"risk: {r}" for r in risk_gate["reasons"]] + [f"tools: {r}" for r in tools_gate["reasons"]]
    passed = bool(risk_gate["passed"] and tools_gate["passed"])
    out = {"risk": risk_gate["gate"], "tools": tools_gate["gate"]}
    if limits_gate is not None:
        reasons += [f"limits: {r}" for r in limits_gate["reasons"]]
        passed = passed and bool(limits_gate["passed"])
        out["limits"] = limits_gate["gate"]
    return {"gate": "pass" if passed else "fail", "passed": passed, "reasons": reasons, **out}


def run_cycle(store: HarnessConfigStore, proposer, auto_activate_low_risk: bool = False, surface: str = "all") -> dict:
    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}")
    active = store.get_active()
    if active is None:
        raise RuntimeError("no active manifest; seed the store first")
    base_manifest, parent = active["manifest"], active["version"]  # may lack tool_access: means baseline
    baseline = evaluate_all(base_manifest)
    tools_baseline = evaluate_tools_all(base_manifest)
    limits_baseline = evaluate_limits_all(base_manifest)  # may lack limits: means baseline
    proposal = _collect_proposals(proposer, surface, base_manifest, baseline["dev"], tools_baseline["dev"],
                                  limits_baseline["dev"])
    ops, rationale = proposal["ops"], proposal["rationale"]
    report: dict[str, Any] = {
        "parent": parent, "proposer": proposer.name, "surface": surface, "ops": ops, "rationale": rationale,
        "baseline": {s: _summary(r) for s, r in baseline.items()},
        "tools_baseline": {s: _summary(r) for s, r in tools_baseline.items()},
        "limits_baseline": {s: _summary(r) for s, r in limits_baseline.items()},
    }
    if not ops:
        return report | {"version": None, "status": "no_change", "gate": None, "risk": None}

    try:
        candidate = apply_patch(base_manifest, ops)
    except PatchError as e:
        # Invariant-breaking or malformed patches never reach the store.
        return report | {"version": None, "status": "rejected", "risk": None, "error": str(e),
                         "gate": {"gate": "fail", "passed": False, "reasons": [f"patch rejected: {e}"]}}
    risk = classify_risk(ops, base_manifest)  # raising a limit is never low risk
    version = store.propose(candidate, parent=parent, reason=rationale, author=f"proposer:{proposer.name}")
    result = evaluate_all(candidate)
    tools_result = evaluate_tools_all(candidate)
    limits_result = evaluate_limits_all(candidate)
    # A surface the patch did not touch cannot have changed, so its gate is not re-litigated:
    # compare() fails on ANY critical violation, including ones the baseline already has, which
    # would block every patch to the other surface. Both surfaces are still evaluated and stored.
    risk_touched = any(o["op"] in RISK_OPS for o in ops)
    tools_touched = any(o["op"] in TOOL_OPS for o in ops)
    limits_touched = any(o["op"] in LIMIT_OPS for o in ops)
    risk_gate = compare(baseline, result) if risk_touched else _untouched("risk", baseline, result)
    tools_gate = compare_tools(tools_baseline, tools_result) if tools_touched else _untouched("tools", tools_baseline, tools_result)
    limits_gate = (compare_limits(limits_baseline, limits_result) if limits_touched
                   else _untouched("limits", limits_baseline, limits_result))
    gate = combine_gates(risk_gate, tools_gate, limits_gate)
    record_scores(store, version, {
        "baseline": {s: _summary(r) for s, r in baseline.items()},
        "candidate": {s: _summary(r) for s, r in result.items()},
        "tools_baseline": {s: _summary(r) for s, r in tools_baseline.items()},
        "tools_candidate": {s: _summary(r) for s, r in tools_result.items()},
        "limits_baseline": {s: _summary(r) for s, r in limits_baseline.items()},
        "limits_candidate": {s: _summary(r) for s, r in limits_result.items()},
        "risk_gate": risk_gate, "tools_gate": tools_gate, "limits_gate": limits_gate,
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
        "risk_gate": risk_gate, "tools_gate": tools_gate, "limits_gate": limits_gate,
        "candidate": {s: _summary(r) for s, r in result.items()},
        "tools_candidate": {s: _summary(r) for s, r in tools_result.items()},
        "limits_candidate": {s: _summary(r) for s, r in limits_result.items()},
    }


def _audit_db(store, audit_db):
    if audit_db is not None:
        return audit_db
    db = getattr(store, "audit_db", None)
    return db if db is not None else getattr(getattr(store, "_collection", None), "database", None)


def approve(store: HarnessConfigStore, version: int, *, confirmed: bool = False, actor: str = "unknown",
            audit_db=None) -> None:
    """Human approval: activate a candidate whose gate passed. Needs confirmed=True; always audited."""
    def go() -> None:
        doc = store.get(version)
        if not doc:
            raise ValueError(f"no version {version}")
        if doc["status"] != "candidate":
            raise ValueError(f"version {version} is {doc['status']}, not a candidate")
        if not (doc.get("scores", {}).get("gate") or {}).get("passed"):
            raise ValueError(f"version {version} has no passing gate result")
        store.activate(version)
    run_confirmed(_audit_db(store, audit_db), "manifest.approve", actor, version, {}, confirmed, go)


def rollback(store: HarnessConfigStore, to_version: int, *, confirmed: bool = False, actor: str = "unknown",
             audit_db=None) -> None:
    """Re-activate an older version (retires the current one). Needs confirmed=True; always audited."""
    def go() -> None:
        doc = store.get(to_version)
        if not doc:
            raise ValueError(f"no version {to_version}")
        if doc["status"] == "rejected":
            raise ValueError(f"version {to_version} was rejected")
        store.activate(to_version)
    run_confirmed(_audit_db(store, audit_db), "manifest.rollback", actor, to_version, {}, confirmed, go)


def reject(store: HarnessConfigStore, version: int, *, confirmed: bool = False, actor: str = "unknown",
           audit_db=None) -> None:
    """Human rejection of a candidate. Needs confirmed=True; always audited."""
    def go() -> None:
        doc = store.get(version)
        if not doc or doc["status"] != "candidate":
            raise ValueError(f"version {version} is not a candidate")
        store.set_status(version, "rejected")
    run_confirmed(_audit_db(store, audit_db), "manifest.reject", actor, version, {}, confirmed, go)
