"""One self-improvement cycle: propose -> patch -> store candidate -> evaluate -> gate."""

import copy
from typing import Any

from agent_agent46.harness_config import HarnessConfigStore
from evals.run_eval import compare, evaluate_all

from .patch import apply_patch, classify_risk


def _summary(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "failures"} | {"failed_ids": [f["id"] for f in res["failures"]]}


def record_scores(store: HarnessConfigStore, version: int, scores: dict[str, Any]) -> None:
    store._collection.update_one({"version": version}, {"$set": {"scores": scores}})


def run_cycle(store: HarnessConfigStore, proposer, auto_activate_low_risk: bool = False) -> dict:
    active = store.get_active()
    if active is None:
        raise RuntimeError("no active manifest; seed the store first")
    base_manifest, parent = active["manifest"], active["version"]
    baseline = evaluate_all(base_manifest)
    dev = baseline["dev"]  # proposer sees dev only
    proposal = proposer.propose(copy.deepcopy(base_manifest), copy.deepcopy(dev["failures"]), _summary(dev))
    ops, rationale = proposal["ops"], proposal.get("rationale", "")
    report: dict[str, Any] = {
        "parent": parent, "proposer": proposer.name, "ops": ops, "rationale": rationale,
        "baseline": {s: _summary(r) for s, r in baseline.items()},
    }
    if not ops:
        return report | {"version": None, "status": "no_change", "gate": None, "risk": None}

    candidate = apply_patch(base_manifest, ops)
    risk = classify_risk(ops)
    version = store.propose(candidate, parent=parent, reason=rationale, author=f"proposer:{proposer.name}")
    result = evaluate_all(candidate)
    gate = compare(baseline, result)
    record_scores(store, version, {
        "baseline": {s: _summary(r) for s, r in baseline.items()},
        "candidate": {s: _summary(r) for s, r in result.items()},
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
        "candidate": {s: _summary(r) for s, r in result.items()},
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
