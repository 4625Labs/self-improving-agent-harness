"""Whitelisted manifest patches. The only way a proposer may change the harness."""

import copy
from typing import Any

from agent_agent46.harness_config import validate_manifest

# op name -> required fields (exactly these, no extras)
OP_SCHEMA: dict[str, tuple[str, ...]] = {
    "add_rule": ("id", "text"),
    "remove_rule": ("id",),
    "set_threshold": ("name", "value"),
    "add_keyword": ("level", "keyword"),
}
LOW_RISK_OPS = {"add_rule", "add_keyword"}


class PatchError(ValueError):
    pass


def validate_ops(ops: Any) -> list[dict[str, Any]]:
    """Strictly check shape: list of {op, <exact fields>}. Returns ops, raises PatchError."""
    if not isinstance(ops, list):
        raise PatchError("ops must be a list")
    for op in ops:
        if not isinstance(op, dict) or "op" not in op:
            raise PatchError(f"malformed op: {op!r}")
        name = op["op"]
        if name not in OP_SCHEMA:
            raise PatchError(f"unknown op {name!r}; allowed: {sorted(OP_SCHEMA)}")
        extra = set(op) - {"op", *OP_SCHEMA[name]}
        missing = set(OP_SCHEMA[name]) - set(op)
        if extra or missing:
            raise PatchError(f"op {name}: unexpected fields {sorted(extra)}, missing {sorted(missing)}")
    return ops


def apply_patch(manifest: dict[str, Any], ops: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a new manifest with ops applied; never mutates the input."""
    validate_ops(ops)
    new = copy.deepcopy(manifest)
    for op in ops:
        name = op["op"]
        if name == "add_rule":
            if any(r["id"] == op["id"] for r in new["rules"]):
                raise PatchError(f"rule id {op['id']!r} already exists")
            new["rules"].append({"id": op["id"], "text": op["text"]})
        elif name == "remove_rule":
            kept = [r for r in new["rules"] if r["id"] != op["id"]]
            if len(kept) == len(new["rules"]):
                raise PatchError(f"no rule with id {op['id']!r}")
            new["rules"] = kept
        elif name == "set_threshold":
            if op["name"] not in new["thresholds"]:
                raise PatchError(f"unknown threshold {op['name']!r}")
            new["thresholds"][op["name"]] = op["value"]
        elif name == "add_keyword":
            if op["level"] not in ("high", "medium"):
                raise PatchError("level must be 'high' or 'medium'")
            if not isinstance(op["keyword"], str) or not op["keyword"].strip():
                raise PatchError("keyword must be a non-empty string")
            kw = op["keyword"].strip().lower()
            if kw not in (k.lower() for k in new["keywords"][op["level"]]):
                new["keywords"][op["level"]].append(kw)
    errors = validate_manifest(new)
    if errors:
        raise PatchError("patched manifest violates invariants: " + "; ".join(errors))
    return new


def classify_risk(ops: list[dict[str, Any]]) -> str:
    """'low' only if every op is add_rule/add_keyword; anything else needs a human."""
    validate_ops(ops)
    return "low" if all(op["op"] in LOW_RISK_OPS for op in ops) else "needs_approval"
