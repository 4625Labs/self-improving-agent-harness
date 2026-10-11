"""
Harness manifest - the evolvable part of the agent, stored as versioned data in MongoDB.

Split of responsibilities:
- EVOLVABLE (this manifest, in Mongo): learned prompt rules, risk thresholds, risk keywords.
- FIXED (code, never in Mongo): INVARIANTS below, the claim state machine in policy_store.py,
  the human_review interrupt, tool sandboxes and network egress in agent.yaml.

A proposer (a later phase) may only write manifests. Every manifest is checked against the
invariants before it can be stored, so a bad proposal cannot be activated by accident.
"""

import copy
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

from pymongo import MongoClient

logger = logging.getLogger(__name__)

MANIFEST_COLLECTION = "harness_manifests"
BUILTIN_VERSION = 0  # Used when Mongo is unavailable or empty; never stored.
CACHE_TTL_SECONDS = 15

# Baseline manifest. This is what the agent shipped with, before any evolution.
DEFAULT_MANIFEST: dict[str, Any] = {
    "rules": [],  # [{"id": "r1", "text": "..."}] appended to the system prompt
    "thresholds": {
        "auto_approve_limit": 1000,  # Low-risk claims under this may skip human review
        "medium_amount": 5000,  # Claims at or above this classify as medium risk
        "high_amount": 15000,  # Claims at or above this classify as high risk
        "medium_score": 0.4,  # Policy risk score at or above this is medium risk
        "high_score": 0.6,  # Policy risk score at or above this is high risk
    },
    "keywords": {
        "high": ["total loss", "totaled", "stolen", "theft", "fire", "flood"],
        "medium": ["significant damage", "major repair", "structural"],
    },
}

# Hard limits a manifest must satisfy, owned by humans. Thresholds can tighten freely but
# cannot loosen past the baseline, and baseline keywords can be added to but never removed.
INVARIANTS: dict[str, Any] = {
    "max_rules": 20,
    "max_rule_chars": 300,
    "max_keywords_per_level": 25,
    "keyword_chars": (4, 40),  # min/max length; letters, digits, space, hyphen only
    "bounds": {
        "auto_approve_limit": (0, 1000),
        "medium_amount": (1000, 10000),
        "high_amount": (5000, 30000),
        "medium_score": (0.1, 0.9),
        "high_score": (0.1, 0.9),
    },
}


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return a list of invariant violations; an empty list means the manifest is valid."""
    errors: list[str] = []

    # Only these fields are evolvable. Anything else (e.g. memory label settings) is rejected so
    # a proposer cannot smuggle new behavior into the manifest.
    unknown = set(manifest) - {"rules", "thresholds", "keywords"}
    if unknown:
        errors.append(f"unknown manifest fields: {sorted(unknown)}")

    rules = manifest.get("rules")
    if not isinstance(rules, list):
        errors.append("rules must be a list")
    else:
        if len(rules) > INVARIANTS["max_rules"]:
            errors.append(f"too many rules ({len(rules)} > {INVARIANTS['max_rules']})")
        ids = [r.get("id") if isinstance(r, dict) else None for r in rules]
        if len(set(ids)) != len(ids) or None in ids:
            errors.append("every rule needs a unique id")
        for r in rules:
            text = r.get("text") if isinstance(r, dict) else None
            if not isinstance(text, str) or not text.strip():
                errors.append(f"rule {r!r} needs non-empty text")
            elif len(text) > INVARIANTS["max_rule_chars"]:
                errors.append(f"rule {r.get('id')} exceeds {INVARIANTS['max_rule_chars']} chars")

    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, dict):
        errors.append("thresholds must be an object")
    else:
        for name, (low, high) in INVARIANTS["bounds"].items():
            value = thresholds.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"thresholds.{name} must be a number")
            elif not low <= value <= high:
                errors.append(f"thresholds.{name}={value} outside allowed range [{low}, {high}]")
        if not errors:
            if thresholds["medium_amount"] >= thresholds["high_amount"]:
                errors.append("medium_amount must be below high_amount")
            if thresholds["medium_score"] >= thresholds["high_score"]:
                errors.append("medium_score must be below high_score")

    keywords = manifest.get("keywords")
    if not isinstance(keywords, dict):
        errors.append("keywords must be an object")
    else:
        for level in ("high", "medium"):
            current = keywords.get(level)
            if not isinstance(current, list) or not all(isinstance(k, str) for k in current):
                errors.append(f"keywords.{level} must be a list of strings")
                continue
            if len(current) > INVARIANTS["max_keywords_per_level"]:
                errors.append(f"keywords.{level} has too many entries ({len(current)})")
            lo, hi = INVARIANTS["keyword_chars"]
            for k in current:
                if not lo <= len(k.strip()) <= hi or not re.fullmatch(r"[a-z0-9 \-]+", k.lower()):
                    errors.append(f"keywords.{level} entry {k!r} must be {lo}-{hi} chars of a-z, 0-9, space, hyphen")
            removed = set(DEFAULT_MANIFEST["keywords"][level]) - {k.lower() for k in current}
            if removed:
                errors.append(f"keywords.{level} may not remove baseline keywords: {sorted(removed)}")

    return errors


def render_system_prompt(base_prompt: str, manifest: dict[str, Any]) -> str:
    """Apply the manifest to the base system prompt: live auto-approve limit plus learned rules."""
    limit = manifest["thresholds"]["auto_approve_limit"]
    prompt = base_prompt.replace("$1,000", f"${limit:,.0f}")
    rules = manifest.get("rules", [])
    if rules:
        lines = "\n".join(f"- {r['text']}" for r in rules)
        prompt += f"\n\n## Learned Rules\n{lines}\n"
    return prompt


class HarnessConfigStore:
    """MongoDB-backed, versioned store of harness manifests.

    Document shape: {version, parent, status, manifest, reason, author, scores,
    created_at, activated_at}. Status is one of: candidate, active, retired, rejected.
    """

    def __init__(self, mongodb_uri: str, database: str):
        self._client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=2000)
        self._collection = self._client[database][MANIFEST_COLLECTION]
        self._collection.create_index("version", unique=True)
        self._collection.create_index("status")

    def seed_if_empty(self) -> None:
        """Store the baseline as version 1 and activate it, if nothing exists yet."""
        if self._collection.count_documents({}) == 0:
            self.propose(DEFAULT_MANIFEST, parent=None, reason="baseline", author="human")
            self.activate(1)

    def get_active(self) -> Optional[dict]:
        return self._collection.find_one({"status": "active"}, {"_id": 0}, sort=[("version", -1)])

    def get(self, version: int) -> Optional[dict]:
        return self._collection.find_one({"version": version}, {"_id": 0})

    def list_versions(self) -> list[dict]:
        return list(self._collection.find({}, {"_id": 0}).sort("version", 1))

    def propose(
        self, manifest: dict[str, Any], parent: Optional[int], reason: str, author: str
    ) -> int:
        """Validate and store a manifest as a candidate. Returns its version number."""
        errors = validate_manifest(manifest)
        if errors:
            raise ValueError("manifest violates invariants: " + "; ".join(errors))
        latest = self._collection.find_one({}, {"version": 1}, sort=[("version", -1)])
        version = (latest["version"] + 1) if latest else 1
        self._collection.insert_one(
            {
                "version": version,
                "parent": parent,
                "status": "candidate",
                "manifest": copy.deepcopy(manifest),
                "reason": reason,
                "author": author,
                "scores": {},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "activated_at": None,
            }
        )
        return version

    def activate(self, version: int) -> None:
        """Make a version the active one and retire the previous active version(s)."""
        doc = self.get(version)
        if not doc:
            raise ValueError(f"No manifest version {version}")
        if validate_manifest(doc["manifest"]):
            raise ValueError(f"Version {version} fails invariants and cannot be activated")
        self._collection.update_one(
            {"version": version},
            {"$set": {"status": "active", "activated_at": datetime.now(timezone.utc).isoformat()}},
        )
        self._collection.update_many(
            {"status": "active", "version": {"$ne": version}}, {"$set": {"status": "retired"}}
        )

    def set_status(self, version: int, status: str) -> None:
        self._collection.update_one({"version": version}, {"$set": {"status": status}})


def create_config_store(mongodb_uri: str, database: str) -> Optional[HarnessConfigStore]:
    """Factory mirroring create_policy_store: None when Mongo is not configured."""
    if not mongodb_uri:
        return None
    try:
        store = HarnessConfigStore(mongodb_uri, database)
        store.seed_if_empty()
        return store
    except Exception as e:
        logger.warning(f"Failed to create HarnessConfigStore: {e}")
        return None


_cache: dict[str, Any] = {"at": 0.0, "value": None}


def load_active_manifest(store: Optional[HarnessConfigStore]) -> tuple[int, dict[str, Any]]:
    """Return (version, manifest) for the active harness, cached briefly.

    Falls back to the built-in baseline when Mongo is down, so the agent never fails to start
    just because the config store is unreachable.
    """
    now = time.monotonic()
    if _cache["value"] is not None and now - _cache["at"] < CACHE_TTL_SECONDS:
        return _cache["value"]
    value: tuple[int, dict[str, Any]] = (BUILTIN_VERSION, copy.deepcopy(DEFAULT_MANIFEST))
    if store is not None:
        try:
            doc = store.get_active()
            if doc and not validate_manifest(doc["manifest"]):
                value = (doc["version"], doc["manifest"])
            elif doc:
                logger.error(f"Active manifest v{doc['version']} fails invariants; using baseline")
        except Exception as e:
            logger.warning(f"Could not load active manifest, using baseline: {e}")
    _cache.update(at=now, value=value)
    return value
