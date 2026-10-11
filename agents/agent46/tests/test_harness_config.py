import copy
import uuid

import pytest
from pymongo import MongoClient

from agent_agent46 import harness_config as hc
from agent_agent46.harness_config import (
    DEFAULT_MANIFEST,
    HarnessConfigStore,
    load_active_manifest,
    render_system_prompt,
    validate_manifest,
)

# directConnection: the replica set advertises its internal host "mongodb", unreachable from the host
MONGO_URI = "mongodb://localhost:53087/?directConnection=true"


def make(**over):
    m = copy.deepcopy(DEFAULT_MANIFEST)
    for k, v in over.items():
        if k == "thresholds":
            m["thresholds"].update(v)
        else:
            m[k] = v
    return m


def test_default_valid():
    assert validate_manifest(DEFAULT_MANIFEST) == []


@pytest.mark.parametrize(
    "thr",
    [
        {"auto_approve_limit": 1001},
        {"auto_approve_limit": -1},
        {"medium_amount": 999},
        {"medium_amount": 10001},
        {"high_amount": 4999},
        {"high_amount": 30001},
        {"medium_score": 0.05},
        {"high_score": 0.95},
        {"medium_amount": 9000, "high_amount": 9000},
        {"medium_amount": 9000, "high_amount": 6000},
        {"medium_score": 0.7, "high_score": 0.6},
        {"medium_score": 0.6, "high_score": 0.6},
        {"auto_approve_limit": "500"},
        {"auto_approve_limit": True},
    ],
)
def test_bad_thresholds(thr):
    assert validate_manifest(make(thresholds=thr))


def test_tightened_thresholds_ok():
    assert validate_manifest(make(thresholds={"auto_approve_limit": 500, "medium_amount": 2000})) == []


def test_missing_sections():
    assert validate_manifest({})


def test_removed_baseline_keywords():
    m = make()
    m["keywords"]["high"] = [k for k in m["keywords"]["high"] if k != "fire"]
    assert any("fire" in e for e in validate_manifest(m))
    m = make()
    m["keywords"]["medium"] = []
    assert validate_manifest(m)


def test_added_keywords_ok():
    m = make()
    m["keywords"]["high"] = m["keywords"]["high"] + ["arson"]
    assert validate_manifest(m) == []


def test_rule_limits():
    assert validate_manifest(make(rules=[{"id": f"r{i}", "text": "x"} for i in range(21)]))
    assert validate_manifest(make(rules=[{"id": f"r{i}", "text": "x"} for i in range(20)])) == []
    assert validate_manifest(make(rules=[{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]))
    assert validate_manifest(make(rules=[{"id": "a", "text": "x" * 301}]))
    assert validate_manifest(make(rules=[{"id": "a", "text": "x" * 300}])) == []
    assert validate_manifest(make(rules=[{"id": "a", "text": "  "}]))
    assert validate_manifest(make(rules=[{"text": "no id"}]))
    assert validate_manifest(make(rules="not a list"))
    assert validate_manifest(make(rules=["string rule"]))


def test_render_replaces_limit():
    out = render_system_prompt("Auto approve under $1,000 only.", make(thresholds={"auto_approve_limit": 500}))
    assert "$500" in out and "$1,000" not in out
    assert "## Learned Rules" not in out


def test_render_default_unchanged_without_rules():
    base = "limit $1,000."
    assert render_system_prompt(base, DEFAULT_MANIFEST) == base


def test_render_appends_rules():
    out = render_system_prompt("base", make(rules=[{"id": "a", "text": "Rule A"}, {"id": "b", "text": "Rule B"}]))
    assert "## Learned Rules" in out and "- Rule A" in out and "- Rule B" in out
    assert out.startswith("base")


# ---- store tests (local Docker Mongo) ----


@pytest.fixture
def store():
    dbname = f"harness_test_{uuid.uuid4().hex[:10]}"
    assert dbname != "agent_agent46_local"
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
    except Exception:
        pytest.skip("local MongoDB not reachable")
    s = HarnessConfigStore(MONGO_URI, dbname)
    yield s
    client.drop_database(dbname)
    client.close()


@pytest.fixture(autouse=True)
def reset_cache():
    hc._cache.update(at=0.0, value=None)
    yield
    hc._cache.update(at=0.0, value=None)


def test_seed_creates_v1_active(store):
    store.seed_if_empty()
    docs = store.list_versions()
    assert len(docs) == 1
    assert docs[0]["version"] == 1 and docs[0]["status"] == "active"
    assert docs[0]["manifest"] == DEFAULT_MANIFEST
    store.seed_if_empty()  # idempotent
    assert len(store.list_versions()) == 1


def test_propose_rejects_invalid(store):
    store.seed_if_empty()
    with pytest.raises(ValueError):
        store.propose(make(thresholds={"auto_approve_limit": 5000}), parent=1, reason="x", author="t")
    assert len(store.list_versions()) == 1


def test_propose_candidates_increment(store):
    store.seed_if_empty()
    v2 = store.propose(make(thresholds={"auto_approve_limit": 800}), parent=1, reason="r", author="t")
    v3 = store.propose(make(thresholds={"auto_approve_limit": 600}), parent=1, reason="r", author="t")
    assert (v2, v3) == (2, 3)
    d = store.get(2)
    assert d["status"] == "candidate" and d["parent"] == 1 and d["author"] == "t"
    assert store.get_active()["version"] == 1


def test_activate_retires_previous(store):
    store.seed_if_empty()
    v2 = store.propose(make(thresholds={"auto_approve_limit": 800}), parent=1, reason="r", author="t")
    store.activate(v2)
    assert store.get(1)["status"] == "retired"
    assert store.get(2)["status"] == "active" and store.get(2)["activated_at"]
    assert store.get_active()["version"] == 2


def test_activate_unknown_version(store):
    with pytest.raises(ValueError):
        store.activate(99)


def test_activate_refuses_invalid(store):
    store.seed_if_empty()
    bad = make(thresholds={"auto_approve_limit": 5000})
    # bypass propose validation by inserting directly
    store._collection.insert_one(
        {"version": 2, "parent": 1, "status": "candidate", "manifest": bad, "reason": "", "author": "t",
         "scores": {}, "created_at": "", "activated_at": None}
    )
    with pytest.raises(ValueError):
        store.activate(2)
    assert store.get(1)["status"] == "active"
    assert store.get(2)["status"] == "candidate"


def test_load_active_fallback_when_no_store():
    version, manifest = load_active_manifest(None)
    assert version == 0 and manifest == DEFAULT_MANIFEST
    manifest["rules"].append("mutate")  # must be a copy
    hc._cache.update(at=0.0, value=None)
    assert load_active_manifest(None)[1] == DEFAULT_MANIFEST


def test_load_active_from_store_and_cache(store):
    store.seed_if_empty()
    assert load_active_manifest(store)[0] == 1
    v2 = store.propose(make(thresholds={"auto_approve_limit": 800}), parent=1, reason="r", author="t")
    store.activate(v2)
    assert load_active_manifest(store)[0] == 1  # cached
    hc._cache.update(at=0.0, value=None)
    version, manifest = load_active_manifest(store)
    assert version == 2 and manifest["thresholds"]["auto_approve_limit"] == 800


def test_load_active_invalid_active_falls_back(store):
    store.seed_if_empty()
    store._collection.update_one(
        {"version": 1}, {"$set": {"manifest.thresholds.auto_approve_limit": 99999}}
    )
    assert load_active_manifest(store)[0] == 0
