import copy
import uuid
from pathlib import Path

import pytest
from pymongo import MongoClient

from agent_agent46 import harness_config as hc
from agent_agent46.harness_config import DEFAULT_MANIFEST, HarnessConfigStore, validate_manifest
from agent_agent46.tool_access import (
    ALL_TOOLS,
    CLAIM_PIPELINE_TOOLS,
    INTENTS,
    REQUIRED_TOOLS,
    SENSITIVE_TOOLS,
    allowed_tools,
    baseline_tool_access,
    claim_in_flight,
    classify_intent,
    effective_tool_access,
    resolve_tool_access,
    validate_tool_access,
)

MONGO_URI = "mongodb://localhost:53087/?directConnection=true"
AGENT_YAML = Path(__file__).resolve().parent.parent / "agent.yaml"


def narrowed(intent, remove):
    ta = baseline_tool_access()
    ta["intents"][intent]["allow"] = [t for t in ta["intents"][intent]["allow"] if t != remove]
    return ta


# ---- classify_intent ----


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I want to file a claim", "claims"),
        ("My car was stolen last night", "claims"),
        ("There was a hit-and-run", "claims"),
        ("I need a quote for my car", "quote"),
        ("I'd like to buy a policy", "quote"),
        ("What is a premium?", "info"),
        ("Please explain umbrella policies", "info"),
        ("CLAIM STATUS PLEASE", "claims"),
    ],
)
def test_single_group(text, expected):
    assert classify_intent(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "What's the difference between collision and comprehensive coverage?",
        "I had a collision, what is my deductible?",
        "I want a quote after my accident",
    ],
)
def test_ambiguity_fails_open(text):
    assert classify_intent(text) == "general"


def test_no_keywords_general():
    assert classify_intent("hello there") == "general"
    assert classify_intent("hello there", ["lookup_policy", "get_quote"]) == "general"


@pytest.mark.parametrize("tool", sorted(CLAIM_PIPELINE_TOOLS))
def test_sticky_claim(tool):
    assert classify_intent("yes, go ahead", [tool]) == "claims"


def test_sticky_accepts_any_iterable():
    assert classify_intent("ok", iter(["resolve_claim"])) == "claims"
    assert classify_intent("ok", {"file_claim"}) == "claims"


def test_clear_quote_overrides_sticky():
    assert classify_intent("Now I need a quote for my house", ["file_claim"]) == "quote"


def test_ambiguous_overrides_sticky_to_general():
    # Hits in several groups means general, even mid-claim.
    assert classify_intent("what is my deductible after the collision", ["file_claim"]) == "general"


@pytest.mark.parametrize("text", ["", None, "   ", "\n"])
def test_empty_text(text):
    assert classify_intent(text) == "general"


def test_empty_text_sticky():
    assert classify_intent("", ["human_review"]) == "claims"
    assert classify_intent(None, ["human_review"]) == "claims"


def test_word_boundary_prefix():
    assert classify_intent("bypass this") == "general"  # 'buy' not matched inside a word
    assert classify_intent("a disclaimer") == "general"  # 'claim' only matches at word start
    assert classify_intent("prefix") == "general"


def test_compound_claim_words_are_matched():
    assert classify_intent("wildfire") == "claims"
    assert classify_intent("my car was in a wreck") == "claims"
    assert classify_intent("I was rear-ended at a light") == "claims"
    assert classify_intent("Someone vandalized my car overnight") == "claims"


def test_whole_word_matching_with_simple_endings():
    assert classify_intent("the claimant called") == "general"  # 'claimant' is not 'claim'
    assert classify_intent("I filed two claims") == "claims"
    assert classify_intent("my car was damaged") == "claims"
    assert classify_intent("I am buying a car") == "quote"


def test_substring_false_positives_are_gone():
    assert classify_intent("I love my fireplace") == "general"
    assert classify_intent("a priceless heirloom") == "general"
    assert classify_intent("what issues came up") == "general"


# ---- claim in flight ----


def test_claim_in_flight_requires_filed_without_later_notification():
    assert not claim_in_flight([])
    assert claim_in_flight(["file_claim"])
    assert claim_in_flight(["file_claim", "analyze_claim_risk", "human_review"])
    assert not claim_in_flight(["file_claim", "send_notification"])
    assert claim_in_flight(["file_claim", "send_notification", "file_claim"])  # a second claim


def _info_narrowed():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["tool_access"]["intents"]["info"]["allow"] = sorted(REQUIRED_TOOLS["info"])
    return m


def test_mid_claim_info_question_cannot_strand_the_claim():
    manifest = _info_narrowed()
    intent, allowed = resolve_tool_access(manifest, "what is a deductible?", [])
    assert intent == "info" and "resolve_claim" not in allowed  # normal narrowing
    intent, allowed = resolve_tool_access(
        manifest, "what is a deductible?", ["file_claim", "analyze_claim_risk"]
    )
    assert intent == "info"
    assert {"resolve_claim", "human_review", "send_notification"} <= allowed
    assert "create_policy" not in allowed  # only the claim-flow tools come back


def test_resolve_tool_access_baseline_allows_everything():
    for text in ("what is a deductible?", "I was in an accident", "hello"):
        assert resolve_tool_access(DEFAULT_MANIFEST, text, [])[1] == ALL_TOOLS


# ---- validate_tool_access ----


def test_baseline_valid():
    assert validate_tool_access(baseline_tool_access()) == []


@pytest.mark.parametrize(
    "intent,tool",
    [(i, t) for i in INTENTS for t in sorted(REQUIRED_TOOLS[i])],
)
def test_reject_removing_required(intent, tool):
    errs = validate_tool_access(narrowed(intent, tool))
    assert errs and any(tool in e for e in errs)


def test_reject_narrowing_general():
    assert validate_tool_access(narrowed("general", "resolve_claim"))


def test_accept_narrowing_removable():
    assert validate_tool_access(narrowed("info", "resolve_claim")) == []
    assert validate_tool_access(narrowed("quote", "resolve_claim")) == []
    ta = baseline_tool_access()
    ta["intents"]["info"]["allow"] = sorted(REQUIRED_TOOLS["info"])
    assert validate_tool_access(ta) == []


def test_reject_unknown_tool():
    ta = baseline_tool_access()
    ta["intents"]["info"]["allow"].append("drop_database")
    assert validate_tool_access(ta)


def test_reject_duplicates():
    ta = baseline_tool_access()
    ta["intents"]["info"]["allow"].append("get_coverage_options")
    assert validate_tool_access(ta)


def test_reject_missing_and_extra_intents():
    ta = baseline_tool_access()
    del ta["intents"]["quote"]
    assert validate_tool_access(ta)
    ta = baseline_tool_access()
    ta["intents"]["admin"] = {"allow": sorted(ALL_TOOLS)}
    assert validate_tool_access(ta)


def test_reject_extra_keys():
    ta = baseline_tool_access()
    ta["extra"] = 1
    assert validate_tool_access(ta)
    ta = baseline_tool_access()
    ta["intents"]["info"]["deny"] = []
    assert validate_tool_access(ta)
    assert validate_tool_access({})


def test_reject_non_list_allow():
    for bad in ("resolve_claim", None, {"a": 1}, 5, [1, 2], ("lookup_policy",)):
        ta = baseline_tool_access()
        ta["intents"]["info"]["allow"] = bad
        assert validate_tool_access(ta), bad


@pytest.mark.parametrize("bad", [None, [], "x", 3, ["intents"]])
def test_reject_non_dict(bad):
    assert validate_tool_access(bad)


def test_reject_non_dict_intent_entry():
    ta = baseline_tool_access()
    ta["intents"]["info"] = ["get_coverage_options"]
    assert validate_tool_access(ta)
    ta["intents"] = []
    assert validate_tool_access(ta)


def test_unhashable_allow_entries_do_not_crash():
    ta = baseline_tool_access()
    ta["intents"]["info"]["allow"] = [["x"]]
    assert validate_tool_access(ta)


# ---- allowed_tools / effective_tool_access ----


def test_allowed_unknown_intent_is_general():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["tool_access"] = narrowed("info", "resolve_claim")
    assert allowed_tools(m, "bogus") == allowed_tools(m, "general") == ALL_TOOLS
    assert allowed_tools(m, "") == ALL_TOOLS


def test_allowed_narrowed_and_subset():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["tool_access"] = narrowed("info", "resolve_claim")
    assert "resolve_claim" not in allowed_tools(m, "info")
    assert "resolve_claim" in allowed_tools(m, "claims")
    for i in INTENTS:
        assert allowed_tools(m, i) <= ALL_TOOLS


def test_allowed_filters_unknown_names():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["tool_access"]["intents"]["info"]["allow"].append("ghost")
    assert "ghost" not in allowed_tools(m, "info")


def test_effective_is_deep_copy():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    eff = effective_tool_access(m)
    eff["intents"]["info"]["allow"].clear()
    assert m["tool_access"]["intents"]["info"]["allow"]
    assert effective_tool_access(m) == m["tool_access"]


def test_baseline_returns_fresh_object():
    a = baseline_tool_access()
    a["intents"]["info"]["allow"].clear()
    assert baseline_tool_access()["intents"]["info"]["allow"]


def test_manifest_without_tool_access_is_baseline():
    old = {k: v for k, v in copy.deepcopy(DEFAULT_MANIFEST).items() if k != "tool_access"}
    assert effective_tool_access(old) == baseline_tool_access()
    assert allowed_tools(old, "info") == ALL_TOOLS
    assert validate_manifest(old) == []


def test_explicit_none_tool_access_rejected_by_manifest():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["tool_access"] = None
    assert validate_manifest(m)


# ---- consistency ----


def _yaml_tools():
    text = AGENT_YAML.read_text()
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        sb = yaml.safe_load(text)["sandboxes"]
        return set(sb["agent"]["tools"]) | set(sb["tool"]["tools"])
    tools, in_sb, in_tools = set(), False, False  # minimal fallback parser
    for line in text.splitlines():
        if line.startswith("sandboxes:"):
            in_sb = True
        elif in_sb and line and not line.startswith(" "):
            break
        elif in_sb:
            s = line.strip()
            if s == "tools:":
                in_tools = True
            elif in_tools and s.startswith("- "):
                tools.add(s[2:].strip())
            elif s:
                in_tools = False
    return tools


def test_required_subset_of_all():
    for i in INTENTS:
        assert REQUIRED_TOOLS[i] <= ALL_TOOLS
    assert set(REQUIRED_TOOLS) == set(INTENTS)
    assert REQUIRED_TOOLS["general"] == ALL_TOOLS


def test_sensitive_and_pipeline_subsets():
    assert SENSITIVE_TOOLS <= ALL_TOOLS
    assert CLAIM_PIPELINE_TOOLS <= ALL_TOOLS


def test_all_tools_matches_agent_yaml():
    assert ALL_TOOLS == _yaml_tools()


def test_default_manifest_tool_access_is_baseline():
    assert DEFAULT_MANIFEST["tool_access"] == baseline_tool_access()


# ---- harness_config integration ----


def test_manifest_default_valid():
    assert validate_manifest(DEFAULT_MANIFEST) == []


def test_manifest_rejects_removed_required_tool():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["tool_access"] = narrowed("claims", "human_review")
    assert any("human_review" in e for e in validate_manifest(m))


def test_manifest_unknown_top_level_key_rejected():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["memory"] = {}
    assert any("unknown manifest fields" in e for e in validate_manifest(m))


def test_manifest_narrowed_valid():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["tool_access"] = narrowed("info", "resolve_claim")
    assert validate_manifest(m) == []


@pytest.fixture(autouse=True)
def reset_cache():
    hc._cache.update(at=0.0, value=None)
    yield
    hc._cache.update(at=0.0, value=None)


@pytest.fixture
def store():
    dbname = f"toolaccess_test_{uuid.uuid4().hex[:10]}"
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
    except Exception:
        client.close()
        pytest.skip("local MongoDB not reachable")
    s = HarnessConfigStore(MONGO_URI, dbname)
    yield s
    client.drop_database(dbname)
    client.close()


def test_store_roundtrip_narrowed_tool_access(store):
    store.seed_if_empty()
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["tool_access"] = narrowed("info", "resolve_claim")
    v = store.propose(m, parent=1, reason="narrow info", author="t")
    store.activate(v)
    active = store.get_active()
    assert active["version"] == v
    assert active["manifest"]["tool_access"] == m["tool_access"]
    version, loaded = hc.load_active_manifest(store)
    assert version == v
    assert "resolve_claim" not in allowed_tools(loaded, "info")
    assert "resolve_claim" in allowed_tools(loaded, "claims")


def test_store_rejects_bad_tool_access(store):
    store.seed_if_empty()
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["tool_access"] = narrowed("claims", "resolve_claim")
    with pytest.raises(ValueError):
        store.propose(m, parent=1, reason="bad", author="t")
    assert len(store.list_versions()) == 1


def test_store_activate_refuses_bad_tool_access(store):
    store.seed_if_empty()
    bad = copy.deepcopy(DEFAULT_MANIFEST)
    bad["tool_access"] = narrowed("general", "lookup_policy")
    store._collection.insert_one(
        {"version": 2, "parent": 1, "status": "candidate", "manifest": bad, "reason": "",
         "author": "t", "scores": {}, "created_at": "", "activated_at": None}
    )
    with pytest.raises(ValueError):
        store.activate(2)
    assert store.get(1)["status"] == "active"


def test_store_old_manifest_without_tool_access_loads(store):
    store.seed_if_empty()
    store._collection.update_one({"version": 1}, {"$unset": {"manifest.tool_access": ""}})
    version, manifest = hc.load_active_manifest(store)
    assert version == 1
    assert allowed_tools(manifest, "info") == ALL_TOOLS
