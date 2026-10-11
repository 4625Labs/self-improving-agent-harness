import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import run_eval
from evals.run_eval import load_docket, load_scenarios

KEYS = {"id", "claim", "policy", "previous_claims_count", "expected", "tags", "severity"}
RECS = {"auto_approve", "review_recommended", "manual_review_required"}
DEV = load_scenarios("dev")
DOCKET_IDS = {s["id"] for s in load_docket()}


@pytest.mark.parametrize("sc", DEV, ids=[s["id"] for s in DEV])
def test_scenario_honesty(sc):
    assert set(sc) == KEYS
    assert sc["expected"]["recommendation"] in RECS
    assert sc["severity"] in {"normal", "critical"}
    assert isinstance(sc["claim"]["claim_amount"], (int, float)) and sc["claim"]["claim_amount"] >= 0
    assert 0 <= sc["policy"]["risk_score"] <= 1
    assert sc["policy"]["coverage_limit"] > 0
    assert isinstance(sc["previous_claims_count"], int) and sc["previous_claims_count"] >= 0
    assert isinstance(sc["tags"], list) and all(isinstance(t, str) for t in sc["tags"])
    if sc["id"] in DOCKET_IDS:
        assert "source:court" in sc["tags"]


def _entry(sid="c01", desc="Court convicted unique description"):
    return {"id": sid, "claim": {"claim_amount": 100, "claim_type": "auto", "description": desc},
            "policy": {"risk_score": 0.2, "coverage_limit": 50000}, "previous_claims_count": 0,
            "expected": {"recommendation": "review_recommended"}, "tags": ["source:court"], "severity": "normal"}


def _use(tmp_path, monkeypatch, entries):
    p = tmp_path / "docket.json"
    p.write_text(json.dumps(entries))
    monkeypatch.setenv("EVOLVE_DOCKET_PATH", str(p))


def test_docket_in_dev_not_heldout(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, [_entry("c01")])
    assert "c01" in {s["id"] for s in load_scenarios("dev")}
    assert "c01" not in {s["id"] for s in load_scenarios("heldout")}


def test_missing_docket_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLVE_DOCKET_PATH", str(tmp_path / "nope.json"))
    assert load_scenarios("dev") == json.loads((run_eval.SCENARIO_DIR / "dev.json").read_text())


@pytest.mark.parametrize("mutate", [
    lambda e: e.pop("severity"),
    lambda e: e.update(extra=1),
    lambda e: e.update(severity="bad"),
    lambda e: e["expected"].update(recommendation="nope"),
    lambda e: e.update(tags="source:court"),
    lambda e: e.update(previous_claims_count=-1),
    lambda e: e.update(id="d01"),
])
def test_malformed_docket_rejected(tmp_path, monkeypatch, mutate):
    e = _entry()
    mutate(e)
    _use(tmp_path, monkeypatch, [e])
    with pytest.raises(ValueError):
        load_scenarios("dev")


def test_docket_heldout_description_leak_rejected(tmp_path, monkeypatch):
    held = json.loads((run_eval.SCENARIO_DIR / "heldout.json").read_text())[0]["claim"]["description"]
    _use(tmp_path, monkeypatch, [_entry(desc=held)])
    with pytest.raises(ValueError, match="leakage"):
        load_scenarios("dev")
