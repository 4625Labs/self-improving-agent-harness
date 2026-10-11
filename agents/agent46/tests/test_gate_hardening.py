"""Covers the gate/invariant hardening: escalation regressions and keyword limits."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_agent46.harness_config import DEFAULT_MANIFEST, validate_manifest  # noqa: E402
from evals.run_eval import compare  # noqa: E402


def _result(split="dev", acc=0.8, over=0.1, under=0.1, crit=0):
    return {"split": split, "accuracy": acc, "over_escalation_rate": over,
            "under_escalation_rate": under, "critical_violations": crit}


def _manifest_with(level, keywords):
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["keywords"][level] = keywords
    return m


def test_gate_fails_when_over_escalation_rises_despite_accuracy_gain():
    verdict = compare(_result(acc=0.7, over=0.1), _result(acc=0.9, over=0.2))
    assert verdict["gate"] == "fail"
    assert any("over-escalation" in r for r in verdict["reasons"])


def test_gate_fails_when_under_escalation_rises():
    verdict = compare(_result(under=0.1), _result(acc=0.9, under=0.2))
    assert verdict["gate"] == "fail"


def test_gate_passes_when_all_rates_hold_or_improve():
    assert compare(_result(), _result(acc=0.9, over=0.05, under=0.1))["gate"] == "pass"


def test_keyword_too_short_rejected():
    errs = validate_manifest(_manifest_with("high", DEFAULT_MANIFEST["keywords"]["high"] + ["car"]))
    assert any("'car'" in e for e in errs)


def test_keyword_bad_characters_rejected():
    errs = validate_manifest(_manifest_with("high", DEFAULT_MANIFEST["keywords"]["high"] + ["a.*b+"]))
    assert errs


def test_too_many_keywords_rejected():
    extra = [f"phrase number {i}" for i in range(30)]
    errs = validate_manifest(_manifest_with("medium", DEFAULT_MANIFEST["keywords"]["medium"] + extra))
    assert any("too many" in e for e in errs)


def test_reasonable_keyword_accepted():
    assert validate_manifest(_manifest_with("high", DEFAULT_MANIFEST["keywords"]["high"] + ["arson"])) == []


def test_unknown_manifest_field_rejected():
    m = copy.deepcopy(DEFAULT_MANIFEST)
    m["memory_labels"] = {"scope": "org"}
    errs = validate_manifest(m)
    assert any("unknown manifest fields" in e for e in errs)
