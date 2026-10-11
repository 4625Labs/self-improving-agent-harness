"""Regression court: reviewer/signal-nominated claims become convicted regression scenarios.

Lifecycle: nominated -> filed (draft built, no expected) -> convicted (verdict written to
docket.json) | dismissed. Only a verdict ever writes to docket.json.
"""

import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness_evolve.audit import run_confirmed  # noqa: E402
from agent_agent46.policy_store import CLAIMS_COLLECTION, POLICY_COLLECTION  # noqa: E402
from evals.run_eval import AUTO, MANUAL, REVIEW  # noqa: E402

DOCKET_PATH = Path(__file__).resolve().parent.parent / "evals" / "scenarios" / "docket.json"
COURT_COLLECTION = "court_cases"
VALID_RECOMMENDATIONS = {AUTO, REVIEW, MANUAL}
SOURCES = {"reviewer_disagreement", "raindrop_signal", "manual"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(doc):
    return None if doc is None else {k: v for k, v in doc.items() if k != "_id"}


class CourtDocket:
    def __init__(self, db, docket_path=None):
        self._db = db
        self._cases = db[COURT_COLLECTION]
        if docket_path is None:  # same override the evaluator reads, so reads and writes cannot diverge
            from evals.run_eval import _docket_path

            docket_path = _docket_path()
        self.docket_path = Path(docket_path)

    # -- helpers ---------------------------------------------------------
    def _get(self, case_id: str) -> dict:
        case = _clean(self._cases.find_one({"case_id": case_id}))
        if not case:
            raise ValueError(f"unknown case: {case_id}")
        return case

    def _load_docket(self) -> list:
        if not self.docket_path.exists():
            return []
        return json.loads(self.docket_path.read_text())

    def _write_docket(self, scenarios: list) -> None:
        self.docket_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.docket_path.parent, prefix=".docket-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(scenarios, f, indent=1)
                f.write("\n")
            os.replace(tmp, self.docket_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # -- public API ------------------------------------------------------
    def nominate(self, claim_id: str, source: str, reason: str) -> dict:
        if source not in SOURCES:
            raise ValueError(f"source must be one of {sorted(SOURCES)}")
        claim_id = claim_id.upper()
        existing = _clean(self._cases.find_one({"claim_id": claim_id}))
        if existing:
            return existing
        case = {
            "case_id": uuid.uuid4().hex[:8],
            "claim_id": claim_id,
            "source": source,
            "reason": reason,
            "status": "nominated",
            "created_at": _now(),
        }
        self._cases.insert_one(dict(case))
        return case

    def file_case(self, case_id: str) -> dict:
        case = self._get(case_id)
        if case["status"] not in ("nominated", "filed"):
            raise ValueError(f"case {case_id} is {case['status']}; cannot file")
        claim = _clean(self._db[CLAIMS_COLLECTION].find_one({"claim_id": case["claim_id"]}))
        if not claim:
            raise ValueError(f"claim not found: {case['claim_id']}")
        policy_number = claim.get("policy_number", "")
        policy = _clean(self._db[POLICY_COLLECTION].find_one({"policy_number": policy_number})) or {}
        prev = [
            c for c in self._db[CLAIMS_COLLECTION].find({"policy_number": policy_number})
            if c.get("claim_id") != claim["claim_id"]
        ]
        draft = {
            "id": f"court-{case_id}",
            "claim": {
                "claim_amount": claim.get("claim_amount"),
                "claim_type": claim.get("claim_type"),
                "description": claim.get("description"),
            },
            "policy": {
                "risk_score": policy.get("risk_score"),
                "coverage_limit": policy.get("coverage_limit"),
            },
            "previous_claims_count": len(prev),
            "tags": ["source:court"],
        }
        updates = {
            "status": "filed",
            "draft": draft,
            "agent": {
                "recommendation": claim.get("recommendation"),
                "risk_assessment": claim.get("risk_assessment"),
                "harness_version": claim.get("harness_version"),
            },
            "resolution": claim.get("resolution"),
            "filed_at": _now(),
        }
        self._cases.update_one({"case_id": case_id}, {"$set": updates})
        return self._get(case_id)

    def verdict(self, case_id: str, expected_recommendation: str, critical: bool, by: str, *,
                confirmed: bool = False) -> dict:
        """Writes docket.json. Needs confirmed=True; always audited."""
        return run_confirmed(self._db, "court.verdict", by, case_id,
                             {"expected": expected_recommendation, "critical": bool(critical)}, confirmed,
                             lambda: self._verdict(case_id, expected_recommendation, critical, by))

    def _verdict(self, case_id: str, expected_recommendation: str, critical: bool, by: str) -> dict:
        if expected_recommendation not in VALID_RECOMMENDATIONS:
            raise ValueError(f"expected_recommendation must be one of {sorted(VALID_RECOMMENDATIONS)}")
        case = self._get(case_id)
        if case["status"] != "filed":
            raise ValueError(f"case {case_id} is {case['status']}; only filed cases can receive a verdict")
        scenario = dict(case["draft"])
        scenario["expected"] = {"recommendation": expected_recommendation}
        scenario["severity"] = "critical" if critical else "normal"
        scenario["tags"] = list(dict.fromkeys(list(scenario.get("tags", [])) + ["source:court"]))
        docket = [s for s in self._load_docket() if s.get("id") != scenario["id"]]
        docket.append(scenario)
        self._write_docket(docket)
        self._cases.update_one({"case_id": case_id}, {"$set": {
            "status": "convicted", "verdict_by": by, "verdict_at": _now(),
            "expected_recommendation": expected_recommendation, "critical": bool(critical),
        }})
        return self._get(case_id)

    def dismiss(self, case_id: str, by: str, note: str = "", *, confirmed: bool = False) -> dict:
        """Needs confirmed=True; always audited."""
        return run_confirmed(self._db, "court.dismiss", by, case_id, {"note": note}, confirmed,
                             lambda: self._dismiss(case_id, by, note))

    def _dismiss(self, case_id: str, by: str, note: str) -> dict:
        case = self._get(case_id)
        if case["status"] == "convicted":
            raise ValueError(f"case {case_id} is already convicted")
        self._cases.update_one({"case_id": case_id}, {"$set": {
            "status": "dismissed", "dismissed_by": by, "dismiss_note": note, "dismissed_at": _now()}})
        return self._get(case_id)

    def _history_entry(self, case: dict, by: str, reason: str, action: str) -> dict:
        return {"action": action, "previous_expected": case.get("expected_recommendation"),
                "previous_critical": bool(case.get("critical")), "by": by, "reason": reason, "ts": _now()}

    def overturn(self, case_id: str, expected_recommendation: str, critical: bool, by: str, reason: str, *,
                 confirmed: bool = False) -> dict:
        """Appeal: change a convicted case's verdict in place. Needs confirmed=True and a reason; audited."""
        return run_confirmed(self._db, "court.overturn", by, case_id,
                             {"expected": expected_recommendation, "critical": bool(critical), "reason": reason},
                             confirmed,
                             lambda: self._overturn(case_id, expected_recommendation, critical, by, reason))

    def _overturn(self, case_id, expected_recommendation, critical, by, reason) -> dict:
        if expected_recommendation not in VALID_RECOMMENDATIONS:
            raise ValueError(f"expected_recommendation must be one of {sorted(VALID_RECOMMENDATIONS)}")
        if not (reason or "").strip():
            raise ValueError("a reason is required to overturn a verdict")
        case = self._get(case_id)
        if case["status"] != "convicted":
            raise ValueError(f"case {case_id} is {case['status']}; only convicted cases can be overturned")
        sid = case["draft"]["id"]
        docket = self._load_docket()
        new = {"expected": {"recommendation": expected_recommendation},
               "severity": "critical" if critical else "normal"}
        found = False
        out = []
        for s in docket:
            if s.get("id") == sid:
                if found:
                    continue  # drop any pre-existing duplicate of the same id
                s = {**s, **new}
                found = True
            out.append(s)
        if not found:
            s = dict(case["draft"])
            s.update(new)
            s["tags"] = list(dict.fromkeys(list(s.get("tags", [])) + ["source:court"]))
            out.append(s)
        self._write_docket(out)
        history = list(case.get("history", [])) + [self._history_entry(case, by, reason, "overturn")]
        self._cases.update_one({"case_id": case_id}, {"$set": {
            "expected_recommendation": expected_recommendation, "critical": bool(critical), "history": history}})
        return self._get(case_id)

    def retract(self, case_id: str, by: str, reason: str, *, confirmed: bool = False) -> dict:
        """Remove a convicted case's scenario from docket.json and mark it dismissed. Audited."""
        return run_confirmed(self._db, "court.retract", by, case_id, {"reason": reason}, confirmed,
                             lambda: self._retract(case_id, by, reason))

    def _retract(self, case_id, by, reason) -> dict:
        if not (reason or "").strip():
            raise ValueError("a reason is required to retract a verdict")
        case = self._get(case_id)
        if case["status"] != "convicted":
            raise ValueError(f"case {case_id} is {case['status']}; only convicted cases can be retracted")
        sid = case["draft"]["id"]
        self._write_docket([s for s in self._load_docket() if s.get("id") != sid])
        history = list(case.get("history", [])) + [self._history_entry(case, by, reason, "retract")]
        self._cases.update_one({"case_id": case_id}, {"$set": {
            "status": "dismissed", "dismissed_by": by, "dismiss_note": f"retracted: {reason}",
            "dismissed_at": _now(), "history": history}})
        return self._get(case_id)

    def list_cases(self, status=None) -> list:
        q = {"status": status} if status else {}
        return [_clean(c) for c in self._cases.find(q)]

    def nominate_reviewer_disagreements(self) -> list:
        new = []
        for c in self._db[CLAIMS_COLLECTION].find(
            {"recommendation": {"$in": [REVIEW, MANUAL]}, "resolution": "approved"}
        ):
            if self._cases.find_one({"claim_id": c["claim_id"]}):
                continue
            new.append(self.nominate(
                c["claim_id"], "reviewer_disagreement",
                f"agent recommended {c['recommendation']} but reviewer approved"))
        return new
