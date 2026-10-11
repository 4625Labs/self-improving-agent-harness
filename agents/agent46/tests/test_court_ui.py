import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402
from test_court import FakeDb  # noqa: E402

from agent_agent46.harness_config import DEFAULT_MANIFEST  # noqa: E402
from harness_evolve.court_ui.app import create_app  # noqa: E402
from harness_evolve.proposer import LLMProposer, reset_call_count  # noqa: E402

TOKEN = "test-token"


class _Coll:
    def __init__(self, docs):
        self.docs = docs

    def update_one(self, q, u):
        for d in self.docs:
            if all(d.get(k) == v for k, v in q.items()):
                d.update(u["$set"])


class FakeStore:
    def __init__(self):
        self.docs = [{"version": 1, "parent": None, "status": "active", "manifest": copy.deepcopy(DEFAULT_MANIFEST),
                      "reason": "baseline", "author": "human", "scores": {}}]
        self._collection = _Coll(self.docs)

    def get_active(self):
        return next((d for d in self.docs if d["status"] == "active"), None)

    def get(self, v):
        return next((d for d in self.docs if d["version"] == v), None)

    def list_versions(self):
        return list(self.docs)

    def propose(self, manifest, parent, reason, author):
        v = len(self.docs) + 1
        self.docs.append({"version": v, "parent": parent, "status": "candidate", "manifest": copy.deepcopy(manifest),
                          "reason": reason, "author": author, "scores": {}})
        return v

    def activate(self, v):
        for d in self.docs:
            if d["status"] == "active":
                d["status"] = "retired"
        self.get(v)["status"] = "active"

    def set_status(self, v, s):
        self.get(v)["status"] = s


def _make(tmp_path, desc="dent", llm_calls=None):
    db = FakeDb()
    db["insurance_policies"].insert_one({"policy_number": "POL1", "risk_score": 0.4, "coverage_limit": 50000})
    db["insurance_claims"].insert_one({
        "claim_id": "CLM1", "policy_number": "POL1", "user_id": "u", "claim_type": "auto", "claim_amount": 900.0,
        "description": desc, "recommendation": "review_recommended", "risk_assessment": "medium",
        "harness_version": 1, "resolution": "approved"})
    calls = [] if llm_calls is None else llm_calls

    def client(prompt, model, max_tokens):
        calls.append(prompt)
        return json.dumps({"ops": [], "rationale": "nothing"})

    reset_call_count()
    factory = lambda: LLMProposer(client=client, cache_dir=tmp_path / "cache", max_calls=3)  # noqa: E731
    app = create_app(db, FakeStore(), token=TOKEN, llm_factory=factory, docket_path=tmp_path / "docket.json")
    return TestClient(app, follow_redirects=True), calls


def _post(c, url, **data):
    return c.post(url, data={"csrf": TOKEN, **data})


def _case_id(c):
    html = c.get("/").text
    return html.split("/case/")[1].split("/")[0]


def test_docket_renders(tmp_path):
    c, _ = _make(tmp_path)
    r = c.get("/")
    assert r.status_code == 200
    for s in ("Nominated", "Filed", "Convicted / Dismissed", "Scan for disagreements", "Trial", "v1"):
        assert s in r.text
    assert "http://" not in r.text.replace("http://www.w3.org", "") and "https://" not in r.text


def test_scan_file_verdict_flow(tmp_path):
    c, _ = _make(tmp_path)
    assert "nominated 1 new" in _post(c, "/scan").text
    cid = _case_id(c)
    html = _post(c, f"/case/{cid}/file").text
    assert "Case file" in html and "review_recommended" in html
    html = _post(c, f"/case/{cid}/verdict", action="verdict", expected="manual_review_required",
                 critical="1", adjuster="Ann").text
    assert f"scenario written: court-{cid}" in html
    sc = json.loads((tmp_path / "docket.json").read_text())
    assert sc[0]["severity"] == "critical"


def test_verdict_requires_adjuster_and_valid_expected(tmp_path):
    c, _ = _make(tmp_path)
    _post(c, "/scan")
    cid = _case_id(c)
    _post(c, f"/case/{cid}/file")
    assert "adjuster name is required" in _post(c, f"/case/{cid}/verdict", action="verdict",
                                                expected="auto_approve", adjuster=" ").text
    assert "expected must be one of" in _post(c, f"/case/{cid}/verdict", action="verdict",
                                              expected="bogus", adjuster="Ann").text
    assert not (tmp_path / "docket.json").exists()


def test_post_without_token_rejected(tmp_path):
    c, _ = _make(tmp_path)
    assert c.post("/scan", data={}).status_code == 403
    assert c.post("/scan", data={"csrf": "wrong"}).status_code == 403
    assert c.post("/cycle", data={"proposer": "llm", "action": "llm_call"}).status_code == 403


def test_claim_text_is_escaped(tmp_path):
    c, _ = _make(tmp_path, desc="<script>alert(1)</script>")
    _post(c, "/scan")
    _post(c, f"/case/{_case_id(c)}/file")
    html = c.get("/").text
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_llm_path_dry_run_only_until_confirm(tmp_path):
    c, calls = _make(tmp_path)
    c.get("/")
    html = _post(c, "/cycle", proposer="llm").text
    assert calls == [] and "no model call was made" in html and "Call the model" in html
    assert "prompt tokens" in html
    html = _post(c, "/cycle", proposer="llm", action="llm_call").text
    assert len(calls) == 1 and "cycle finished" in html


def test_heuristic_cycle_and_approve_reject(tmp_path):
    c, calls = _make(tmp_path)
    html = _post(c, "/cycle", proposer="heuristic").text
    assert calls == [] and "cycle finished" in html
    assert "Approve" in html or "no_change" in html
    assert "not a candidate" in _post(c, "/version/1/reject").text


def test_nominate_unknown_claim_rejected(tmp_path):
    c, _ = _make(tmp_path)
    html = _post(c, "/nominate", claim_id="NOPE", reason="x").text
    assert "claim not found: NOPE" in html
    assert "<b>NOPE</b>" not in html
    assert "<b>CLM1</b>" in _post(c, "/nominate", claim_id="clm1").text


def test_dismiss_filed_case_needs_only_adjuster_and_forms_separate(tmp_path):
    c, _ = _make(tmp_path)
    _post(c, "/scan")
    cid = _case_id(c)
    html = _post(c, f"/case/{cid}/file").text
    # Dismiss lives in its own form, and no field is natively `required` (in-page error instead).
    assert 'name="adjuster" placeholder="adjuster (required)" required' not in html and 'placeholder="adjuster" required' not in html and "Adjuster name is required." in html
    assert html.count(f'action="/case/{cid}/verdict"') == 2
    assert "adjuster name is required" in _post(c, f"/case/{cid}/verdict", action="dismiss", adjuster="").text
    assert "dismissed case" in _post(c, f"/case/{cid}/verdict", action="dismiss", adjuster="Ann").text
    assert not (tmp_path / "docket.json").exists()


def test_trial_shows_candidate_row_after_cycle(tmp_path):
    c, _ = _make(tmp_path)
    assert "candidate (" not in c.get("/").text
    html = _post(c, "/cycle", proposer="heuristic").text
    assert "candidate (" in html and "held-out accuracy" in html
    assert "viewport" in html and "convicted:" not in html
