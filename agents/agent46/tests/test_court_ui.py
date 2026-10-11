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
    tc = TestClient(app, follow_redirects=True)
    tc.db, tc.docket = db, tmp_path / "docket.json"
    return tc, calls


def _post(c, url, **data):
    return c.post(url, data={"csrf": TOKEN, **data})


def _case_id(c):
    html = c.get("/").text
    return html.split("/case/")[1].split("/")[0]


def test_docket_renders(tmp_path):
    c, _ = _make(tmp_path)
    r = c.get("/")
    assert r.status_code == 200
    for s in ("Nominated", "Filed", "Convicted / Dismissed", "Scan for disagreements", "Policy scorecard", "v1"):
        assert s in r.text
    assert "http://" not in r.text.replace("http://www.w3.org", "") and "https://" not in r.text


def test_scan_file_verdict_flow(tmp_path):
    c, _ = _make(tmp_path)
    assert "nominated 1 new" in _post(c, "/scan").text
    cid = _case_id(c)
    html = _post(c, f"/case/{cid}/file").text
    assert "Case file" in html and "review_recommended" in html
    html = _post(c, f"/case/{cid}/verdict", action="verdict", expected="manual_review_required",
                 critical="1", adjuster="Ann", confirm="1").text
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
    assert calls == [] and "no model call was made" in html and "Ask the AI model now" in html
    assert "prompt tokens" in html
    html = _post(c, "/cycle", proposer="llm", action="llm_call").text
    assert len(calls) == 1 and "cycle finished" in html


def test_heuristic_cycle_and_approve_reject(tmp_path):
    c, calls = _make(tmp_path)
    html = _post(c, "/cycle", proposer="heuristic").text
    assert calls == [] and "cycle finished" in html
    assert "Approve and go live" in html or "no_change" in html
    assert "not a candidate" in _post(c, "/version/1/reject", confirm="1").text


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
    assert "dismissed case" in _post(c, f"/case/{cid}/verdict", action="dismiss", adjuster="Ann", confirm="1").text
    assert not (tmp_path / "docket.json").exists()


def test_trial_shows_candidate_row_after_cycle(tmp_path):
    c, _ = _make(tmp_path)
    assert "<td>Proposed change (awaiting approval)</td>" not in c.get("/").text
    html = _post(c, "/cycle", proposer="heuristic").text
    assert "<td>Proposed change (awaiting approval)</td>" in html and "Accuracy on unseen cases" in html
    assert "viewport" in html and "convicted:" not in html


def _seed_decided(c, n=25, desc="dent"):
    """Insert n extra claims and decide them (alternating convict/dismiss) through the real routes."""
    db = c.db
    for i in range(n):
        db["insurance_claims"].insert_one({
            "claim_id": f"SEED{i:02d}", "policy_number": "POL1", "user_id": "u", "claim_type": "auto",
            "claim_amount": 100.0, "description": desc, "recommendation": "review_recommended",
            "risk_assessment": "low", "harness_version": 1, "resolution": "approved"})
        _post(c, "/nominate", claim_id=f"SEED{i:02d}", reason="seed")
    for k, case in enumerate(_cases(c)):
        cid = case["case_id"]
        _post(c, f"/case/{cid}/file")
        if k % 2:
            _post(c, f"/case/{cid}/verdict", action="dismiss", adjuster="Dana", confirm="1")
        else:
            _post(c, f"/case/{cid}/verdict", action="verdict", expected="auto_approve", adjuster="Ann", confirm="1")


def _cases(c):
    from harness_evolve.court import CourtDocket
    return CourtDocket(c.db, c.docket).list_cases()


def _visible_rows(html):
    return [r for r in html.split('<tr class="drow"')[1:] if " hidden" not in r.split(">", 1)[0]]


def test_title_subtitle_and_constants(tmp_path):
    from harness_evolve.court_ui import app as appmod
    c, _ = _make(tmp_path)
    html = c.get("/").text
    assert f"<title>{appmod.APP_NAME}</title>" in html and appmod.APP_NAME == "Regression Court"
    assert f"<h1>{appmod.APP_NAME}</h1>" in html
    assert "Review agent failures, rule on them, ship safer policy" in html and "Court docket" not in html


def test_info_dialog_content_and_labels(tmp_path):
    c, _ = _make(tmp_path)
    html = c.get("/").text
    assert "<dialog" in html and 'aria-label="About Regression Court' in html and "alert(" not in html
    for s in ("Nominate", "Prove live", "Read the diff", "Verdict dropdown", "Critical", "adjuster", "Convict", "Dismiss",
              "Overturn", "Retract", "confirm box", "History", "Accuracy on known cases", "Accuracy on unseen cases", "Cases tested",
              "Dangerous misses", "Unneeded reviews", "Missed escalations", "Safety check", "Make live again", "Awaiting approval", "Rejected",
              "sent to a human anyway", "needed escalation but got a lighter outcome",
              "auto-approved"):
        assert s in html, s
    for s in ("Nominated", "Filed", "Convicted and Dismissed", "Policy scorecard", "Policy versions", "Activity log"):
        assert f'aria-label="About {s}"' in html, s


def test_tabs_counts_and_collapsible_panels(tmp_path):
    c, _ = _make(tmp_path)
    html = c.get("/").text
    assert 'data-default="nominated"' in html and 'role="tablist"' in html
    assert 'id="trial">' in html and "<details open><summary>Policy scorecard" in html
    assert '<details><summary>Policy versions' in html and '<details><summary>Activity log' in html
    assert "Scan" in html and 'class="ver"' in html
    _post(c, "/scan")
    _post(c, f"/case/{_case_id(c)}/file")
    html = c.get("/").text
    assert 'data-default="filed"' in html and 'aria-selected="false">Filed<span class="cnt">1<' in html


def test_decided_tab_paginates_and_has_working_forms(tmp_path):
    c, _ = _make(tmp_path)
    _seed_decided(c)
    html = c.get("/").text
    assert html.count('class="drow"') == 25 and len(_visible_rows(html)) == 10
    assert 'id="show-more"' in html and 'id="decided-search"' in html
    for chip in ("all", "convicted", "dismissed"):
        assert f'data-chip="{chip}"' in html
    assert 'class="badge b-convicted"' in html and 'class="badge b-dismissed"' in html
    # expanded content (hidden by default) still carries Overturn/Retract forms with the CSRF token
    assert f'name="csrf" value="{TOKEN}"' in html
    assert html.count("/overturn") == 13 and html.count("/retract") == 13
    row = html.split('<tr class="dd"')[1].split("</tr>")[0]
    assert "hidden" in html.split('<tr class="dd"')[1][:80] and "Overturn" in row and f'value="{TOKEN}"' in row
    cid = html.split("/overturn")[0].rsplit("/case/", 1)[1]
    assert "verdict overturned" in _post(c, f"/case/{cid}/overturn", expected="manual_review_required", adjuster="Bob",
                                         reason="new evidence", confirm="1").text
    assert "verdict retracted" in _post(c, f"/case/{cid}/retract", adjuster="Bob", reason="oops", confirm="1").text


def test_script_in_claim_text_escaped_in_decided_tab(tmp_path):
    c, _ = _make(tmp_path, desc="<script>alert(1)</script>")
    _post(c, "/scan")
    cid = _case_id(c)
    _post(c, f"/case/{cid}/file")
    _post(c, f"/case/{cid}/verdict", action="dismiss", adjuster="<script>alert(2)</script>", confirm="1")
    html = c.get("/").text
    assert "<script>alert(" not in html
    assert "&lt;script&gt;alert(2)&lt;/script&gt;" in html


# ---- plain-language labels ----
from harness_evolve.court_ui.app import UI_LABELS, _author_label, _pct  # noqa: E402


def _flat(d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v)
        else:
            yield k, v



def test_every_label_appears_on_page(tmp_path):
    import html as htmllib
    c, _ = _make(tmp_path)
    _post(c, "/scan")
    cid = _case_id(c)
    _post(c, f"/case/{cid}/file")
    _post(c, f"/case/{cid}/verdict", action="verdict", expected="manual_review_required", critical="1",
          adjuster="Ann", confirm="1")
    _post(c, "/cycle", proposer="heuristic")
    html = htmllib.unescape(_post(c, "/cycle", proposer="llm").text)
    for key, label in _flat(UI_LABELS):
        if key in ("row_candidate", "author_limits"):  # shown only in special cases
            continue
        if "{" in label or key.startswith(("op_", "group_", "tool_", "loosens", "no_", "risk_", "more_", "unknown", "show_tools", "impact_reasons")):
            continue  # card content that depends on the ops; covered by the card tests
        if label in UI_LABELS["thresholds"].values() or label in UI_LABELS["intents"].values() or label in UI_LABELS["limits"].values():
            continue
        assert label in html, (key, label)
    assert "<td>Proposed change (awaiting approval)</td>" in html


def test_status_and_author_mapping_renders(tmp_path):
    c, _ = _make(tmp_path)
    html = _post(c, "/cycle", proposer="heuristic").text
    assert ">Live</span>" in html and "Original" in html and "Awaiting approval" in html
    assert 'class="badge b-active">Live<' in html and 'class="badge b-candidate">Awaiting approval<' in html
    assert "Rule-based proposer" in html and "proposer:heuristic" not in html.split('id="versions"')[1]
    for raw, nice in (("human", "Original"), ("proposer:heuristic", "Rule-based proposer"), ("proposer:llm", "AI model"),
                      ("proposer:tool-heuristic", "Tool-access proposer"), ("weird", "weird"),
                      ("proposer:heuristic+tool-heuristic", "Rule-based proposer, Tool-access proposer")):
        assert _author_label(raw) == nice
    assert UI_LABELS["status"] == {"active": "Live", "retired": "Previous", "candidate": "Awaiting approval",
                                   "rejected": "Rejected"}


def test_percent_formatting_and_header_tooltips(tmp_path):
    assert _pct(0.79) == "79%" and _pct(1) == "100%" and _pct(0) == "0%" and _pct(None) == "-"
    c, _ = _make(tmp_path)
    html = c.get("/").text
    import re
    row = html.split('id="trial"')[1].split("</table>")[0]
    assert re.search(r"<td>\d+%</td>", row)
    assert f'title="{UI_LABELS["tip_critical"]}"' in html and "case marked critical that the policy would auto-approve" in html
    assert "must never be auto-approved" in html


def test_gate_badge_and_critical_badge(tmp_path):
    c, _ = _make(tmp_path)
    html = _post(c, "/cycle", proposer="heuristic").text
    assert 'class="badge b-active">Passed<' in html or 'class="badge b-rejected">Failed<' in html
    from jinja2 import Environment
    from harness_evolve.court_ui import app as appmod
    tpl = appmod.TEMPLATE
    assert 'gate == "pass"' in tpl and "L.gate_pass" in tpl and "L.gate_fail" in tpl
    assert "r.critical_violations %}<span class=\"badge b-critical\">" in tpl


def test_all_forms_carry_csrf_and_confirm(tmp_path):
    import re
    c, _ = _make(tmp_path)
    _seed_decided(c, n=4)
    _post(c, "/nominate", claim_id="CLM1", reason="again")
    html = _post(c, "/cycle", proposer="llm").text
    forms = re.findall(r"<form .*?</form>", html, re.S)
    assert len(forms) >= 10
    for f in forms:
        assert f'name="csrf" value="{TOKEN}"' in f, f[:80]
        if "/verdict" in f or "/overturn" in f or "/retract" in f or "/version/" in f:
            assert 'name="confirm"' in f, f[:80]
    html = _post(c, "/cycle", proposer="heuristic").text
    for f in re.findall(r"<form [^>]*/version/.*?</form>", html, re.S):
        assert 'name="confirm"' in f and 'name="csrf"' in f


def test_script_in_claim_text_still_escaped_with_new_labels(tmp_path):
    c, _ = _make(tmp_path, desc="<script>alert(1)</script>")
    _post(c, "/scan")
    _post(c, f"/case/{_case_id(c)}/file")
    html = _post(c, "/cycle", proposer="heuristic").text
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)" in html


# --- proposed-change summary card ---
from harness_evolve.court_ui.app import describe_ops  # noqa: E402


def _one(op, manifest=None):
    r = describe_ops([op], manifest)
    assert len(r) == 1
    return r[0]


def test_describe_ops_every_op_type():
    r = _one({"op": "add_keyword", "level": "high", "keyword": "cannibalized"})
    assert r["group"] == "Risk rules" and r["text"] == 'Treat "cannibalized" as high-severity: send to manual review' and r["risk"] == "ok"
    assert _one({"op": "add_keyword", "level": "medium", "keyword": "no witnesses"})["text"] == \
        'Treat "no witnesses" as medium-severity: recommend review'
    r = _one({"op": "add_rule", "id": "r1", "text": "Escalate staged accidents"})
    assert r["group"] == "Risk rules" and "Escalate staged accidents" in r["text"] and r["risk"] == "ok"
    r = _one({"op": "remove_rule", "id": "r1"})
    assert "r1" in r["text"] and r["group"] == "Risk rules"
    r = _one({"op": "set_threshold", "name": "medium_score", "value": 0.55}, {"thresholds": {"medium_score": 0.5}})
    assert r["text"] == "Raise the review threshold for policy risk score from 0.50 to 0.55" and r["risk"] == "loosens"
    r = _one({"op": "set_threshold", "name": "medium_score", "value": 0.3}, {"thresholds": {"medium_score": 0.5}})
    assert r["text"].startswith("Lower the review threshold") and r["risk"] == "ok"
    r = _one({"op": "set_limit", "name": "max_llm_steps_per_turn", "value": 14}, {"limits": {"max_llm_steps_per_turn": 10}})
    assert r["group"] == "Agent limits" and r["text"] == "Allow up to 14 model steps per turn, up from 10" and r["risk"] == "loosens"
    r = _one({"op": "set_limit", "name": "max_tool_calls_per_turn", "value": 6}, {"limits": {"max_tool_calls_per_turn": 12}})
    assert r["text"] == "Cap at 6 tool calls per turn, down from 12" and r["risk"] == "ok"


def test_describe_ops_tools_grouped_by_intent_and_enable_loosens():
    tools = ["get_quote", "create_policy", "save_customer_info", "recall_customer_info", "send_notification"]
    r = describe_ops([{"op": "disable_tool", "intent": "general", "tool": t} for t in tools])
    assert len(r) == 1 and r[0]["text"] == "Stop offering 5 tools for general questions" and r[0]["group"] == "Tool access"
    assert r[0]["tools"] == sorted(tools) and r[0]["risk"] == "ok"
    r = describe_ops([{"op": "enable_tool", "intent": "claims", "tool": "get_quote"}])
    assert r[0]["text"] == "Start offering 1 tool for claims" and r[0]["risk"] == "loosens"


def test_describe_ops_default_baseline_and_groups_sorted():
    r = describe_ops([{"op": "set_limit", "name": "max_llm_steps_per_turn", "value": 99},
                      {"op": "set_threshold", "name": "high_score", "value": 0.9},
                      {"op": "disable_tool", "intent": "info", "tool": "get_quote"}])
    assert [d["group"] for d in r] == ["Risk rules", "Tool access", "Agent limits"]
    assert r[0]["risk"] == "loosens" and r[2]["risk"] == "loosens"  # above the shipped baseline


def test_describe_ops_unknown_and_malformed_never_raise():
    r = describe_ops([{"op": "format_disk", "x": "<b>"}, "junk", {"op": "add_keyword"}, None], None)
    assert len(r) == 4 and all(d["text"].startswith("Unknown change") for d in r)
    assert describe_ops(None) == [] and describe_ops("x") == []


def _good(acc, crit=0):
    return {"accuracy": acc, "critical_violations": crit, "n": 4, "over_escalation_rate": 0, "under_escalation_rate": 0}


def _render_with(tmp_path, monkeypatch, ops, rationale, risk="needs_approval", gate=None):
    import harness_evolve.court_ui.app as m
    c, _ = _make(tmp_path)
    rep = {"proposer": "heuristic+tool-heuristic", "status": "candidate", "version": 5, "risk": risk, "rationale": rationale,
           "ops": ops, "gate": gate or {"gate": "pass", "reasons": []}, "candidate": {"dev": _good(0.88), "heldout": _good(0.9)}}
    monkeypatch.setattr(m, "run_cycle", lambda store, proposer: rep)
    return _post(c, "/cycle", proposer="heuristic").text


def _card(html):
    return html.split('id="proposed"')[1].split('id="versions"')[0]


def test_card_renders_summary_with_friendly_fields(tmp_path):
    import re
    c, _ = _make(tmp_path)
    card = _card(_post(c, "/cycle", proposer="heuristic").text)
    assert "Proposed change v2" in card and "Awaiting approval" in card
    assert "Rule-based proposer" in card and "What would change" in card
    assert "Show the evidence" in card and "Technical details" in card
    assert "proposer heuristic" not in card and ", status " not in card
    assert re.search(r"Score impact:</b> Accuracy on known cases \d+% to \d+%, dangerous misses \d+ to \d+, safety check", card)
    assert 'class="badge b-active">Passed<' in card or 'class="badge b-rejected">Failed<' in card
    assert card.split("Show the evidence")[1].split("</details>")[0].count("<li>") >= 1


def test_card_badges_loosening_and_escaping(tmp_path, monkeypatch):
    ops = [{"op": "add_keyword", "level": "high", "keyword": "<script>alert(1)</script>"},
           {"op": "enable_tool", "intent": "claims", "tool": "get_quote"},
           {"op": "set_limit", "name": "max_llm_steps_per_turn", "value": 14},
           {"op": "weird<img src=x>"}, {"op": "zap", "evil": "<script>x</script>"}]
    html = _render_with(tmp_path, monkeypatch, ops, "first <b>point</b>; second point; third point")
    card = _card(html)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)" in card
    assert "<img src=x>" not in html and "<script>x</script>" not in html
    assert card.count("Loosens a safeguard") == 2 and "Needs your approval" in card and "Unknown change" in card
    ev = card.split("Show the evidence")[1].split("</details>")[0]
    assert ev.count("<li>") == 3  # one bullet per '; ' piece, not a single text node
    assert "first &lt;b&gt;point&lt;/b&gt;" in ev and "<b>point</b>" not in card


def test_card_low_risk_and_failed_gate_reasons(tmp_path, monkeypatch):
    gate = {"gate": "fail", "reasons": ["held-out accuracy dropped", "critical miss <i>x</i>"]}
    html = _render_with(tmp_path, monkeypatch, [{"op": "disable_tool", "intent": "general", "tool": "get_quote"}],
                        "ok", risk="low", gate=gate)
    card = _card(html)
    assert "Low risk" in card and "Loosens a safeguard" not in card
    assert 'class="badge b-rejected">Failed<' in card and "held-out accuracy dropped" in card
    assert "critical miss &lt;i&gt;x&lt;/i&gt;" in card


def test_card_approve_reject_keep_csrf_and_confirm(tmp_path):
    import re
    c, _ = _make(tmp_path)
    card = _card(_post(c, "/cycle", proposer="heuristic").text)
    forms = re.findall(r"<form .*?</form>", card, re.S)
    assert len(forms) == 2 and any("/approve" in f for f in forms) and any("/reject" in f for f in forms)
    for f in forms:
        assert f'name="csrf" value="{TOKEN}"' in f and 'name="confirm"' in f
        assert "I understand this changes live state" in f
    assert "confirm" in _post(c, "/version/2/approve").text.lower()


def test_version_header_never_wraps_css():
    from harness_evolve.court_ui import app as appmod
    assert "#trial .tw th:nth-child(2){white-space:nowrap" in appmod.TEMPLATE
