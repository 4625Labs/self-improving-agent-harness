"""FastAPI Court docket. Every mutation is a POST carrying a per-process random token.

All text is rendered through Jinja2 autoescape (no |safe anywhere). The model is only called by
the explicit `llm_call` action; the llm path otherwise shows a dry-run (prompt + token estimate).
"""

import copy
import json
import secrets
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import HTMLResponse, RedirectResponse  # noqa: E402
from jinja2 import DictLoader, Environment, select_autoescape  # noqa: E402
from markupsafe import Markup  # noqa: E402

from evals.run_eval import evaluate  # noqa: E402
from evals.tool_eval import evaluate_tools  # noqa: E402
from agent_agent46.policy_store import CLAIMS_COLLECTION  # noqa: E402
from harness_evolve import audit  # noqa: E402
from harness_evolve.court import VALID_RECOMMENDATIONS, CourtDocket  # noqa: E402
from harness_evolve.loop import approve, reject, rollback, run_cycle  # noqa: E402
from harness_evolve.proposer import (  # noqa: E402
    CompositeProposer,
    HeuristicProposer,
    LLMProposer,
    ToolHeuristicProposer,
)

DEFAULT_URI = "mongodb://localhost:53087/?directConnection=true"
DEFAULT_DB = "agent_agent46_local"

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Court docket</title>
<style>
body{font:14px/1.45 system-ui,sans-serif;margin:0;background:#f5f5f2;color:#222}
header{background:#1d2b3a;color:#fff;padding:12px 20px;display:flex;gap:24px;align-items:baseline}
header h1{font-size:18px;margin:0}.ver{font-size:20px;font-weight:700;background:#2f7d4f;padding:2px 10px;border-radius:4px}
main{padding:16px 20px}.cols{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;align-items:start}
section,.panel{background:#fff;border:1px solid #d8d8d0;border-radius:6px;padding:12px}
h2{font-size:15px;margin:0 0 8px}h3{font-size:13px;margin:10px 0 4px}
.case{border-top:1px solid #e4e4dc;padding:8px 0}.muted{color:#666}.flash{padding:8px 12px;border-radius:4px;margin-bottom:12px}
.ok{background:#e3f3e8}.err{background:#f9e0e0}
button{cursor:pointer;padding:4px 10px}input[type=text],select{padding:3px}
pre{background:#f0f0ea;padding:8px;overflow:auto;white-space:pre-wrap;max-height:260px}
table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:3px 8px;text-align:left}
form.inline{display:inline}.panel{margin-top:14px}
.confirm{font-size:12px;margin:0 6px;white-space:nowrap}.cap{color:#555;font-size:12px;margin:-4px 0 8px}.how{background:#eef3f8;border-color:#c9d6e3;margin:0 0 14px}
.how p{margin:4px 0}.err-inline{color:#a11;background:#f9e0e0;border-radius:4px;padding:3px 8px;margin:4px 0;font-size:13px}
.hide{display:none}details summary{cursor:pointer;color:#245}.reason{max-width:360px;overflow-wrap:anywhere}
.tw{overflow-x:auto;max-width:100%}.tw th,.tw td{white-space:nowrap}.tw td.reason{white-space:normal;min-width:200px}section,.panel,.case{min-width:0;overflow-wrap:anywhere}
@media (max-width:800px){.cols{grid-template-columns:1fr}header{flex-wrap:wrap;gap:8px}main{padding:12px}}
</style></head><body>
<header><h1>Court docket</h1>
<div>Active manifest <span class="ver">v{{ active.version if active else "?" }}</span></div></header>
<main>
{% for kind, msg in flashes %}<div class="flash {{ kind }}">{{ msg }}</div>{% endfor %}
<div class="panel" style="margin:0 0 14px">
<form method="post" action="/scan" class="inline">{{ tok() }}<button>Scan for disagreements</button></form>
&nbsp;
<form method="post" action="/nominate" class="inline">{{ tok() }}
<input type="text" name="claim_id" placeholder="claim_id" required>
<input type="text" name="reason" placeholder="why (optional)">
<button>Nominate claim_id</button></form></div>
<div class="panel how"><b>How this works.</b> Raindrop signals and human reviewers nominate claims where the agent may have been wrong.
A human adjuster rules on each case, and only convicted cases become regression test scenarios. The fixed evaluator and gate then decide
whether a proposed harness change is safe, and a human approves it before it goes live.</div>
<div class="cols">
<section><h2>Nominated ({{ nominated|length }})</h2><p class="cap">Flagged, not yet reviewed.</p>
{% for c in nominated %}<div class="case"><b>{{ c.claim_id }}</b> <span class="muted">[{{ c.source }}]</span><br>
{{ c.reason }}<br>
<form method="post" action="/case/{{ c.case_id }}/file" class="inline">{{ tok() }}<button>File case</button></form>
<form method="post" action="/case/{{ c.case_id }}/verdict" class="inline adj" novalidate>{{ tok() }}
<input type="hidden" name="action" value="dismiss"><input type="text" name="adjuster" placeholder="adjuster">
<label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label>
<button>Dismiss</button><div class="err-inline hide" role="alert">Adjuster name is required.</div></form></div>{% else %}<p class="muted">None.</p>{% endfor %}</section>

<section><h2>Filed ({{ filed|length }})</h2><p class="cap">Case file built, awaiting adjuster verdict.</p>
{% for c in filed %}<div class="case"><b>{{ c.claim_id }}</b> <span class="muted">case {{ c.case_id }} [{{ c.source }}]</span>
<h3>Case file</h3>
<table>
<tr><th>Claim</th><td>{{ c.draft.claim.description }}</td></tr>
<tr><th>Amount</th><td>{{ c.draft.claim.claim_amount }}</td></tr>
<tr><th>Type</th><td>{{ c.draft.claim.claim_type }}</td></tr>
<tr><th>Agent decided</th><td>{{ c.agent.recommendation }}</td></tr>
<tr><th>Risk assessment</th><td>{{ c.agent.risk_assessment }}</td></tr>
<tr><th>Harness version</th><td>{{ c.agent.harness_version }}</td></tr>
<tr><th>Reviewer resolution</th><td>{{ c.resolution }}</td></tr>
<tr><th>Why nominated</th><td>{{ c.reason }}</td></tr></table>
<h3>Verdict</h3>
<form method="post" action="/case/{{ c.case_id }}/verdict" class="adj" novalidate>{{ tok() }}
<input type="hidden" name="action" value="verdict">
<select name="expected"><option value="">expected...</option>
{% for r in recs %}<option value="{{ r }}">{{ r }}</option>{% endfor %}</select>
<label><input type="checkbox" name="critical" value="1"> Critical</label><br>
<input type="text" name="adjuster" placeholder="adjuster (required)"><br>
<label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label>
<button>Convict</button><div class="err-inline hide" role="alert">Adjuster name is required.</div></form>
<form method="post" action="/case/{{ c.case_id }}/verdict" class="adj" novalidate style="margin-top:6px">{{ tok() }}
<input type="hidden" name="action" value="dismiss">
<input type="text" name="adjuster" placeholder="adjuster (required)">
<label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label>
<button>Dismiss</button><div class="err-inline hide" role="alert">Adjuster name is required.</div></form></div>
{% else %}<p class="muted">None.</p>{% endfor %}</section>

<section><h2>Convicted / Dismissed ({{ closed|length }})</h2><p class="cap">Decided.</p>
{% for c in closed %}<div class="case"><b>{{ c.claim_id }}</b> {{ c.status }} {% if c.status == "convicted" %}: expected {{ c.expected_recommendation }}{% if c.critical %} (critical){% endif %}
by {{ c.verdict_by }}<br><span class="muted">Scenario written: {{ c.draft.id }} to docket.json (dev split)</span>
{% else %}by {{ c.dismissed_by }} {{ c.dismiss_note }}{% endif %}
{% if c.history %}<details><summary>History ({{ c.history|length }})</summary><ul>
{% for h in c.history %}<li>{{ h.ts }} {{ h.action }} by {{ h.by }}: was {{ h.previous_expected }}{% if h.previous_critical %} (critical){% endif %}. Reason: {{ h.reason }}</li>{% endfor %}</ul></details>{% endif %}
{% if c.status == "convicted" %}<h3>Overturn</h3>
<form method="post" action="/case/{{ c.case_id }}/overturn" class="adj" novalidate>{{ tok() }}
<select name="expected">{% for r in recs %}<option value="{{ r }}"{% if r == c.expected_recommendation %} selected{% endif %}>{{ r }}</option>{% endfor %}</select>
<label><input type="checkbox" name="critical" value="1"{% if c.critical %} checked{% endif %}> Critical</label><br>
<input type="text" name="adjuster" placeholder="adjuster (required)"><br>
<input type="text" name="reason" placeholder="reason (required)" required><br>
<label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label>
<button>Overturn</button><div class="err-inline hide" role="alert">Adjuster name is required.</div></form>
<h3>Retract</h3>
<form method="post" action="/case/{{ c.case_id }}/retract" class="adj" novalidate>{{ tok() }}
<input type="text" name="adjuster" placeholder="adjuster (required)"><br>
<input type="text" name="reason" placeholder="reason (required)" required><br>
<label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label>
<button>Retract</button><div class="err-inline hide" role="alert">Adjuster name is required.</div></form>{% endif %}</div>
{% else %}<p class="muted">None.</p>{% endfor %}</section>
</div>

<div class="panel"><h2>Trial (dev split, no LLM)</h2>
<div class="tw"><table><tr><th></th><th>version</th><th>dev accuracy</th><th>held-out accuracy</th><th>dev n</th><th>critical violations</th><th>over-esc.</th><th>under-esc.</th><th>gate</th></tr>
{% for label, ver, r, h, gate in trial %}<tr><td>{{ label }}</td><td>{{ ver }}</td><td>{{ "%.3f"|format(r.accuracy) }}</td><td>{{ "%.3f"|format(h.accuracy) }}</td><td>{{ r.n }}</td>
<td>{{ r.critical_violations }}</td><td>{{ "%.3f"|format(r.over_escalation_rate) }}</td><td>{{ "%.3f"|format(r.under_escalation_rate) }}</td><td>{{ gate }}</td></tr>{% endfor %}</table></div>
<h3>Run cycle</h3>
<form method="post" action="/cycle">{{ tok() }}
<select name="proposer"><option value="heuristic">heuristic (default)</option><option value="llm">llm (dry-run first)</option></select>
<button>Run cycle</button></form>
<p class="muted">The llm option only shows the prompt and a token estimate. The model is called only by the separate confirm button.</p>
{% if dry %}<h3>LLM dry-run (no model call made)</h3>
<p>model {{ dry.model }}, about <b>{{ dry.estimated_prompt_tokens }} prompt tokens</b>, max output {{ dry.max_output_tokens }}, cached {{ dry.cached }}, calls allowed {{ dry.calls_allowed }}</p>
<details><summary>Show the prompt that would be sent</summary><pre>{{ dry.prompt }}</pre></details>
<form method="post" action="/cycle">{{ tok() }}<input type="hidden" name="proposer" value="llm">
<button name="action" value="llm_call">Call the model</button></form>{% endif %}
{% if report %}<h3>Candidate</h3>
<p>proposer {{ report.proposer }}, status <b>{{ report.status }}</b>{% if report.version %}, version v{{ report.version }}{% endif %}, risk {{ report.risk }}</p>
<p>{{ report.rationale }}</p>
<p>Ops:</p><pre>{{ report.ops_json }}</pre>
{% if report.gate %}<p>Gate: <b>{{ report.gate.gate }}</b></p>
<ul>{% for r in report.gate.reasons %}<li>{{ r }}</li>{% endfor %}</ul>{% endif %}
{% if report.error %}<p class="err flash">{{ report.error }}</p>{% endif %}
{% if report.status == "candidate" %}
<form method="post" action="/version/{{ report.version }}/approve" class="inline">{{ tok() }}<input type="text" name="reviewer" placeholder="reviewer"><label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label><button>Approve</button></form>
<form method="post" action="/version/{{ report.version }}/reject" class="inline">{{ tok() }}<input type="text" name="reviewer" placeholder="reviewer"><label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label><button>Reject</button></form>{% endif %}{% endif %}
<h3>Versions</h3>
<div class="tw"><table><tr><th>v</th><th>status</th><th>author</th><th>reason</th><th></th></tr>
{% for v in versions %}<tr><td>{{ v.version }}</td><td>{{ v.status }}</td><td>{{ v.author }}</td><td class="reason">{% set rs = v.reason or "" %}{% if rs|length > 120 %}<details><summary>{{ rs[:120] }}...</summary>{{ rs }}</details>{% else %}{{ rs }}{% endif %}</td>
<td>{% if v.status == "retired" %}<form method="post" action="/version/{{ v.version }}/rollback" class="inline">{{ tok() }}<input type="text" name="reviewer" placeholder="reviewer"><label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label><button>Rollback to</button></form>{% endif %}</td></tr>{% endfor %}</table></div>
</div>
<div class="panel"><details><summary>Audit log (last {{ audit|length }})</summary>
{% if audit %}<div class="tw"><table><tr><th>time (UTC)</th><th>action</th><th>actor</th><th>target</th><th>outcome</th><th>detail</th></tr>
{% for r in audit %}<tr><td>{{ r.ts }}</td><td>{{ r.action }}</td><td>{{ r.actor }}</td><td>{{ r.target }}</td><td>{{ r.outcome }}</td><td class="reason">{{ r.detail }}</td></tr>{% endfor %}</table></div>
{% else %}<p class="muted">No audit rows yet.</p>{% endif %}</details></div>
<script>
document.querySelectorAll("form.adj").forEach(function(f){f.addEventListener("submit",function(e){
var a=f.querySelector("[name=adjuster]"),m=f.querySelector(".err-inline");
if(!a.value.trim()){e.preventDefault();m.classList.remove("hide");a.focus();}else{m.classList.add("hide");}});});
</script></main></body></html>"""


def _fmt_report(rep: dict) -> dict:
    out = {k: rep.get(k) for k in ("proposer", "status", "version", "risk", "rationale", "gate", "error", "candidate")}
    out["ops_json"] = json.dumps(rep.get("ops", []), indent=1)
    return out


def create_app(db, store, *, token: str | None = None, llm_factory: Callable[[], Any] | None = None,
               docket_path=None) -> FastAPI:
    """db: pymongo-like database; store: HarnessConfigStore-like. Both injectable for tests."""
    token = token or secrets.token_urlsafe(32)
    court = CourtDocket(db, docket_path) if docket_path else CourtDocket(db)
    llm_factory = llm_factory or LLMProposer
    env = Environment(loader=DictLoader({"index.html": TEMPLATE}), autoescape=select_autoescape(default=True))
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.token = token
    state: dict[str, Any] = {"flashes": [], "report": None, "dry": None}

    def tok() -> Markup:
        # Only raw HTML in the page: built from our own random token, never from user text.
        return Markup('<input type="hidden" name="csrf" value="{}">').format(token)

    async def form(request: Request) -> dict:
        data = await request.form()
        sent = str(data.get("csrf", ""))
        if not secrets.compare_digest(sent.encode(), token.encode()):
            raise HTTPException(403, "missing or invalid CSRF token")
        return {k: str(v) for k, v in data.items()}

    def flash(kind: str, msg: str) -> RedirectResponse:
        state["flashes"].append((kind, msg))
        return RedirectResponse("/", status_code=303)

    def guarded(fn: Callable[[], str]) -> RedirectResponse:
        try:
            return flash("ok", fn())
        except (ValueError, RuntimeError) as e:  # includes ConfirmationRequired
            return flash("err", str(e))

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        cases = court.list_cases()
        active = store.get_active()
        trial = []
        if active:
            from agent_agent46.harness_config import DEFAULT_MANIFEST
            trial = [("baseline", "-", evaluate(DEFAULT_MANIFEST, "dev"), evaluate(DEFAULT_MANIFEST, "heldout"), "-"),
                     ("active", f"v{active['version']}", evaluate(active["manifest"], "dev"),
                      evaluate(active["manifest"], "heldout"), "-")]
            rep_ = state["report"]
            if rep_ and rep_.get("version") and (rep_.get("candidate") or {}).get("dev"):
                cand = rep_["candidate"]
                gate = (rep_.get("gate") or {}).get("gate", "-")
                trial.append((f"candidate ({rep_['status']})", f"v{rep_['version']}", cand["dev"], cand["heldout"], gate))
        flashes, state["flashes"] = state["flashes"], []
        return env.get_template("index.html").render(
            active=active, flashes=flashes, trial=trial, versions=store.list_versions(),
            nominated=[c for c in cases if c["status"] == "nominated"],
            filed=[c for c in cases if c["status"] == "filed"],
            closed=[c for c in cases if c["status"] in ("convicted", "dismissed")],
            recs=sorted(VALID_RECOMMENDATIONS), audit=audit.recent(db, 10), report=state["report"], dry=state["dry"], tok=tok)

    @app.post("/scan")
    async def scan(request: Request):
        await form(request)
        return guarded(lambda: f"scan nominated {len(court.nominate_reviewer_disagreements())} new case(s)")

    @app.post("/nominate")
    async def nominate(request: Request):
        f = await form(request)
        cid = f.get("claim_id", "").strip()
        if not cid:
            return flash("err", "claim_id is required")
        if not db[CLAIMS_COLLECTION].find_one({"claim_id": cid.upper()}):
            return flash("err", f"claim not found: {cid.upper()}; nothing was nominated")
        return guarded(lambda: f"nominated {court.nominate(cid, 'manual', f.get('reason', ''))['claim_id']}")

    @app.post("/case/{case_id}/file")
    async def file_case(case_id: str, request: Request):
        await form(request)
        return guarded(lambda: f"filed case {court.file_case(case_id)['case_id']}")

    @app.post("/case/{case_id}/verdict")
    async def verdict(case_id: str, request: Request):
        f = await form(request)
        adjuster = f.get("adjuster", "").strip()
        if not adjuster:
            return flash("err", "adjuster name is required")
        confirmed = f.get("confirm") == "1"
        if f.get("action") == "dismiss":
            return guarded(lambda: f"dismissed case {court.dismiss(case_id, adjuster, confirmed=confirmed)['case_id']}")
        expected = f.get("expected", "")
        if expected not in VALID_RECOMMENDATIONS:
            return flash("err", f"expected must be one of {sorted(VALID_RECOMMENDATIONS)}")
        return guarded(lambda: "verdict recorded; scenario written: "
                       f"{court.verdict(case_id, expected, f.get('critical') == '1', adjuster, confirmed=confirmed)['draft']['id']}")

    @app.post("/case/{case_id}/overturn")
    async def overturn(case_id: str, request: Request):
        f = await form(request)
        adjuster, reason = f.get("adjuster", "").strip(), f.get("reason", "").strip()
        if not adjuster:
            return flash("err", "adjuster name is required")
        if not reason:
            return flash("err", "reason is required")
        expected = f.get("expected", "")
        if expected not in VALID_RECOMMENDATIONS:
            return flash("err", f"expected must be one of {sorted(VALID_RECOMMENDATIONS)}")
        return guarded(lambda: "verdict overturned; scenario updated: " + court.overturn(
            case_id, expected, f.get("critical") == "1", adjuster, reason,
            confirmed=f.get("confirm") == "1")["draft"]["id"])

    @app.post("/case/{case_id}/retract")
    async def retract(case_id: str, request: Request):
        f = await form(request)
        adjuster, reason = f.get("adjuster", "").strip(), f.get("reason", "").strip()
        if not adjuster:
            return flash("err", "adjuster name is required")
        if not reason:
            return flash("err", "reason is required")
        return guarded(lambda: "verdict retracted; scenario removed: " + court.retract(
            case_id, adjuster, reason, confirmed=f.get("confirm") == "1")["draft"]["id"])

    @app.post("/cycle")
    async def cycle(request: Request):
        f = await form(request)
        which, action = f.get("proposer", "heuristic"), f.get("action", "")
        state["dry"] = None
        if which not in ("heuristic", "llm"):
            return flash("err", "unknown proposer")
        if which == "heuristic":
            proposer = CompositeProposer([HeuristicProposer(), ToolHeuristicProposer()])
        else:
            proposer = llm_factory()
            if action != "llm_call":  # default: dry-run only, never a model call
                def dry() -> str:
                    active = store.get_active()
                    if active is None:
                        raise RuntimeError("no active manifest")
                    risk, tools = evaluate(active["manifest"], "dev"), evaluate_tools(active["manifest"], "dev")
                    state["dry"] = proposer.dry_run(
                        copy.deepcopy(active["manifest"]), risk["failures"],
                        {k: v for k, v in risk.items() if k != "failures"}, tool_failures=tools["failures"])
                    return "llm dry-run ready; no model call was made"
                return guarded(dry)

        def run() -> str:
            rep = run_cycle(store, proposer)
            state["report"] = _fmt_report(rep)
            return f"cycle finished: {rep['status']}"
        return guarded(run)

    @app.post("/version/{version}/approve")
    async def do_approve(version: int, request: Request):
        f = await form(request)
        confirmed, actor = f.get("confirm") == "1", f.get("reviewer", "").strip() or "ui-anonymous"

        def go() -> str:
            approve(store, version, confirmed=confirmed, actor=actor, audit_db=db)
            state["report"] = None
            return f"v{version} is now active"
        return guarded(go)

    @app.post("/version/{version}/reject")
    async def do_reject(version: int, request: Request):
        f = await form(request)
        confirmed, actor = f.get("confirm") == "1", f.get("reviewer", "").strip() or "ui-anonymous"

        def go() -> str:
            reject(store, version, confirmed=confirmed, actor=actor, audit_db=db)
            state["report"] = None
            return f"v{version} rejected"
        return guarded(go)

    @app.post("/version/{version}/rollback")
    async def do_rollback(version: int, request: Request):
        f = await form(request)
        confirmed, actor = f.get("confirm") == "1", f.get("reviewer", "").strip() or "ui-anonymous"

        def go() -> str:
            rollback(store, version, confirmed=confirmed, actor=actor, audit_db=db)
            return f"rolled back to v{version}"
        return guarded(go)

    return app


def build_default_app(uri: str, dbname: str) -> FastAPI:
    from pymongo import MongoClient

    from agent_agent46.harness_config import HarnessConfigStore

    store = HarnessConfigStore(uri, dbname)
    store.seed_if_empty()
    return create_app(MongoClient(uri, serverSelectionTimeoutMS=2000)[dbname], store)
