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
from harness_evolve.court import VALID_RECOMMENDATIONS, CourtDocket  # noqa: E402
from harness_evolve.loop import approve, rollback, run_cycle  # noqa: E402
from harness_evolve.proposer import (  # noqa: E402
    CompositeProposer,
    HeuristicProposer,
    LLMProposer,
    ToolHeuristicProposer,
)

DEFAULT_URI = "mongodb://localhost:53087/?directConnection=true"
DEFAULT_DB = "agent_agent46_local"

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Court docket</title>
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
<div class="cols">
<section><h2>Nominated ({{ nominated|length }})</h2>
{% for c in nominated %}<div class="case"><b>{{ c.claim_id }}</b> <span class="muted">[{{ c.source }}]</span><br>
{{ c.reason }}<br>
<form method="post" action="/case/{{ c.case_id }}/file" class="inline">{{ tok() }}<button>File case</button></form>
<form method="post" action="/case/{{ c.case_id }}/verdict" class="inline">{{ tok() }}
<input type="hidden" name="action" value="dismiss"><input type="text" name="adjuster" placeholder="adjuster" required>
<button>Dismiss</button></form></div>{% else %}<p class="muted">None.</p>{% endfor %}</section>

<section><h2>Filed ({{ filed|length }})</h2>
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
<form method="post" action="/case/{{ c.case_id }}/verdict">{{ tok() }}
<select name="expected" required><option value="">expected...</option>
{% for r in recs %}<option value="{{ r }}">{{ r }}</option>{% endfor %}</select>
<label><input type="checkbox" name="critical" value="1"> Critical</label><br>
<input type="text" name="adjuster" placeholder="adjuster (required)" required>
<button name="action" value="verdict">Convict</button>
<button name="action" value="dismiss">Dismiss</button></form></div>
{% else %}<p class="muted">None.</p>{% endfor %}</section>

<section><h2>Convicted / Dismissed ({{ closed|length }})</h2>
{% for c in closed %}<div class="case"><b>{{ c.claim_id }}</b> {{ c.status }}
{% if c.status == "convicted" %}: expected {{ c.expected_recommendation }}{% if c.critical %} (critical){% endif %}
by {{ c.verdict_by }}<br><span class="muted">Scenario written: {{ c.draft.id }} to docket.json (dev split)</span>
{% else %}by {{ c.dismissed_by }} {{ c.dismiss_note }}{% endif %}</div>
{% else %}<p class="muted">None.</p>{% endfor %}</section>
</div>

<div class="panel"><h2>Trial (dev split, no LLM)</h2>
<table><tr><th></th><th>version</th><th>accuracy</th><th>n</th><th>critical violations</th><th>over-esc.</th><th>under-esc.</th></tr>
{% for label, ver, r in trial %}<tr><td>{{ label }}</td><td>{{ ver }}</td><td>{{ "%.3f"|format(r.accuracy) }}</td><td>{{ r.n }}</td>
<td>{{ r.critical_violations }}</td><td>{{ "%.3f"|format(r.over_escalation_rate) }}</td><td>{{ "%.3f"|format(r.under_escalation_rate) }}</td></tr>{% endfor %}</table>
<h3>Run cycle</h3>
<form method="post" action="/cycle">{{ tok() }}
<select name="proposer"><option value="heuristic">heuristic (default)</option><option value="llm">llm (dry-run first)</option></select>
<button>Run cycle</button></form>
<p class="muted">The llm option only shows the prompt and a token estimate. The model is called only by the separate confirm button.</p>
{% if dry %}<h3>LLM dry-run (no model call made)</h3>
<p>model {{ dry.model }}, about {{ dry.estimated_prompt_tokens }} prompt tokens, max output {{ dry.max_output_tokens }}, cached {{ dry.cached }}, calls allowed {{ dry.calls_allowed }}</p>
<pre>{{ dry.prompt }}</pre>
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
<form method="post" action="/version/{{ report.version }}/approve" class="inline">{{ tok() }}<button>Approve</button></form>
<form method="post" action="/version/{{ report.version }}/reject" class="inline">{{ tok() }}<button>Reject</button></form>{% endif %}{% endif %}
<h3>Versions</h3>
<table><tr><th>v</th><th>status</th><th>author</th><th>reason</th><th></th></tr>
{% for v in versions %}<tr><td>{{ v.version }}</td><td>{{ v.status }}</td><td>{{ v.author }}</td><td>{{ v.reason }}</td>
<td>{% if v.status == "retired" %}<form method="post" action="/version/{{ v.version }}/rollback" class="inline">{{ tok() }}<button>Rollback to</button></form>{% endif %}</td></tr>{% endfor %}</table>
</div></main></body></html>"""


def _fmt_report(rep: dict) -> dict:
    out = {k: rep.get(k) for k in ("proposer", "status", "version", "risk", "rationale", "gate", "error")}
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
        except (ValueError, RuntimeError) as e:
            return flash("err", str(e))

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        cases = court.list_cases()
        active = store.get_active()
        trial = []
        if active:
            from agent_agent46.harness_config import DEFAULT_MANIFEST
            trial = [("baseline", "-", evaluate(DEFAULT_MANIFEST, "dev")),
                     ("active", f"v{active['version']}", evaluate(active["manifest"], "dev"))]
        flashes, state["flashes"] = state["flashes"], []
        return env.get_template("index.html").render(
            active=active, flashes=flashes, trial=trial, versions=store.list_versions(),
            nominated=[c for c in cases if c["status"] == "nominated"],
            filed=[c for c in cases if c["status"] == "filed"],
            closed=[c for c in cases if c["status"] in ("convicted", "dismissed")],
            recs=sorted(VALID_RECOMMENDATIONS), report=state["report"], dry=state["dry"], tok=tok)

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
        if f.get("action") == "dismiss":
            return guarded(lambda: f"dismissed case {court.dismiss(case_id, adjuster)['case_id']}")
        expected = f.get("expected", "")
        if expected not in VALID_RECOMMENDATIONS:
            return flash("err", f"expected must be one of {sorted(VALID_RECOMMENDATIONS)}")
        return guarded(lambda: "verdict recorded; scenario written: "
                       f"{court.verdict(case_id, expected, f.get('critical') == '1', adjuster)['draft']['id']}")

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
        await form(request)

        def go() -> str:
            approve(store, version)
            state["report"] = None
            return f"v{version} is now active"
        return guarded(go)

    @app.post("/version/{version}/reject")
    async def do_reject(version: int, request: Request):
        await form(request)

        def go() -> str:
            doc = store.get(version)
            if not doc or doc["status"] != "candidate":
                raise ValueError(f"version {version} is not a candidate")
            store.set_status(version, "rejected")
            state["report"] = None
            return f"v{version} rejected"
        return guarded(go)

    @app.post("/version/{version}/rollback")
    async def do_rollback(version: int, request: Request):
        await form(request)

        def go() -> str:
            rollback(store, version)
            return f"rolled back to v{version}"
        return guarded(go)

    return app


def build_default_app(uri: str, dbname: str) -> FastAPI:
    from pymongo import MongoClient

    from agent_agent46.harness_config import HarnessConfigStore

    store = HarnessConfigStore(uri, dbname)
    store.seed_if_empty()
    return create_app(MongoClient(uri, serverSelectionTimeoutMS=2000)[dbname], store)
