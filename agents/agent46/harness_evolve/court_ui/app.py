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

APP_NAME = "Regression Court"
APP_SUBTITLE = "Review agent failures, rule on them, ship safer policy"
DECIDED_PAGE_SIZE = 10

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{{ app_name }}</title>
<style>
*{box-sizing:border-box}
body{font:14px/1.45 system-ui,sans-serif;margin:0;background:#f5f5f2;color:#222;overflow-x:hidden}
header{background:#1d2b3a;color:#fff;padding:8px 20px;display:flex;gap:10px 18px;align-items:center;flex-wrap:wrap;position:sticky;top:0;z-index:30;box-shadow:0 1px 4px rgba(0,0,0,.25)}
.brand{display:flex;flex-direction:column;min-width:0;margin-right:auto}
.brand .t{display:flex;align-items:center;gap:8px}header h1{font-size:18px;margin:0}.sub{font-size:12px;color:#b9c6d3}
.ver{font-size:16px;font-weight:700;background:#2f7d4f;padding:2px 10px;border-radius:4px;white-space:nowrap}
.vlab{font-size:12px;color:#b9c6d3;margin-right:4px}
header button.scan{background:#fff;color:#1d2b3a;border:0;border-radius:4px;padding:5px 12px;font-weight:600}
main{padding:14px 20px;max-width:1400px;margin:0 auto}
section,.panel{background:#fff;border:1px solid #d8d8d0;border-radius:6px;padding:12px}
h2{font-size:15px;margin:0 0 8px;display:inline-block}h3{font-size:13px;margin:10px 0 4px}
.case{border-top:1px solid #e4e4dc;padding:8px 0}.muted{color:#666}.flash{padding:8px 12px;border-radius:4px;margin-bottom:12px}
.ok{background:#e3f3e8}.err{background:#f9e0e0}
button{cursor:pointer;padding:4px 10px}input[type=text],input[type=search],select{padding:3px;max-width:100%}
pre{background:#f0f0ea;padding:8px;overflow:auto;white-space:pre-wrap;max-height:260px}
table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:3px 8px;text-align:left}
form.inline{display:inline}.panel{margin-top:12px}
.confirm{font-size:12px;margin:0 6px;white-space:nowrap}.cap{color:#555;font-size:12px;margin:-4px 0 8px}.how{background:#eef3f8;border-color:#c9d6e3;margin:0 0 12px;padding:8px 12px}
.how p{margin:4px 0}.err-inline{color:#a11;background:#f9e0e0;border-radius:4px;padding:3px 8px;margin:4px 0;font-size:13px}
.hide,[hidden]{display:none!important}details summary{cursor:pointer;color:#245}.reason{max-width:360px;overflow-wrap:anywhere}
.tw{overflow-x:auto;max-width:100%}.tw th,.tw td{white-space:nowrap}.tw td.reason{white-space:normal;min-width:200px}section,.panel,.case{min-width:0;overflow-wrap:anywhere}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 12px;padding:8px 12px}
.tabs{display:flex;gap:4px;overflow-x:auto;border-bottom:2px solid #d8d8d0;margin-bottom:12px;-webkit-overflow-scrolling:touch}
.tab{flex:0 0 auto;background:none;border:0;border-bottom:3px solid transparent;margin-bottom:-2px;padding:8px 14px;font-size:14px;color:#456;white-space:nowrap}
.tab[aria-selected=true]{border-bottom-color:#1d2b3a;color:#1d2b3a;font-weight:700}
.cnt{display:inline-block;min-width:20px;text-align:center;background:#d8dde3;color:#223;border-radius:10px;padding:0 6px;font-size:12px;margin-left:4px}
.tab[aria-selected=true] .cnt{background:#1d2b3a;color:#fff}
.badge{display:inline-block;border-radius:10px;padding:0 8px;font-size:12px;font-weight:600;white-space:nowrap}
.b-convicted{background:#d4efdc;color:#145a32}.b-dismissed{background:#e2e2e2;color:#444}.b-critical{background:#f9d6d6;color:#8a1111}
.b-active{background:#d4efdc;color:#145a32}.b-retired{background:#e2e2e2;color:#444}.b-candidate{background:#fdf0c9;color:#7a5600}.b-rejected{background:#f9d6d6;color:#8a1111}
.chips{display:flex;gap:6px;flex-wrap:wrap}.chip{border:1px solid #b9c0c8;background:#fff;border-radius:14px;padding:2px 12px}
.chip[aria-pressed=true]{background:#1d2b3a;color:#fff;border-color:#1d2b3a}
.dtools{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:8px}.dtools input{flex:1 1 180px;min-width:0}
.note{max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap!important}
.expand{padding:0 8px;font-size:12px}.dbox{white-space:normal;position:sticky;left:0;width:min(860px,calc(100vw - 72px));padding:4px 0}
.dd td{background:#fafaf7;white-space:normal}.drow.open td{background:#f3f6fa}
.pos{position:relative}.hd{position:relative;display:flex;align-items:center;gap:6px;margin-bottom:8px}.hd h2{margin:0}
.si,.info{width:20px;height:20px;padding:0;border-radius:50%;border:1px solid #6b7b8c;background:#fff;color:#245;font:italic 700 12px/1 Georgia,serif;flex:0 0 auto}
.info{border-color:#b9c6d3;background:transparent;color:#fff;width:24px;height:24px;font-size:14px}
.si:focus-visible,.info:focus-visible,.tab:focus-visible,.chip:focus-visible{outline:2px solid #2d7ff9;outline-offset:2px}
.tipbody{display:none;position:absolute;left:0;top:100%;z-index:25;width:min(300px,calc(100vw - 48px));background:#1d2b3a;color:#fff;border-radius:6px;padding:8px 10px;font-size:12px;box-shadow:0 4px 12px rgba(0,0,0,.3)}
.si.on+.tipbody,.si:hover+.tipbody,.si:focus+.tipbody{display:block}
.pos>.si{position:absolute;right:12px;top:11px;z-index:2}.pos>.tipbody{left:auto;right:12px;top:36px}
dialog#info-dialog{border:0;border-radius:8px;padding:0;width:min(760px,calc(100vw - 24px));max-height:calc(100vh - 32px);box-shadow:0 8px 40px rgba(0,0,0,.4)}
dialog::backdrop{background:rgba(0,0,0,.45)}
.dlg-h{position:sticky;top:0;background:#1d2b3a;color:#fff;padding:10px 16px;display:flex;justify-content:space-between;align-items:center}.dlg-h h2{color:#fff;margin:0;font-size:16px}
.dlg-b{padding:4px 16px 16px;overflow:auto}.dlg-b h3{font-size:14px;margin:14px 0 4px}.dlg-b ol{margin:4px 0;padding-left:22px}.dlg-b dt{font-weight:700;margin-top:6px}.dlg-b dd{margin:0 0 0 12px;color:#333}
.dlg-b{max-height:calc(100vh - 90px)}
div.panel>details>summary{font-weight:700;font-size:15px;color:#222;padding-right:34px}
div.panel>details[open]>summary{margin-bottom:8px}
@media (max-width:800px){header{padding:8px 12px}main{padding:10px 12px}.bar input[type=text]{flex:1 1 120px}}
</style></head><body>
{% macro tip(id, label, text) %}<button type="button" class="si" aria-label="{{ label }}" aria-describedby="{{ id }}" title="{{ text }}">i</button><span class="tipbody" id="{{ id }}" role="tooltip">{{ text }}</span>{% endmacro %}
<header><div class="brand"><div class="t"><h1>{{ app_name }}</h1>
<button type="button" class="info" id="info-btn" aria-label="About {{ app_name }}: how it works and what every field means" aria-haspopup="dialog">i</button></div>
<span class="sub">{{ app_subtitle }}</span></div>
<div><span class="vlab">Active</span><span class="ver" title="Active manifest version">v{{ active.version if active else "?" }}</span></div>
<form method="post" action="/scan" class="inline">{{ tok() }}<button class="scan" title="Scan for disagreements" aria-label="Scan for disagreements">Scan</button></form></header>
<dialog id="info-dialog" aria-labelledby="info-title"><div class="dlg-h"><h2 id="info-title">How {{ app_name }} works</h2><button type="button" id="info-close" aria-label="Close help">Close</button></div>
<div class="dlg-b">
<p>{{ app_subtitle }}. A human adjuster rules on cases where the agent may have been wrong; only convicted cases become regression tests, and a human approves every policy change before it goes live.</p>
<h3>The 9-step loop</h3>
<ol><li><b>Nominate</b>: a claim is flagged (by a scan, Raindrop signal, or by hand) as one the agent may have handled wrongly.</li>
<li><b>File</b>: build the case file (claim, agent decision, reviewer resolution) so the adjuster can read it.</li>
<li><b>Verdict</b>: the adjuster convicts (states the correct outcome) or dismisses (the agent was fine).</li>
<li><b>Scenario in dev split</b>: a conviction writes a regression scenario into the dev split of the scenario docket.</li>
<li><b>Cycle</b>: run a proposer to suggest a harness change and score it against the scenarios.</li>
<li><b>Gate</b>: the fixed evaluator and gate decide whether the candidate is safe to consider.</li>
<li><b>Read the diff</b>: a human reads the proposed changes and the Trial numbers.</li>
<li><b>Approve</b>: a human approves (or rejects) the candidate version.</li>
<li><b>Prove live</b>: watch the active version on real claims and roll back if it misbehaves.</li></ol>
<h3>Glossary of fields</h3><dl>
<dt>Verdict dropdown</dt><dd>The outcome the agent should have given for the claim (the expected recommendation in the new test).</dd>
<dt>Critical</dt><dd>Tick when a wrong answer on this case is unacceptable; the scenario is marked critical and the gate blocks any version that auto-approves it.</dd>
<dt>adjuster</dt><dd>The name of the human making the ruling. It is required and is written to the audit log.</dd>
<dt>Convict</dt><dd>Rule that the agent was wrong. Writes a regression scenario to the dev split.</dd>
<dt>Dismiss</dt><dd>Rule that the agent was not wrong. No scenario is written.</dd>
<dt>Overturn</dt><dd>Change the expected outcome or critical flag of an existing conviction, and update its scenario. A reason is required.</dd>
<dt>Retract</dt><dd>Withdraw a conviction and remove its scenario from the dev split. A reason is required.</dd>
<dt>reason</dt><dd>A short written justification for an overturn or retract; kept in the case History and audit log.</dd>
<dt>confirm box</dt><dd>The tick box saying you understand the action changes live state. Nothing is changed unless it is ticked.</dd>
<dt>History</dt><dd>The list of past overturns and retractions for a case: when, by whom, what it was before, and why.</dd></dl>
<h3>Trial table columns</h3><dl>
<dt>dev accuracy</dt><dd>Share of dev-split scenarios where the version gives the expected outcome.</dd>
<dt>held-out accuracy</dt><dd>The same measure on scenarios kept out of tuning, to check the change is not overfitted.</dd>
<dt>dev n</dt><dd>How many dev-split scenarios were scored.</dd>
<dt>critical violations</dt><dd>Critical scenarios that were auto-approved. This should be zero.</dd>
<dt>over-esc.</dt><dd>Over-escalation rate: expected-auto claims sent to a human anyway.</dd>
<dt>under-esc.</dt><dd>Under-escalation rate: claims that needed escalation but got a lighter outcome.</dd>
<dt>gate</dt><dd>The gate result for a proposed version: pass or fail, with reasons listed below the table.</dd></dl>
<h3>Versions table</h3><dl>
<dt>v</dt><dd>The harness version number.</dd>
<dt>status</dt><dd><b>active</b> is live now; <b>retired</b> was live before and can be rolled back to; <b>candidate</b> is proposed and awaiting approval; <b>rejected</b> was declined.</dd>
<dt>author</dt><dd>Who or what proposed the version.</dd>
<dt>reason</dt><dd>Why the version was proposed.</dd>
<dt>Rollback to</dt><dd>Make a retired version the active one again (needs the confirm box).</dd></dl>
</div></dialog>
<main>
{% for kind, msg in flashes %}<div class="flash {{ kind }}">{{ msg }}</div>{% endfor %}
<div class="panel bar" style="margin-top:0">
<form method="post" action="/nominate" class="inline">{{ tok() }}
<input type="text" name="claim_id" placeholder="claim_id" required>
<input type="text" name="reason" placeholder="why (optional)">
<button>Nominate claim_id</button></form></div>
<div class="panel how"><b>How this works.</b> Raindrop signals and human reviewers nominate claims where the agent may have been wrong.
A human adjuster rules on each case, and only convicted cases become regression test scenarios. The fixed evaluator and gate then decide
whether a proposed harness change is safe, and a human approves it before it goes live.</div>

<div class="tabs" role="tablist" id="tabs" data-default="{{ 'filed' if filed else 'nominated' }}">
<button type="button" class="tab" role="tab" id="t-nominated" data-tab="nominated" aria-controls="tab-nominated" aria-selected="false">Nominated<span class="cnt">{{ nominated|length }}</span></button>
<button type="button" class="tab" role="tab" id="t-filed" data-tab="filed" aria-controls="tab-filed" aria-selected="false">Filed<span class="cnt">{{ filed|length }}</span></button>
<button type="button" class="tab" role="tab" id="t-decided" data-tab="decided" aria-controls="tab-decided" aria-selected="false">Decided<span class="cnt">{{ closed|length }}</span></button></div>

<section id="tab-nominated" role="tabpanel" aria-labelledby="t-nominated"><div class="hd"><h2>Nominated ({{ nominated|length }})</h2>{{ tip("tip-nom", "About Nominated", "Claims flagged as possibly mishandled by the agent, waiting for someone to file a case file.") }}</div><p class="cap">Flagged, not yet reviewed.</p>
{% for c in nominated %}<div class="case"><b>{{ c.claim_id }}</b> <span class="muted">[{{ c.source }}]</span><br>
{{ c.reason }}<br>
<form method="post" action="/case/{{ c.case_id }}/file" class="inline">{{ tok() }}<button>File case</button></form>
<form method="post" action="/case/{{ c.case_id }}/verdict" class="inline adj" novalidate>{{ tok() }}
<input type="hidden" name="action" value="dismiss"><input type="text" name="adjuster" placeholder="adjuster">
<label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label>
<button>Dismiss</button><div class="err-inline hide" role="alert">Adjuster name is required.</div></form></div>{% else %}<p class="muted">None.</p>{% endfor %}</section>

<section id="tab-filed" role="tabpanel" aria-labelledby="t-filed"><div class="hd"><h2>Filed ({{ filed|length }})</h2>{{ tip("tip-filed", "About Filed", "Cases with a case file built, waiting for an adjuster to convict or dismiss.") }}</div><p class="cap">Case file built, awaiting adjuster verdict.</p>
{% for c in filed %}<div class="case"><b>{{ c.claim_id }}</b> <span class="muted">case {{ c.case_id }} [{{ c.source }}]</span>
<h3>Case file</h3>
<div class="tw"><table>
<tr><th>Claim</th><td style="white-space:normal">{{ c.draft.claim.description }}</td></tr>
<tr><th>Amount</th><td>{{ c.draft.claim.claim_amount }}</td></tr>
<tr><th>Type</th><td>{{ c.draft.claim.claim_type }}</td></tr>
<tr><th>Agent decided</th><td>{{ c.agent.recommendation }}</td></tr>
<tr><th>Risk assessment</th><td>{{ c.agent.risk_assessment }}</td></tr>
<tr><th>Harness version</th><td>{{ c.agent.harness_version }}</td></tr>
<tr><th>Reviewer resolution</th><td>{{ c.resolution }}</td></tr>
<tr><th>Why nominated</th><td style="white-space:normal">{{ c.reason }}</td></tr></table></div>
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

<section id="tab-decided" role="tabpanel" aria-labelledby="t-decided"><div class="hd"><h2>Convicted / Dismissed ({{ closed|length }})</h2>{{ tip("tip-dec", "About Convicted and Dismissed", "Cases an adjuster has ruled on. Convicted cases became regression scenarios; dismissed cases did not. Open a row to see History, Overturn and Retract.") }}</div>
{% if closed %}<div class="dtools"><div class="chips" role="group" aria-label="Filter decided cases">
<button type="button" class="chip" data-chip="all" aria-pressed="true">All</button>
<button type="button" class="chip" data-chip="convicted" aria-pressed="false">Convicted</button>
<button type="button" class="chip" data-chip="dismissed" aria-pressed="false">Dismissed</button></div>
<input type="search" id="decided-search" placeholder="Search claim id or text" aria-label="Search decided cases by claim id or text" autocomplete="off"></div>
<div class="tw"><table id="decided-table"><thead><tr><th>claim</th><th>status</th><th>expected</th><th>critical</th><th>adjuster</th><th>date</th><th>reason / note</th><th></th></tr></thead><tbody>
{% for c in closed %}<tr class="drow" data-status="{{ c.status }}" data-text="{{ (c.claim_id ~ ' ' ~ c.status ~ ' ' ~ c.who ~ ' ' ~ c.expected_recommendation ~ ' ' ~ c.note)|lower }}"{% if loop.index > page_size %} hidden{% endif %}>
<td><b>{{ c.claim_id }}</b></td><td><span class="badge b-{{ c.status }}">{{ c.status }}</span></td>
<td>{{ c.expected_recommendation or "-" }}</td><td>{% if c.critical %}<span class="badge b-critical">critical</span>{% else %}-{% endif %}</td>
<td>{{ c.who }}</td><td>{{ c.when }}</td><td class="note" title="{{ c.note }}">{{ c.note }}</td>
<td><button type="button" class="expand" aria-expanded="false" aria-controls="dd-{{ c.case_id }}" aria-label="Details for {{ c.claim_id }}">Details</button></td></tr>
<tr class="dd" id="dd-{{ c.case_id }}" hidden><td colspan="8"><div class="dbox">
{% if c.status == "convicted" %}<span class="muted">Scenario written: {{ c.draft.id }} to docket.json (dev split)</span>{% endif %}
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
<button>Retract</button><div class="err-inline hide" role="alert">Adjuster name is required.</div></form>{% endif %}
</div></td></tr>
{% endfor %}</tbody></table></div>
<p class="muted" id="decided-empty" hidden>No decided cases match.</p>
<p><button type="button" id="show-more" hidden>Show more</button> <span class="muted" id="decided-count"></span></p>
{% else %}<p class="muted">None.</p>{% endif %}</section>

<div class="panel pos" id="trial">{{ tip("tip-trial", "About Trial", "Scores the baseline, the active version and any candidate on the dev and held-out scenarios; the gate decides if a candidate is safe.") }}<details open><summary>Trial (dev split, no LLM)</summary>
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
</details></div>

<div class="panel pos" id="versions">{{ tip("tip-ver", "About Versions", "Every harness version: active is live, retired can be rolled back to, candidate awaits approval, rejected was declined.") }}<details><summary>Versions ({{ versions|length }})</summary>
<div class="tw"><table><tr><th>v</th><th>status</th><th>author</th><th>reason</th><th></th></tr>
{% for v in versions %}<tr><td>{{ v.version }}</td><td><span class="badge b-{{ v.status }}">{{ v.status }}</span></td><td>{{ v.author }}</td><td class="reason">{% set rs = v.reason or "" %}{% if rs|length > 120 %}<details><summary>{{ rs[:120] }}...</summary>{{ rs }}</details>{% else %}{{ rs }}{% endif %}</td>
<td>{% if v.status == "retired" %}<form method="post" action="/version/{{ v.version }}/rollback" class="inline">{{ tok() }}<input type="text" name="reviewer" placeholder="reviewer"><label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label><button>Rollback to</button></form>{% endif %}</td></tr>{% endfor %}</table></div>
</details></div>

<div class="panel pos" id="audit">{{ tip("tip-aud", "About Audit", "A record of every action taken here: when, who, what it targeted and whether it succeeded.") }}<details><summary>Audit log (last {{ audit|length }})</summary>
{% if audit %}<div class="tw"><table><tr><th>time (UTC)</th><th>action</th><th>actor</th><th>target</th><th>outcome</th><th>detail</th></tr>
{% for r in audit %}<tr><td>{{ r.ts }}</td><td>{{ r.action }}</td><td>{{ r.actor }}</td><td>{{ r.target }}</td><td>{{ r.outcome }}</td><td class="reason">{{ r.detail }}</td></tr>{% endfor %}</table></div>
{% else %}<p class="muted">No audit rows yet.</p>{% endif %}</details></div>
<script>
document.querySelectorAll("form.adj").forEach(function(f){f.addEventListener("submit",function(e){
var a=f.querySelector("[name=adjuster]"),m=f.querySelector(".err-inline");
if(!a.value.trim()){e.preventDefault();m.classList.remove("hide");a.focus();}else{m.classList.add("hide");}});});
(function(){
var $=function(s,r){return (r||document).querySelector(s)},$$=function(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s))};
// tabs
var tabs=$$(".tab"),names=tabs.map(function(t){return t.dataset.tab});
function store(v){try{localStorage.setItem("court.tab",v)}catch(e){}}
function saved(){try{return localStorage.getItem("court.tab")}catch(e){return null}}
function show(n,persist){if(names.indexOf(n)<0)n=$("#tabs").dataset["default"];
tabs.forEach(function(t){t.setAttribute("aria-selected",String(t.dataset.tab===n))});
names.forEach(function(x){$("#tab-"+x).hidden=(x!==n)});
if(persist){store(n);try{history.replaceState(null,"","#"+n)}catch(e){}}}
tabs.forEach(function(t){t.addEventListener("click",function(){show(t.dataset.tab,true)})});
var h=(location.hash||"").slice(1);show(names.indexOf(h)>=0?h:(names.indexOf(saved())>=0?saved():$("#tabs").dataset["default"]),false);
// decided table
var rows=$$("#decided-table tr.drow"),st={chip:"all",q:"",limit:{{ page_size }}};
function apply(){var shown=0,match=0;rows.forEach(function(r){
var ok=(st.chip==="all"||r.dataset.status===st.chip)&&(!st.q||r.dataset.text.indexOf(st.q)>=0),vis=false;
if(ok){match++;if(shown<st.limit){vis=true;shown++}}
r.hidden=!vis;var d=r.nextElementSibling;d.hidden=!(vis&&r.classList.contains("open"));});
var more=$("#show-more");if(!more)return;more.hidden=match<=shown;more.textContent="Show more ("+(match-shown)+" left)";
$("#decided-empty").hidden=match>0;$("#decided-count").textContent="Showing "+shown+" of "+match;}
if(rows.length){
$$(".chip").forEach(function(c){c.addEventListener("click",function(){st.chip=c.dataset.chip;st.limit={{ page_size }};
$$(".chip").forEach(function(o){o.setAttribute("aria-pressed",String(o===c))});apply()})});
$("#decided-search").addEventListener("input",function(){st.q=this.value.trim().toLowerCase();st.limit={{ page_size }};apply()});
$("#show-more").addEventListener("click",function(){st.limit+={{ page_size }};apply()});
$$("button.expand").forEach(function(b){b.addEventListener("click",function(){var r=b.closest("tr"),o=!r.classList.contains("open");
r.classList.toggle("open",o);b.setAttribute("aria-expanded",String(o));b.textContent=o?"Hide":"Details";apply()})});
apply();}
// info dialog
var dlg=$("#info-dialog");
$("#info-btn").addEventListener("click",function(){if(dlg.showModal){dlg.showModal()}else{dlg.setAttribute("open","")}});
$("#info-close").addEventListener("click",function(){if(dlg.close){dlg.close()}else{dlg.removeAttribute("open")}});
dlg.addEventListener("click",function(e){if(e.target===dlg&&dlg.close)dlg.close()});
// section tips (click/tap toggles; hover and focus also show)
$$(".si").forEach(function(b){b.addEventListener("click",function(e){e.stopPropagation();var o=!b.classList.contains("on");
$$(".si.on").forEach(function(x){x.classList.remove("on")});b.classList.toggle("on",o)})});
document.addEventListener("click",function(){$$(".si.on").forEach(function(x){x.classList.remove("on")})});
document.addEventListener("keydown",function(e){if(e.key==="Escape")$$(".si.on").forEach(function(x){x.classList.remove("on")})});
})();
</script></main></body></html>"""


def _fmt_report(rep: dict) -> dict:
    out = {k: rep.get(k) for k in ("proposer", "status", "version", "risk", "rationale", "gate", "error", "candidate")}
    out["ops_json"] = json.dumps(rep.get("ops", []), indent=1)
    return out


def _decided_view(c: dict) -> dict:
    """Adds display-only fields (who / when / one-line note) for the compact Decided table."""
    convicted = c["status"] == "convicted"
    when = str((c.get("verdict_at") if convicted else c.get("dismissed_at")) or c.get("created_at") or "")
    history = c.get("history") or []
    if convicted:
        note = (history[-1].get("reason") if history else "") or f"scenario {(c.get('draft') or {}).get('id', '')} (dev split)"
    else:
        note = c.get("dismiss_note") or "-"
    return {**c, "who": (c.get("verdict_by") if convicted else c.get("dismissed_by")) or "-",
            "when": when[:10] or "-", "when_full": when, "note": str(note)}


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
        closed = [_decided_view(c) for c in cases if c["status"] in ("convicted", "dismissed")]
        closed.sort(key=lambda c: c["when_full"], reverse=True)
        return env.get_template("index.html").render(
            active=active, flashes=flashes, trial=trial, versions=store.list_versions(),
            nominated=[c for c in cases if c["status"] == "nominated"],
            filed=[c for c in cases if c["status"] == "filed"],
            closed=closed, app_name=APP_NAME, app_subtitle=APP_SUBTITLE, page_size=DECIDED_PAGE_SIZE,
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
