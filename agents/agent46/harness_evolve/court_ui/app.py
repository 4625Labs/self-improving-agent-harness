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

# Every user-facing label on the page lives here so wording is trivially changeable.
# Backend values (routes, form field names, status strings) are never taken from this dict.
UI_LABELS: dict[str, Any] = {
    "live_policy": "Live policy",
    "scan": "Scan",
    "scan_full": "Scan for disagreements",
    "scorecard": "Policy scorecard",
    "scorecard_sub": "Replays saved test cases through each policy version. Instant, free, no AI calls.",
    "row_baseline": "Original rules",
    "row_active": "Live policy",
    "row_candidate": "Proposed change (awaiting approval)",
    "col_version": "Version",
    "col_dev_acc": "Accuracy on known cases",
    "col_heldout_acc": "Accuracy on unseen cases",
    "col_cases": "Cases tested",
    "col_critical": "Dangerous misses",
    "col_over": "Unneeded reviews",
    "col_under": "Missed escalations",
    "col_gate": "Safety check",
    "tip_dev_acc": "Share of saved test cases where this policy gives the answer the adjuster expects.",
    "tip_heldout_acc": "The same score on cases kept out of tuning, to check the change is not overfitted.",
    "tip_cases": "How many saved test cases were replayed.",
    "tip_critical": "A dangerous miss is a case marked critical that the policy would auto-approve. Should be zero.",
    "tip_over": "Share of cases that could be auto-approved but were sent to a human anyway.",
    "tip_under": "Share of cases that needed escalation but got a lighter outcome.",
    "tip_gate": "Passed means the proposed change is safe to consider; Failed lists the reasons below.",
    "gate_pass": "Passed",
    "gate_fail": "Failed",
    "propose_title": "Propose a policy change",
    "propose_button": "Propose change",
    "proposer_heuristic": "Rule-based (free)",
    "proposer_llm": "AI model (uses API budget)",
    "propose_note": "The AI model option only shows a preview (no call made). The model is called only by the separate Ask the AI model now button.",
    "dry_title": "AI model preview (no call made)",
    "dry_text": "Preview (no call made)",
    "dry_prompt": "Show the prompt that would be sent",
    "ask_llm": "Ask the AI model now",
    "proposed_title": "Proposed change",
    "approve": "Approve and go live",
    "reject": "Reject",
    "versions": "Policy versions",
    "ver_col_version": "Version",
    "ver_col_status": "Status",
    "ver_col_author": "Proposed by",
    "ver_col_reason": "What changed",
    "ver_col_actions": "Actions",
    "make_live": "Make live again",
    "activity_log": "Activity log",
    "status": {"active": "Live", "retired": "Previous", "candidate": "Awaiting approval", "rejected": "Rejected"},
    "author": {"human": "Original", "proposer:heuristic": "Rule-based proposer", "proposer:llm": "AI model",
               "proposer:tool-heuristic": "Tool-access proposer"},
    "author_limits": "Limits proposer",
    "tab_nominated": "Nominated", "tab_filed": "Filed", "tab_decided": "Decided",
    "cap_nominated": "Flagged as possibly mishandled; no case file yet.",
    "cap_filed": "Case file built; waiting for an adjuster to convict or dismiss.",
    "cap_decided": "Ruled on by an adjuster. Convicted cases became saved test cases; dismissed ones did not.",
    "convict": "Convict", "dismiss": "Dismiss", "overturn": "Overturn", "retract": "Retract",
    "tip_convict": "Rule that the agent was wrong; saves this case as a test case.",
    "tip_dismiss": "Rule that the agent was fine; no test case is saved.",
    "tip_overturn": "Change the expected answer or critical flag of an earlier conviction.",
    "tip_retract": "Withdraw an earlier conviction and remove its test case.",
    "critical_badge": "critical",
    "tip_critical_badge": "must never be auto-approved",
}


def _author_label(author: str) -> str:
    """Friendly name for a version author; composite proposers ('a+b') list each member once."""
    names = UI_LABELS["author"]
    if author in names:
        return names[author]
    if author.startswith("proposer:"):
        parts = []
        for m in author[len("proposer:"):].split("+"):
            n = names.get("proposer:" + m) or (UI_LABELS["author_limits"] if m == "limits-heuristic" else m)
            if n not in parts:
                parts.append(n)
        return ", ".join(parts)
    return author


def _pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.0f}%"
    except (TypeError, ValueError):
        return "-"

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
.tw{overflow-x:auto;max-width:100%}.tw th,.tw td{white-space:nowrap}#trial .tw th{white-space:normal;vertical-align:bottom;min-width:76px}.tw td.reason{white-space:normal;min-width:200px}section,.panel,.case{min-width:0;overflow-wrap:anywhere}
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
<div><span class="vlab">{{ L.live_policy }}</span><span class="ver" title="{{ L.live_policy }} version">v{{ active.version if active else "?" }}</span></div>
<form method="post" action="/scan" class="inline">{{ tok() }}<button class="scan" title="{{ L.scan_full }}: find claims where the reviewer disagreed with the agent" aria-label="{{ L.scan_full }}">{{ L.scan }}</button></form></header>
<dialog id="info-dialog" aria-labelledby="info-title"><div class="dlg-h"><h2 id="info-title">How {{ app_name }} works</h2><button type="button" id="info-close" aria-label="Close help">Close</button></div>
<div class="dlg-b">
<p>{{ app_subtitle }}. A human adjuster rules on cases where the agent may have been wrong; only convicted cases become regression tests, and a human approves every policy change before it goes live.</p>
<h3>The 9-step loop</h3>
<ol><li><b>Nominate</b>: a claim is flagged (by a scan, Raindrop signal, or by hand) as one the agent may have handled wrongly.</li>
<li><b>File</b>: build the case file (claim, agent decision, reviewer resolution) so the adjuster can read it.</li>
<li><b>Verdict</b>: the adjuster convicts (states the correct outcome) or dismisses (the agent was fine).</li>
<li><b>Scenario in dev split</b>: a conviction writes a regression scenario into the dev split of the scenario docket.</li>
<li><b>Propose</b>: run a proposer to suggest a policy change and score it against the saved test cases.</li>
<li><b>Safety check</b>: the fixed evaluator and gate decide whether the proposed change is safe to consider.</li>
<li><b>Read the diff</b>: a human reads the proposed changes and the {{ L.scorecard }} numbers.</li>
<li><b>Approve</b>: a human approves (or rejects) the proposed version.</li>
<li><b>Prove live</b>: watch the live policy on real claims and make a previous version live again if it misbehaves.</li></ol>
<h3>Glossary of fields</h3><dl>
<dt>Verdict dropdown</dt><dd>The outcome the agent should have given for the claim (the expected recommendation in the new test).</dd>
<dt>Critical</dt><dd>Tick when a wrong answer on this case is unacceptable. It must never be auto-approved; the safety check blocks any version that does.</dd>
<dt>adjuster</dt><dd>The name of the human making the ruling. It is required and is written to the audit log.</dd>
<dt>{{ L.convict }}</dt><dd>{{ L.tip_convict }}</dd>
<dt>{{ L.dismiss }}</dt><dd>{{ L.tip_dismiss }}</dd>
<dt>{{ L.overturn }}</dt><dd>{{ L.tip_overturn }} A reason is required.</dd>
<dt>{{ L.retract }}</dt><dd>{{ L.tip_retract }} A reason is required.</dd>
<dt>reason</dt><dd>A short written justification for an overturn or retract; kept in the case History and audit log.</dd>
<dt>confirm box</dt><dd>The tick box saying you understand the action changes live state. Nothing is changed unless it is ticked.</dd>
<dt>History</dt><dd>The list of past overturns and retractions for a case: when, by whom, what it was before, and why.</dd></dl>
<h3>{{ L.scorecard }} columns</h3><dl>
<dt>{{ L.col_dev_acc }}</dt><dd>{{ L.tip_dev_acc }}</dd>
<dt>{{ L.col_heldout_acc }}</dt><dd>{{ L.tip_heldout_acc }}</dd>
<dt>{{ L.col_cases }}</dt><dd>{{ L.tip_cases }}</dd>
<dt>{{ L.col_critical }}</dt><dd>{{ L.tip_critical }}</dd>
<dt>{{ L.col_over }}</dt><dd>{{ L.tip_over }}</dd>
<dt>{{ L.col_under }}</dt><dd>{{ L.tip_under }}</dd>
<dt>{{ L.col_gate }}</dt><dd>{{ L.tip_gate }}</dd></dl>
<h3>{{ L.scorecard }} rows</h3><dl>
<dt>{{ L.row_baseline }}</dt><dd>The policy the agent shipped with.</dd>
<dt>{{ L.row_active }}</dt><dd>The version running on real claims now.</dd>
<dt>{{ L.row_candidate }}</dt><dd>A new version waiting for a human to approve or reject it.</dd></dl>
<h3>{{ L.versions }}</h3><dl>
<dt>{{ L.ver_col_version }}</dt><dd>The policy version number.</dd>
<dt>{{ L.ver_col_status }}</dt><dd><b>{{ L.status.active }}</b> is running now; <b>{{ L.status.retired }}</b> was live before and can be made live again; <b>{{ L.status.candidate }}</b> is proposed and needs approval; <b>{{ L.status.rejected }}</b> was declined.</dd>
<dt>{{ L.ver_col_author }}</dt><dd>Who or what proposed the version.</dd>
<dt>{{ L.ver_col_reason }}</dt><dd>Why the version was proposed.</dd>
<dt>{{ L.make_live }}</dt><dd>Make a previous version the live policy again (needs the confirm box).</dd>
<dt>{{ L.approve }}</dt><dd>Make the proposed change the live policy (needs the confirm box).</dd>
<dt>{{ L.reject }}</dt><dd>Decline the proposed change (needs the confirm box).</dd></dl>
<h3>{{ L.propose_title }}</h3><dl>
<dt>{{ L.proposer_heuristic }}</dt><dd>Built-in rules suggest a change. Instant and free.</dd>
<dt>{{ L.proposer_llm }}</dt><dd>{{ L.dry_text }} first; nothing is sent until you press {{ L.ask_llm }}.</dd></dl>
<h3>{{ L.activity_log }}</h3><dl><dt>{{ L.activity_log }}</dt><dd>A record of every action taken here: when, who, what it targeted and whether it succeeded.</dd></dl>
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
<button type="button" class="tab" role="tab" id="t-nominated" data-tab="nominated" aria-controls="tab-nominated" aria-selected="false">{{ L.tab_nominated }}<span class="cnt">{{ nominated|length }}</span></button>
<button type="button" class="tab" role="tab" id="t-filed" data-tab="filed" aria-controls="tab-filed" aria-selected="false">{{ L.tab_filed }}<span class="cnt">{{ filed|length }}</span></button>
<button type="button" class="tab" role="tab" id="t-decided" data-tab="decided" aria-controls="tab-decided" aria-selected="false">{{ L.tab_decided }}<span class="cnt">{{ closed|length }}</span></button></div>

<section id="tab-nominated" role="tabpanel" aria-labelledby="t-nominated"><div class="hd"><h2>Nominated ({{ nominated|length }})</h2>{{ tip("tip-nom", "About " ~ L.tab_nominated, "Claims flagged as possibly mishandled by the agent, waiting for someone to file a case file.") }}</div><p class="cap">{{ L.cap_nominated }}</p>
{% for c in nominated %}<div class="case"><b>{{ c.claim_id }}</b> <span class="muted">[{{ c.source }}]</span><br>
{{ c.reason }}<br>
<form method="post" action="/case/{{ c.case_id }}/file" class="inline">{{ tok() }}<button>File case</button></form>
<form method="post" action="/case/{{ c.case_id }}/verdict" class="inline adj" novalidate>{{ tok() }}
<input type="hidden" name="action" value="dismiss"><input type="text" name="adjuster" placeholder="adjuster">
<label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label>
<button title="{{ L.tip_dismiss }}">{{ L.dismiss }}</button><div class="err-inline hide" role="alert">Adjuster name is required.</div></form></div>{% else %}<p class="muted">None.</p>{% endfor %}</section>

<section id="tab-filed" role="tabpanel" aria-labelledby="t-filed"><div class="hd"><h2>Filed ({{ filed|length }})</h2>{{ tip("tip-filed", "About " ~ L.tab_filed, "Cases with a case file built, waiting for an adjuster to convict or dismiss.") }}</div><p class="cap">{{ L.cap_filed }}</p>
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
<label title="Critical: {{ L.tip_critical_badge }}"><input type="checkbox" name="critical" value="1"> Critical</label><br>
<input type="text" name="adjuster" placeholder="adjuster (required)"><br>
<label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label>
<button title="{{ L.tip_convict }}">{{ L.convict }}</button><div class="err-inline hide" role="alert">Adjuster name is required.</div></form>
<form method="post" action="/case/{{ c.case_id }}/verdict" class="adj" novalidate style="margin-top:6px">{{ tok() }}
<input type="hidden" name="action" value="dismiss">
<input type="text" name="adjuster" placeholder="adjuster (required)">
<label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label>
<button title="{{ L.tip_dismiss }}">{{ L.dismiss }}</button><div class="err-inline hide" role="alert">Adjuster name is required.</div></form></div>
{% else %}<p class="muted">None.</p>{% endfor %}</section>

<section id="tab-decided" role="tabpanel" aria-labelledby="t-decided"><div class="hd"><h2>Convicted / Dismissed ({{ closed|length }})</h2>{{ tip("tip-dec", "About Convicted and Dismissed", "Cases an adjuster has ruled on. Convicted cases became regression scenarios; dismissed cases did not. Open a row to see History, Overturn and Retract.") }}</div>
{% if closed %}<p class="cap">{{ L.cap_decided }}</p><div class="dtools"><div class="chips" role="group" aria-label="Filter decided cases">
<button type="button" class="chip" data-chip="all" aria-pressed="true">All</button>
<button type="button" class="chip" data-chip="convicted" aria-pressed="false">Convicted</button>
<button type="button" class="chip" data-chip="dismissed" aria-pressed="false">Dismissed</button></div>
<input type="search" id="decided-search" placeholder="Search claim id or text" aria-label="Search decided cases by claim id or text" autocomplete="off"></div>
<div class="tw"><table id="decided-table"><thead><tr><th>claim</th><th>status</th><th>expected</th><th>critical</th><th>adjuster</th><th>date</th><th>reason / note</th><th></th></tr></thead><tbody>
{% for c in closed %}<tr class="drow" data-status="{{ c.status }}" data-text="{{ (c.claim_id ~ ' ' ~ c.status ~ ' ' ~ c.who ~ ' ' ~ c.expected_recommendation ~ ' ' ~ c.note)|lower }}"{% if loop.index > page_size %} hidden{% endif %}>
<td><b>{{ c.claim_id }}</b></td><td><span class="badge b-{{ c.status }}">{{ c.status }}</span></td>
<td>{{ c.expected_recommendation or "-" }}</td><td>{% if c.critical %}<span class="badge b-critical" title="{{ L.tip_critical_badge }}">{{ L.critical_badge }}</span>{% else %}-{% endif %}</td>
<td>{{ c.who }}</td><td>{{ c.when }}</td><td class="note" title="{{ c.note }}">{{ c.note }}</td>
<td><button type="button" class="expand" aria-expanded="false" aria-controls="dd-{{ c.case_id }}" aria-label="Details for {{ c.claim_id }}">Details</button></td></tr>
<tr class="dd" id="dd-{{ c.case_id }}" hidden><td colspan="8"><div class="dbox">
{% if c.status == "convicted" %}<span class="muted">Scenario written: {{ c.draft.id }} to docket.json (dev split)</span>{% endif %}
{% if c.history %}<details><summary>History ({{ c.history|length }})</summary><ul>
{% for h in c.history %}<li>{{ h.ts }} {{ h.action }} by {{ h.by }}: was {{ h.previous_expected }}{% if h.previous_critical %} (critical){% endif %}. Reason: {{ h.reason }}</li>{% endfor %}</ul></details>{% endif %}
{% if c.status == "convicted" %}<h3>{{ L.overturn }}</h3><p class="cap">{{ L.tip_overturn }}</p>
<form method="post" action="/case/{{ c.case_id }}/overturn" class="adj" novalidate>{{ tok() }}
<select name="expected">{% for r in recs %}<option value="{{ r }}"{% if r == c.expected_recommendation %} selected{% endif %}>{{ r }}</option>{% endfor %}</select>
<label title="Critical: {{ L.tip_critical_badge }}"><input type="checkbox" name="critical" value="1"{% if c.critical %} checked{% endif %}> Critical</label><br>
<input type="text" name="adjuster" placeholder="adjuster (required)"><br>
<input type="text" name="reason" placeholder="reason (required)" required><br>
<label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label>
<button title="{{ L.tip_overturn }}">{{ L.overturn }}</button><div class="err-inline hide" role="alert">Adjuster name is required.</div></form>
<h3>{{ L.retract }}</h3><p class="cap">{{ L.tip_retract }}</p>
<form method="post" action="/case/{{ c.case_id }}/retract" class="adj" novalidate>{{ tok() }}
<input type="text" name="adjuster" placeholder="adjuster (required)"><br>
<input type="text" name="reason" placeholder="reason (required)" required><br>
<label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label>
<button title="{{ L.tip_retract }}">{{ L.retract }}</button><div class="err-inline hide" role="alert">Adjuster name is required.</div></form>{% endif %}
</div></td></tr>
{% endfor %}</tbody></table></div>
<p class="muted" id="decided-empty" hidden>No decided cases match.</p>
<p><button type="button" id="show-more" hidden>Show more</button> <span class="muted" id="decided-count"></span></p>
{% else %}<p class="muted">None.</p>{% endif %}</section>

<div class="panel pos" id="trial">{{ tip("tip-trial", "About " ~ L.scorecard, L.scorecard_sub ~ " The safety check decides if a proposed change is safe.") }}<details open><summary>{{ L.scorecard }}</summary>
<p class="cap">{{ L.scorecard_sub }}</p>
<div class="tw"><table><tr><th></th><th>{{ L.col_version }}</th><th title="{{ L.tip_dev_acc }}">{{ L.col_dev_acc }}</th><th title="{{ L.tip_heldout_acc }}">{{ L.col_heldout_acc }}</th><th title="{{ L.tip_cases }}">{{ L.col_cases }}</th><th title="{{ L.tip_critical }}">{{ L.col_critical }}</th><th title="{{ L.tip_over }}">{{ L.col_over }}</th><th title="{{ L.tip_under }}">{{ L.col_under }}</th><th title="{{ L.tip_gate }}">{{ L.col_gate }}</th></tr>
{% for label, ver, r, h, gate in trial %}<tr><td>{{ label }}</td><td>{{ ver }}</td><td>{{ r.accuracy|pct }}</td><td>{{ h.accuracy|pct }}</td><td>{{ r.n }}</td>
<td>{% if r.critical_violations %}<span class="badge b-critical">{{ r.critical_violations }}</span>{% else %}{{ r.critical_violations }}{% endif %}</td><td>{{ r.over_escalation_rate|pct }}</td><td>{{ r.under_escalation_rate|pct }}</td>
<td>{% if gate == "pass" %}<span class="badge b-active">{{ L.gate_pass }}</span>{% elif gate == "fail" %}<span class="badge b-rejected">{{ L.gate_fail }}</span>{% else %}-{% endif %}</td></tr>{% endfor %}</table></div>
<h3>{{ L.propose_title }}</h3>
<form method="post" action="/cycle">{{ tok() }}
<select name="proposer" aria-label="Proposer"><option value="heuristic">{{ L.proposer_heuristic }}</option><option value="llm">{{ L.proposer_llm }}</option></select>
<button>{{ L.propose_button }}</button></form>
<p class="muted">{{ L.propose_note }}</p>
{% if dry %}<h3>{{ L.dry_title }}</h3>
<p>model {{ dry.model }}, about <b>{{ dry.estimated_prompt_tokens }} prompt tokens</b>, max output {{ dry.max_output_tokens }}, cached {{ dry.cached }}, calls allowed {{ dry.calls_allowed }}</p>
<details><summary>{{ L.dry_prompt }}</summary><pre>{{ dry.prompt }}</pre></details>
<form method="post" action="/cycle">{{ tok() }}<input type="hidden" name="proposer" value="llm">
<button name="action" value="llm_call">{{ L.ask_llm }}</button></form>{% endif %}
{% if report %}<h3>{{ L.proposed_title }}</h3>
<p>proposer {{ report.proposer }}, status <b>{{ report.status }}</b>{% if report.version %}, version v{{ report.version }}{% endif %}, risk {{ report.risk }}</p>
<p>{{ report.rationale }}</p>
<p>Ops:</p><pre>{{ report.ops_json }}</pre>
{% if report.gate %}<p>{{ L.col_gate }}: <b>{{ L.gate_pass if report.gate.gate == "pass" else L.gate_fail }}</b></p>
<ul>{% for r in report.gate.reasons %}<li>{{ r }}</li>{% endfor %}</ul>{% endif %}
{% if report.error %}<p class="err flash">{{ report.error }}</p>{% endif %}
{% if report.status == "candidate" %}
<form method="post" action="/version/{{ report.version }}/approve" class="inline">{{ tok() }}<input type="text" name="reviewer" placeholder="reviewer"><label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label><button>{{ L.approve }}</button></form>
<form method="post" action="/version/{{ report.version }}/reject" class="inline">{{ tok() }}<input type="text" name="reviewer" placeholder="reviewer"><label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label><button>{{ L.reject }}</button></form>{% endif %}{% endif %}
</details></div>

<div class="panel pos" id="versions">{{ tip("tip-ver", "About " ~ L.versions, "Every policy version: " ~ L.status.active ~ " is running now, " ~ L.status.retired ~ " can be made live again, " ~ L.status.candidate ~ " needs a human decision, " ~ L.status.rejected ~ " was declined.") }}<details><summary>{{ L.versions }} ({{ versions|length }})</summary>
<div class="tw"><table><tr><th>{{ L.ver_col_version }}</th><th>{{ L.ver_col_status }}</th><th>{{ L.ver_col_author }}</th><th>{{ L.ver_col_reason }}</th><th>{{ L.ver_col_actions }}</th></tr>
{% for v in versions %}<tr><td>{{ v.version }}</td><td><span class="badge b-{{ v.status }}">{{ L.status.get(v.status, v.status) }}</span></td><td>{{ v.author|author }}</td><td class="reason">{% set rs = v.reason or "" %}{% if rs|length > 120 %}<details><summary>{{ rs[:120] }}...</summary>{{ rs }}</details>{% else %}{{ rs }}{% endif %}</td>
<td>{% if v.status == "retired" %}<form method="post" action="/version/{{ v.version }}/rollback" class="inline">{{ tok() }}<input type="text" name="reviewer" placeholder="reviewer"><label class="confirm"><input type="checkbox" name="confirm" value="1"> I understand this changes live state</label><button>{{ L.make_live }}</button></form>{% endif %}</td></tr>{% endfor %}</table></div>
</details></div>

<div class="panel pos" id="audit">{{ tip("tip-aud", "About " ~ L.activity_log, "A record of every action taken here: when, who, what it targeted and whether it succeeded.") }}<details><summary>{{ L.activity_log }} (last {{ audit|length }})</summary>
{% if audit %}<div class="tw"><table><tr><th>time (UTC)</th><th>action</th><th>actor</th><th>target</th><th>outcome</th><th>detail</th></tr>
{% for r in audit %}<tr><td>{{ r.ts }}</td><td>{{ r.action }}</td><td>{{ r.actor }}</td><td>{{ r.target }}</td><td>{{ r.outcome }}</td><td class="reason">{{ r.detail }}</td></tr>{% endfor %}</table></div>
{% else %}<p class="muted">No activity yet.</p>{% endif %}</details></div>
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
    env.filters["pct"] = _pct
    env.filters["author"] = _author_label
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
            trial = [(UI_LABELS["row_baseline"], "-", evaluate(DEFAULT_MANIFEST, "dev"), evaluate(DEFAULT_MANIFEST, "heldout"), "-"),
                     (UI_LABELS["row_active"], f"v{active['version']}", evaluate(active["manifest"], "dev"),
                      evaluate(active["manifest"], "heldout"), "-")]
            rep_ = state["report"]
            if rep_ and rep_.get("version") and (rep_.get("candidate") or {}).get("dev"):
                cand = rep_["candidate"]
                gate = (rep_.get("gate") or {}).get("gate", "-")
                trial.append((UI_LABELS["row_candidate"] if rep_["status"] == "candidate" else f"Proposed change ({rep_['status']})", f"v{rep_['version']}", cand["dev"], cand["heldout"], gate))
        flashes, state["flashes"] = state["flashes"], []
        closed = [_decided_view(c) for c in cases if c["status"] in ("convicted", "dismissed")]
        closed.sort(key=lambda c: c["when_full"], reverse=True)
        return env.get_template("index.html").render(
            active=active, flashes=flashes, trial=trial, versions=store.list_versions(),
            nominated=[c for c in cases if c["status"] == "nominated"],
            filed=[c for c in cases if c["status"] == "filed"],
            closed=closed, L=UI_LABELS, app_name=APP_NAME, app_subtitle=APP_SUBTITLE, page_size=DECIDED_PAGE_SIZE,
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
