"""Judge-demo simulator: a simulated production incident stream that ends at a human approval.

    uv run python -m harness_evolve.cli demo seed   [--user-id U] [--allow-remote]
    uv run python -m harness_evolve.cli demo run    [--scenario NAME] [--pace SECONDS] [--ui-url URL] [--allow-remote]
    uv run python -m harness_evolve.cli demo status
    uv run python -m harness_evolve.cli demo reset  --yes [--allow-remote]

Everything created here is tagged `demo: true` (claims CLM-SIM-*, policy POL-SIM-DEMO, their court cases,
and any candidate version the run stores) so `demo reset` removes only that. The adjuster verdict is
SIMULATED and labelled so in the audit log. Nothing here ever approves, activates or rolls back a
version: that stays a human click in the Regression Court UI. Rule-based proposer only; no LLM.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_agent46.harness_config import DEFAULT_MANIFEST  # noqa: E402
from agent_agent46.policy_store import CLAIMS_COLLECTION, POLICY_COLLECTION, Claim, Policy  # noqa: E402
from agent_agent46.risk import assess_claim_risk  # noqa: E402
from evals.run_eval import AUTO, MANUAL, RANK, REVIEW  # noqa: E402

from .audit import check_remote, os_user, record, run_confirmed  # noqa: E402
from .court import COURT_COLLECTION, CourtDocket  # noqa: E402
from .loop import reject, run_cycle  # noqa: E402
from .proposer import HeuristicProposer  # noqa: E402

POLICY_NUMBER = "POL-SIM-DEMO"
DEFAULT_USER = "harness-test-user-1"
SIMULATED_ADJUSTER = "Demo Adjuster (simulated)"
DEFAULT_UI_URL = "http://127.0.0.1:8765"

# Seeded claims. stored recommendation is computed at seed time with the real risk function under the
# live manifest. The first 7 are candidates for the Nominated / Filed cases; 08-10 are look-alike controls.
CLAIMS = [
    {"id": "CLM-SIM-01", "type": "comprehensive", "amount": 850.0, "source": "manual",
     "reason": "Closed in 40 minutes with no inspection; matches a parts-stripping pattern seen at the same car park.",
     "desc": "Returned to the car after a night shift and found all four wheels, the catalytic converter and "
             "the stereo gone; the car was left stripped on its rims in the staff car park."},
    {"id": "CLM-SIM-02", "type": "collision", "amount": 900.0, "source": "raindrop_signal",
     "reason": "Raindrop signal 'vin_mismatch_after_approval' fired on the follow-up conversation.",
     "desc": "Registry check returned two vehicles with the same VIN after a minor parking scrape; the car "
             "was bought privately last month and the plates look cloned."},
    {"id": "CLM-SIM-03", "type": "collision", "amount": 5400.0, "source": "reviewer_disagreement", "reason": None,  # derived from the stored recommendation
     "desc": "Low speed rear-end collision on the ring road at 2 a.m.; the other driver left before police "
             "arrived and the footage suggests the crash was staged between two cars braking together."},
    {"id": "CLM-SIM-04", "type": "comprehensive", "amount": 6200.0, "source": "manual",
     "reason": "Investigator found the shell had been sold for parts before the claim was paid.",
     "desc": "Engine, gearbox and both doors removed from a car parked at a storage yard for three weeks; "
             "the shell looks cannibalized for parts."},
    {"id": "CLM-SIM-05", "type": "comprehensive", "amount": 780.0, "source": "manual",
     "reason": "Neighbours reported the car was left on the service road for weeks before the damage was claimed.",
     "desc": "Car sat on a service road for weeks with flat tyres and no plates before the owner reported "
             "damage; neighbours describe it as abandoned."},
    {"id": "CLM-SIM-06", "type": "collision", "amount": 700.0, "source": "raindrop_signal",
     "reason": "Raindrop signal 'repeat_counterparty' fired: same two drivers as an earlier claim.",
     "desc": "Fender damage reported after a collision at a roundabout; the same two drivers filed a near "
             "identical claim last spring and the scene looks staged."},
    {"id": "CLM-SIM-07", "type": "comprehensive", "amount": 950.0, "source": "raindrop_signal",
     "reason": "Raindrop signal 'vin_plate_replaced' fired on the glass-break claim.",
     "desc": "Glass break claim on an imported hatchback whose VIN plate was replaced; the dealer says the "
             "vehicle may be cloned from a scrapped car."},
    {"id": "CLM-SIM-08", "type": "comprehensive", "amount": 400.0, "source": "manual",
     "reason": "Control case: a correctly approved claim that must stay auto-approved.",
     "desc": "Abandoned shopping trolley rolled across the supermarket car park and dented the rear door "
             "of the insured car; the store accepted liability."},
    {"id": "CLM-SIM-09", "type": "comprehensive", "amount": 350.0, "source": "manual",
     "reason": "Control case: a correctly approved claim that must stay auto-approved.",
     "desc": "A neighbour's abandoned bicycle fell against the driver door overnight and scratched the paint; "
             "the neighbour has apologised and offered to pay."},
    {"id": "CLM-SIM-10", "type": "comprehensive", "amount": 300.0, "source": "manual",
     "reason": "Control case: a correctly approved claim that must stay auto-approved.",
     "desc": "Rear window cracked by a rock kicked up while driving past an abandoned building site; the "
             "site manager confirmed the hoarding was down."},
]
CLAIM_IDS = [c["id"] for c in CLAIMS]
_BY_ID = {c["id"]: c for c in CLAIMS}

# Alternative (description, amount) per claim, tried in order when the original wording is no longer a miss
# under the live manifest (for example its keyword has since been added). Each uses only phrases from
# proposer.RISK_LEXICON and amounts that keep the baseline risk function below manual review.
VARIANTS = {
    "CLM-SIM-01": [
        ("The owner away on a work trip for ten days; the car was found on bricks in the driveway with all four "
         "wheels, the catalytic converter and the stereo gone.", 850.0),
        ("Wheels, catalytic converter and stereo taken from the car in the staff car park in what looks like "
         "organised vandalism; two other cars had the same damage.", 850.0),
    ],
    "CLM-SIM-02": [
        ("Registry check returned two vehicles with the same VIN after a parking scrape; the other driver left "
         "in a hit and run and the plates on the car do not match the registry record.", 900.0),
        ("Minor parking scrape with no witnesses; the registry shows two cars on the road with this VIN and "
         "the plates do not match the paperwork.", 900.0),
    ],
    "CLM-SIM-03": [
        ("Low speed rear-end collision on the ring road at 2 a.m.; the other driver left before police arrived "
         "in what officers are treating as a hit and run, and the footage is unclear.", 5400.0),
        ("Low speed rear-end collision on the ring road at 2 a.m. with no witnesses; the other driver left "
         "before police arrived and the dashcam footage is unclear.", 5400.0),
    ],
    "CLM-SIM-04": [
        ("Engine, gearbox and both doors removed from a car parked at a storage yard for three weeks; the "
         "shell looks stripped for parts.", 6200.0),
        ("Car left at a storage yard with the owner away abroad for three weeks; on return the engine, "
         "gearbox and both doors had been removed.", 6200.0),
    ],
    "CLM-SIM-05": [
        ("Car sat on a service road for weeks with flat tyres and no plates, the owner away abroad, and "
         "damage was reported on return.", 780.0),
        ("Car damaged overnight on a service road where it had stood for weeks; there were no witnesses and "
         "the owner only reported it on return.", 780.0),
    ],
    "CLM-SIM-06": [
        ("Fender damage reported after a collision at a roundabout; the other driver drove off in a hit and "
         "run and the same two drivers filed a near identical claim last spring.", 700.0),
        ("Fender damage reported after a collision at a roundabout with no witnesses; the same two drivers "
         "filed a near identical claim last spring.", 700.0),
    ],
    "CLM-SIM-07": [
        ("Glass break claim on an imported hatchback after vandalism in a car park; the dealer notes the VIN "
         "plate was replaced and the paperwork does not match.", 950.0),
        ("Glass break claim on an imported hatchback with no witnesses; the dealer notes the VIN plate was "
         "replaced and the history looks thin.", 950.0),
    ],
}

# (claim_id, expected recommendation, critical)
SCENARIOS = {
    "parts-theft": {
        "keywords": ("stripped", "cannibalized"),
        "title": "Parts stripping ring",
        "summary": "adds 'stripped' and 'cannibalized' as high-risk keywords",
        "claims": [("CLM-SIM-01", MANUAL, True), ("CLM-SIM-04", MANUAL, True)],
        "trap": False,
    },
    "staged-accident": {
        "keywords": ("staged",),
        "title": "Staged collisions",
        "summary": "adds 'staged' as a high-risk keyword",
        "claims": [("CLM-SIM-03", MANUAL, True), ("CLM-SIM-06", MANUAL, True)],
        "trap": False,
    },
    "cloned-vehicle": {
        "keywords": ("cloned",),
        "title": "Cloned VINs",
        "summary": "adds 'cloned' as a high-risk keyword",
        "claims": [("CLM-SIM-02", MANUAL, True), ("CLM-SIM-07", MANUAL, True)],
        "trap": False,
    },
    "abandoned-lookalike": {
        "keywords": ("abandoned",),
        "title": "Abandoned vehicle, with a look-alike trap",
        "summary": "adds 'abandoned' as a medium keyword, which also catches legitimate claims that merely "
                   "mention the word, so the gate REJECTS it (over-escalation rises)",
        "claims": [("CLM-SIM-05", REVIEW, False), ("CLM-SIM-08", AUTO, False), ("CLM-SIM-09", AUTO, False),
                   ("CLM-SIM-10", AUTO, False)],
        "trap": True,
    },
}
DEFAULT_SCENARIO = "parts-theft"
EXPECTED = {cid: exp for sc in SCENARIOS.values() for cid, exp, _ in sc["claims"]}
POOL = [c["id"] for c in CLAIMS if EXPECTED[c["id"]] != AUTO]  # candidates for Nominated / Filed, in order
N_NOMINATED, N_FILED = 3, 2
BASE_POLICY = {"risk_score": 0.3, "coverage_limit": 50000}


class AlreadyFixed(RuntimeError):
    """The scenario's fix is already in the live manifest; there is nothing to demonstrate."""


def _rec(claim_type, amount, desc, policy, manifest) -> str:
    return assess_claim_risk({"claim_amount": amount, "claim_type": claim_type, "description": desc},
                             policy, [], manifest)["recommendation"]


def is_miss(cid: str, rec: str) -> bool:
    """A real miss: the policy under-escalates relative to the scenario's expected outcome."""
    return RANK[rec] < RANK[EXPECTED[cid]]


def resolve(manifest: dict, policy: dict | None = None) -> list[dict]:
    """Every claim with the first wording (original, then variants) that is a real miss under `manifest`.
    `miss` is None for controls; `variant` is 0 for the original wording."""
    policy = policy or BASE_POLICY
    out = []
    for c in CLAIMS:
        if EXPECTED[c["id"]] == AUTO:
            out.append(c | {"miss": None, "variant": 0})
            continue
        options = [(c["desc"], c["amount"])] + VARIANTS.get(c["id"], [])
        for i, (desc, amount) in enumerate(options):
            if is_miss(c["id"], _rec(c["type"], amount, desc, policy, manifest)):
                out.append(c | {"desc": desc, "amount": amount, "miss": True, "variant": i})
                break
        else:
            out.append(c | {"miss": False, "variant": 0})
    return out


def _covered(manifest: dict, kw: str) -> bool:
    return any(kw in k.lower() for lvl in ("high", "medium") for k in manifest["keywords"][lvl])


def scenario_fixed(manifest: dict, name: str) -> bool:
    return all(_covered(manifest, kw) for kw in SCENARIOS[name]["keywords"])


def _label(doc: dict, manifest: dict, policy: dict | None = None) -> str:
    """'real miss' or 'already handled' for a stored claim under the live manifest."""
    rec = _rec(doc["claim_type"], doc["claim_amount"], doc["description"], policy or BASE_POLICY, manifest)
    return "real miss" if is_miss(doc["claim_id"], rec) else "already handled"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(doc):
    return None if doc is None else {k: v for k, v in doc.items() if k != "_id"}


def _active(store):
    doc = store.get_active()
    return (doc["version"], doc["manifest"]) if doc else (0, DEFAULT_MANIFEST)


def candidates(store) -> list[dict]:
    return [v for v in store.list_versions() if v.get("status") == "candidate"]


def _policy_doc(db, user_id: str) -> dict:
    base = Policy(policy_number=POLICY_NUMBER, user_id=user_id, holder_name="Dana Whitfield",
                  holder_email="dana.whitfield@example.com", type="Auto Insurance", coverage="Comprehensive",
                  coverage_limit=50000, premium="$120/month", deductible="$500", status="Active",
                  risk_score=0.3, car_model="Corolla", car_year=2021, car_edition="LE",
                  created_at=_now()).model_dump()
    sample = db[POLICY_COLLECTION].find_one({"policy_number": {"$ne": POLICY_NUMBER}})
    for k in (sample or {}):  # follow whatever shape live policies have
        if k != "_id" and k not in base:
            base[k] = None
    return base | {"demo": True}


def _claim_doc(c: dict, user_id: str, policy: dict, version: int, manifest: dict) -> dict:
    risk = assess_claim_risk({"claim_amount": c["amount"], "claim_type": c["type"], "description": c["desc"]},
                             policy, [], manifest)
    rec = risk["recommendation"]
    notes = ("Approved at standard review." if rec == REVIEW else "Auto-approved by the agent; no human review.")
    doc = Claim(claim_id=c["id"], policy_number=POLICY_NUMBER, user_id=user_id, claim_type=c["type"],
                claim_amount=c["amount"], description=c["desc"], status="approved",
                risk_assessment=risk["risk_assessment"], risk_confidence=risk["confidence"],
                recommendation=rec, reviewer_notes=notes, resolution="approved",
                created_at=_now(), updated_at=_now(), resolved_at=_now()).model_dump()
    return doc | {"harness_version": version, "demo": True}


def _file(court: CourtDocket, case_id: str) -> dict:
    """file_case counts every other claim on the policy as 'previous claims'. The demo incidents are
    independent events that share one demo policy only for tidiness, so the draft starts at zero."""
    court.file_case(case_id)
    case = court._get(case_id)
    draft = dict(case["draft"], previous_claims_count=0)
    court._cases.update_one({"case_id": case_id}, {"$set": {"draft": draft}})
    return court._get(case_id)


def seed(db, store, user_id: str = DEFAULT_USER, actor: str = "unknown", out=print) -> dict:
    """Idempotent. Creates the demo policy, claims and the Nominated / Filed court cases."""
    version, manifest = _active(store)
    if not db[POLICY_COLLECTION].find_one({"policy_number": POLICY_NUMBER}):
        db[POLICY_COLLECTION].insert_one(_policy_doc(db, user_id))
    policy = _clean(db[POLICY_COLLECTION].find_one({"policy_number": POLICY_NUMBER}))
    court = CourtDocket(db)
    created_claims, notes = 0, []
    has_case = {c["claim_id"] for c in court.list_cases()}
    live = policy
    resolved = {r["id"]: r for r in resolve(manifest, live)}
    docs = {}
    for c in CLAIMS:
        r = resolved[c["id"]]
        doc = _clean(db[CLAIMS_COLLECTION].find_one({"claim_id": c["id"]}))
        if doc is not None and c["id"] not in has_case and (doc["description"], doc["claim_amount"]) != (
                r["desc"], r["amount"]):
            db[CLAIMS_COLLECTION].delete_many({"claim_id": c["id"]})  # stale wording from an older live policy
            doc = None
        if doc is None:
            doc = _claim_doc(r, user_id, policy, version, manifest)
            db[CLAIMS_COLLECTION].insert_one(dict(doc))
            created_claims += 1
        docs[c["id"]] = doc
        if r["variant"]:
            notes.append(f"{c['id']}: original wording is already handled by v{version}; using alternative "
                         f"wording #{r['variant']} so the live policy really misses it")
    # A claim counts only if its stored wording is a real miss under the live manifest.
    misses = [cid for cid in POOL if _label(docs[cid], manifest, live) == "real miss"]
    for cid in POOL:
        if cid not in misses:
            notes.append(f"{cid}: already fixed by v{version}; excluded from the Nominated/Filed cases")
    if len(misses) < N_NOMINATED + N_FILED:
        raise RuntimeError(f"only {len(misses)} demo claims are real misses under v{version}; need "
                           f"{N_NOMINATED + N_FILED}. Nothing left worth demonstrating")
    chosen = misses[:N_NOMINATED + N_FILED]
    for cid in chosen:
        c, doc = _BY_ID[cid], docs[cid]
        reason = c["reason"] or f"agent recommended {doc['recommendation']} but reviewer approved"
        case = court.nominate(cid, c["source"], reason)
        if chosen.index(cid) >= N_NOMINATED and case["status"] == "nominated":
            _file(court, case["case_id"])
    counts = {}
    for case in court.list_cases():
        if case["claim_id"] in CLAIM_IDS:
            counts[case["status"]] = counts.get(case["status"], 0) + 1
    detail = {"claims_created": created_claims, "cases": counts, "policy": POLICY_NUMBER}
    record(db, "demo.seed", actor, POLICY_NUMBER, detail, "ok")
    out(f"demo seed: policy {POLICY_NUMBER}, {len(CLAIMS)} claims ({created_claims} new), court cases {counts}")
    for n in notes:
        out(f"  note: {n}")
    return detail


def _say(out, pace, text="", sleep=time.sleep):
    out(text)
    if pace and pace > 0:
        sleep(pace)


def run(db, store, scenario: str = DEFAULT_SCENARIO, pace: float = 2.0, ui_url: str = DEFAULT_UI_URL,
        actor: str = "unknown", out=print, sleep=time.sleep) -> dict:
    """Narrated playthrough. Stores at most one candidate; never approves, activates or rolls back."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    sc = SCENARIOS[scenario]
    live_version, live_manifest = _active(store)
    if scenario_fixed(live_manifest, scenario):
        left = [f"{n} ({', '.join(k for k in SCENARIOS[n]['keywords'] if not _covered(live_manifest, k))})"
                for n in SCENARIOS if not scenario_fixed(live_manifest, n)]
        raise AlreadyFixed(f"scenario {scenario} is already fixed by v{live_version} (the live policy already "
                           f"has {', '.join(sc['keywords'])}); " + (
                               f"choose another scenario: {', '.join(left)}" if left
                               else "no scenario has anything left to fix"))
    if not db[POLICY_COLLECTION].find_one({"policy_number": POLICY_NUMBER}):
        raise RuntimeError("demo data not found; run `demo seed` first")
    pending = candidates(store)
    if pending:
        vs = ", ".join(f"v{v['version']}" for v in pending)
        raise RuntimeError(f"candidate version {vs} is already awaiting approval; approve or reject it in the "
                           "Regression Court (Policy versions) first, then run the demo again")
    say = lambda t="": _say(out, pace, t, sleep)  # noqa: E731
    court = CourtDocket(db)
    say(f"=== Simulated incident stream: {sc['title']} (scenario {scenario}) ===")
    say(f"Live policy is v{live_version}. Everything below is a SIMULATION on demo data (CLM-SIM-*).")

    say("\n[1/6] Incident detected")
    claims = {}
    for cid, expected, _crit in list(sc["claims"]):
        claims[cid] = _clean(db[CLAIMS_COLLECTION].find_one({"claim_id": cid}))
        if claims[cid] is None:
            raise RuntimeError(f"claim {cid} missing; run `demo seed` first")
        cl = claims[cid]
        if expected != AUTO and _label(cl, live_manifest) != "real miss":
            say(f"  {cid}: already handled by v{live_version}, left out of this scenario")
            claims.pop(cid)
            continue
        say(f"  {cid}  ${cl['claim_amount']:,.0f}  agent said {cl['recommendation']}, claim paid, later flagged wrong")
        say(f"      \"{cl['description']}\"")

    in_scope = [t for t in sc["claims"] if t[0] in claims]
    say("\n[2/6] Nominate to the Regression Court")
    cases = {}
    for cid, _e, _c in in_scope:
        meta = _BY_ID[cid]
        reason = meta["reason"] or f"agent recommended {claims[cid]['recommendation']} but reviewer approved"
        cases[cid] = court.nominate(cid, meta["source"], reason)
        say(f"  {cid}: case {cases[cid]['case_id']} ({meta['source']}) - {cases[cid]['reason']}")

    say("\n[3/6] File the cases (draft scenario built from the claim, agent decision and harness version)")
    for cid in cases:
        if cases[cid]["status"] in ("nominated", "filed"):
            cases[cid] = _file(court, cases[cid]["case_id"])
            say(f"  {cid}: filed (agent: {cases[cid]['agent']['recommendation']}, "
                f"harness v{cases[cid]['agent']['harness_version']})")
        else:
            say(f"  {cid}: already {cases[cid]['status']}")

    say(f"\n[4/6] SIMULATED adjuster verdict by '{SIMULATED_ADJUSTER}' (not a real person)")
    verdicts = []
    for cid, expected, critical in in_scope:
        case = cases[cid]
        if case["status"] != "filed":
            say(f"  {cid}: skipped, case is {case['status']}")
            continue
        court.verdict(case["case_id"], expected, critical, SIMULATED_ADJUSTER, confirmed=True)
        record(db, "demo.simulated_verdict", SIMULATED_ADJUSTER, case["case_id"],
               {"simulated": True, "claim_id": cid, "expected": expected, "critical": critical,
                "note": "simulated adjuster verdict from the demo scenario table; no real adjuster involved"}, "ok")
        verdicts.append(cid)
        say(f"  {cid}: SIMULATED verdict expected={expected}{' (critical)' if critical else ''} -> docket")

    say("\n[5/6] Rule-based improvement cycle (heuristic proposer, risk surface, no LLM)")
    report = run_cycle(store, HeuristicProposer(), auto_activate_low_risk=False, surface="risk")
    version = report.get("version")
    if version is not None:
        store._collection.update_one({"version": version}, {"$set": {"demo": True, "demo_scenario": scenario}})
    say(f"  proposer: {report['proposer']}   parent: v{report['parent']}   status: {report['status']}")
    for op in report["ops"]:
        say(f"  op: {op}")
    if not report["ops"]:
        say("  no ops proposed; the live policy already handles these cases")
    gate = report.get("gate")
    if gate:
        say(f"  risk tier: {report['risk']}")
        for surf in ("risk", "tools", "limits"):
            g = report.get(f"{surf}_gate") or {}
            say(f"  gate {surf}: {g.get('gate', 'n/a')}")
        say(f"  combined gate: {gate['gate']}")
        for r in gate["reasons"]:
            say(f"    reason: {r}")
    result = {"scenario": scenario, "version": version, "status": report["status"],
              "gate": (gate or {}).get("gate"), "ops": report["ops"], "verdicts": verdicts}
    record(db, "demo.run", actor, scenario, {k: v for k, v in result.items() if k != "ops"} | {
        "simulated": True, "ops": len(report["ops"])}, "ok")

    say("\n[6/6] Waiting for a human")
    if report["status"] == "candidate":
        say(f"  Proposed policy version v{version} is stored as a CANDIDATE. The demo will not approve it.")
        say("  NEXT STEP: Open the Regression Court, find the proposed version under Policy versions, "
            "and approve it.")
        say(f"  Regression Court: {ui_url}")
    elif report["status"] == "rejected":
        say(f"  The gate REJECTED the proposal (v{version} is shown as rejected under Policy versions); "
            "there is nothing to approve.")
        say(f"  Open the Regression Court to see why: {ui_url}")
    else:
        say(f"  Nothing to approve (status: {report['status']}). Regression Court: {ui_url}")
    return result


def status(db, store, out=print) -> dict:
    pol = _clean(db[POLICY_COLLECTION].find_one({"policy_number": POLICY_NUMBER}))
    out(f"demo policy {POLICY_NUMBER}: {'present' if pol else 'absent'}")
    claims = [_clean(c) for c in db[CLAIMS_COLLECTION].find({"claim_id": {"$in": CLAIM_IDS}})]
    out(f"demo claims ({len(claims)}):")
    _v, manifest = _active(store)
    labels = {c["claim_id"]: ("control" if EXPECTED[c["claim_id"]] == AUTO else _label(c, manifest, pol))
              for c in claims}
    for c in sorted(claims, key=lambda c: c["claim_id"]):
        out(f"  {c['claim_id']}  ${c['claim_amount']:,.0f}  agent={c['recommendation']}  {c['resolution']}  "
            f"[{labels[c['claim_id']]}]")
    cases = [c for c in CourtDocket(db).list_cases() if c["claim_id"] in CLAIM_IDS]
    out(f"demo court cases ({len(cases)}):")
    for c in sorted(cases, key=lambda c: c["claim_id"]):
        out(f"  {c['case_id']}  {c['claim_id']}  {c['status']}  ({c['source']})  "
            f"{labels.get(c['claim_id'], 'claim missing')} under the live policy")
    active = store.get_active()
    out(f"live version: {('v%s' % active['version']) if active else 'none'}")
    cands = candidates(store)
    for v in cands:
        out(f"candidate version: v{v['version']}{' (demo)' if v.get('demo') else ''} - {v.get('reason', '')[:100]}")
    if not cands:
        out("candidate version: none")
    return {"policy": bool(pol), "claims": len(claims), "cases": len(cases),
            "case_labels": {c["claim_id"]: labels.get(c["claim_id"]) for c in cases},
            "live": active["version"] if active else None, "candidates": [v["version"] for v in cands]}


def reset(db, store, actor: str = "unknown", *, confirmed: bool = False, out=print) -> dict:
    """Remove ONLY demo data. Needs confirmed=True; audited."""
    def go() -> dict:
        court = CourtDocket(db)
        removed = {"cases": 0, "retracted": 0, "claims": 0, "policy": 0, "rejected_versions": []}
        for case in [c for c in court.list_cases() if c["claim_id"] in CLAIM_IDS]:
            if case["status"] == "convicted":
                court.retract(case["case_id"], actor, "demo reset", confirmed=True)  # audited; drops docket entry
                removed["retracted"] += 1
            db[COURT_COLLECTION].delete_many({"case_id": case["case_id"]})
            removed["cases"] += 1
        removed["claims"] = len(list(db[CLAIMS_COLLECTION].find({"claim_id": {"$in": CLAIM_IDS}, "demo": True})))
        db[CLAIMS_COLLECTION].delete_many({"claim_id": {"$in": CLAIM_IDS}, "demo": True})
        removed["policy"] = len(list(db[POLICY_COLLECTION].find({"policy_number": POLICY_NUMBER, "demo": True})))
        db[POLICY_COLLECTION].delete_many({"policy_number": POLICY_NUMBER, "demo": True})
        for v in candidates(store):
            if v.get("demo"):
                reject(store, v["version"], confirmed=True, actor=actor, audit_db=db)
                removed["rejected_versions"].append(v["version"])
        return removed

    removed = run_confirmed(db, "demo.reset", actor, POLICY_NUMBER, {}, confirmed, go)
    out(f"demo reset: {removed}")
    return removed


def demo_main(a, rest) -> int:
    """Entry for `cli demo ...`; `a` is the parsed global namespace (uri, db, yes, by, allow_remote)."""
    from agent_agent46.harness_config import HarnessConfigStore
    from pymongo import MongoClient

    p = argparse.ArgumentParser(prog="harness_evolve demo")
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("seed")
    s.add_argument("--user-id", default=DEFAULT_USER)
    r = sub.add_parser("run")
    r.add_argument("--pace", type=float, default=2.0)
    r.add_argument("--scenario", choices=sorted(SCENARIOS), default=DEFAULT_SCENARIO)
    r.add_argument("--ui-url", default=DEFAULT_UI_URL)
    sub.add_parser("status")
    sub.add_parser("reset")
    c = p.parse_args(rest)
    actor = a.by or os_user()
    if c.action != "status":
        check_remote(a.uri, a.allow_remote)
    db = MongoClient(a.uri, serverSelectionTimeoutMS=2000)[a.db]
    store = HarnessConfigStore(a.uri, a.db)
    if c.action != "status":
        store.seed_if_empty()
    try:
        if c.action == "seed":
            seed(db, store, c.user_id, actor)
        elif c.action == "run":
            run(db, store, c.scenario, c.pace, c.ui_url, actor)
        elif c.action == "status":
            status(db, store)
        else:
            reset(db, store, actor, confirmed=a.yes)
    except AlreadyFixed as e:
        print(f"demo: {e}")
        return 0
    except RuntimeError as e:
        print(f"demo: {e}", file=sys.stderr)
        return 1
    return 0
