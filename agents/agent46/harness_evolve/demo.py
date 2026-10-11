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
from evals.run_eval import AUTO, MANUAL, REVIEW  # noqa: E402

from .audit import check_remote, os_user, record, run_confirmed  # noqa: E402
from .court import COURT_COLLECTION, CourtDocket  # noqa: E402
from .loop import reject, run_cycle  # noqa: E402
from .proposer import HeuristicProposer  # noqa: E402

POLICY_NUMBER = "POL-SIM-DEMO"
DEFAULT_USER = "harness-test-user-1"
SIMULATED_ADJUSTER = "Demo Adjuster (simulated)"
DEFAULT_UI_URL = "http://127.0.0.1:8765"

# Seeded claims. stored recommendation is computed at seed time with the real risk function under the
# live manifest. `seed_case` is the court status seeded for the claim (None = appears only during `run`).
CLAIMS = [
    {"id": "CLM-SIM-01", "type": "comprehensive", "amount": 850.0, "seed_case": "nominated",
     "source": "manual",
     "reason": "Closed in 40 minutes with no inspection; matches a parts-stripping pattern seen at the same car park.",
     "desc": "Returned to the car after a night shift and found all four wheels, the catalytic converter and "
             "the stereo gone; the car was left stripped on its rims in the staff car park."},
    {"id": "CLM-SIM-02", "type": "collision", "amount": 900.0, "seed_case": "nominated",
     "source": "raindrop_signal",
     "reason": "Raindrop signal 'vin_mismatch_after_approval' fired on the follow-up conversation.",
     "desc": "Registry check returned two vehicles with the same VIN after a minor parking scrape; the car "
             "was bought privately last month and the plates look cloned."},
    {"id": "CLM-SIM-03", "type": "collision", "amount": 5400.0, "seed_case": "nominated",
     "source": "reviewer_disagreement", "reason": None,  # derived from the stored recommendation
     "desc": "Low speed rear-end collision on the ring road at 2 a.m.; the other driver left before police "
             "arrived and the footage suggests the crash was staged between two cars braking together."},
    {"id": "CLM-SIM-04", "type": "comprehensive", "amount": 6200.0, "seed_case": "filed",
     "source": "manual",
     "reason": "Investigator found the shell had been sold for parts before the claim was paid.",
     "desc": "Engine, gearbox and both doors removed from a car parked at a storage yard for three weeks; "
             "the shell looks cannibalized for parts."},
    {"id": "CLM-SIM-05", "type": "comprehensive", "amount": 780.0, "seed_case": "filed",
     "source": "manual",
     "reason": "Neighbours reported the car was left on the service road for weeks before the damage was claimed.",
     "desc": "Car sat on a service road for weeks with flat tyres and no plates before the owner reported "
             "damage; neighbours describe it as abandoned."},
    {"id": "CLM-SIM-06", "type": "collision", "amount": 700.0, "seed_case": None,
     "source": "raindrop_signal",
     "reason": "Raindrop signal 'repeat_counterparty' fired: same two drivers as an earlier claim.",
     "desc": "Fender damage reported after a collision at a roundabout; the same two drivers filed a near "
             "identical claim last spring and the scene looks staged."},
    {"id": "CLM-SIM-07", "type": "comprehensive", "amount": 950.0, "seed_case": None,
     "source": "raindrop_signal",
     "reason": "Raindrop signal 'vin_plate_replaced' fired on the glass-break claim.",
     "desc": "Glass break claim on an imported hatchback whose VIN plate was replaced; the dealer says the "
             "vehicle may be cloned from a scrapped car."},
    {"id": "CLM-SIM-08", "type": "comprehensive", "amount": 400.0, "seed_case": None,
     "source": "manual",
     "reason": "Control case: a correctly approved claim that must stay auto-approved.",
     "desc": "Abandoned shopping trolley rolled across the supermarket car park and dented the rear door "
             "of the insured car; the store accepted liability."},
    {"id": "CLM-SIM-09", "type": "comprehensive", "amount": 350.0, "seed_case": None,
     "source": "manual",
     "reason": "Control case: a correctly approved claim that must stay auto-approved.",
     "desc": "A neighbour's abandoned bicycle fell against the driver door overnight and scratched the paint; "
             "the neighbour has apologised and offered to pay."},
    {"id": "CLM-SIM-10", "type": "comprehensive", "amount": 300.0, "seed_case": None,
     "source": "manual",
     "reason": "Control case: a correctly approved claim that must stay auto-approved.",
     "desc": "Rear window cracked by a rock kicked up while driving past an abandoned building site; the "
             "site manager confirmed the hoarding was down."},
]
CLAIM_IDS = [c["id"] for c in CLAIMS]
_BY_ID = {c["id"]: c for c in CLAIMS}

# (claim_id, expected recommendation, critical)
SCENARIOS = {
    "parts-theft": {
        "title": "Parts stripping ring",
        "summary": "adds 'stripped' and 'cannibalized' as high-risk keywords",
        "claims": [("CLM-SIM-01", MANUAL, True), ("CLM-SIM-04", MANUAL, True)],
        "trap": False,
    },
    "staged-accident": {
        "title": "Staged collisions",
        "summary": "adds 'staged' as a high-risk keyword",
        "claims": [("CLM-SIM-03", MANUAL, True), ("CLM-SIM-06", MANUAL, True)],
        "trap": False,
    },
    "cloned-vehicle": {
        "title": "Cloned VINs",
        "summary": "adds 'cloned' as a high-risk keyword",
        "claims": [("CLM-SIM-02", MANUAL, True), ("CLM-SIM-07", MANUAL, True)],
        "trap": False,
    },
    "abandoned-lookalike": {
        "title": "Abandoned vehicle, with a look-alike trap",
        "summary": "adds 'abandoned' as a medium keyword, which also catches legitimate claims that merely "
                   "mention the word, so the gate REJECTS it (over-escalation rises)",
        "claims": [("CLM-SIM-05", REVIEW, False), ("CLM-SIM-08", AUTO, False), ("CLM-SIM-09", AUTO, False),
                   ("CLM-SIM-10", AUTO, False)],
        "trap": True,
    },
}
DEFAULT_SCENARIO = "parts-theft"


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
    for c in CLAIMS:
        doc = _clean(db[CLAIMS_COLLECTION].find_one({"claim_id": c["id"]}))
        if doc is None:
            doc = _claim_doc(c, user_id, policy, version, manifest)
            db[CLAIMS_COLLECTION].insert_one(dict(doc))
            created_claims += 1
        if doc["recommendation"] == MANUAL:
            notes.append(f"{c['id']}: live policy already routes this to manual review; the story is weaker")
        if c["seed_case"] is None:
            continue
        reason = c["reason"] or f"agent recommended {doc['recommendation']} but reviewer approved"
        case = court.nominate(c["id"], c["source"], reason)
        if c["seed_case"] == "filed" and case["status"] == "nominated":
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
    if not db[POLICY_COLLECTION].find_one({"policy_number": POLICY_NUMBER}):
        raise RuntimeError("demo data not found; run `demo seed` first")
    pending = candidates(store)
    if pending:
        vs = ", ".join(f"v{v['version']}" for v in pending)
        raise RuntimeError(f"candidate version {vs} is already awaiting approval; approve or reject it in the "
                           "Regression Court (Policy versions) first, then run the demo again")
    say = lambda t="": _say(out, pace, t, sleep)  # noqa: E731
    live_version, _ = _active(store)
    court = CourtDocket(db)
    say(f"=== Simulated incident stream: {sc['title']} (scenario {scenario}) ===")
    say(f"Live policy is v{live_version}. Everything below is a SIMULATION on demo data (CLM-SIM-*).")

    say("\n[1/6] Incident detected")
    claims = {}
    for cid, expected, _crit in sc["claims"]:
        claims[cid] = _clean(db[CLAIMS_COLLECTION].find_one({"claim_id": cid}))
        if claims[cid] is None:
            raise RuntimeError(f"claim {cid} missing; run `demo seed` first")
        cl = claims[cid]
        say(f"  {cid}  ${cl['claim_amount']:,.0f}  agent said {cl['recommendation']}, claim paid, later flagged wrong")
        say(f"      \"{cl['description']}\"")

    say("\n[2/6] Nominate to the Regression Court")
    cases = {}
    for cid, _e, _c in sc["claims"]:
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
    for cid, expected, critical in sc["claims"]:
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
    for c in sorted(claims, key=lambda c: c["claim_id"]):
        out(f"  {c['claim_id']}  ${c['claim_amount']:,.0f}  agent={c['recommendation']}  {c['resolution']}")
    cases = [c for c in CourtDocket(db).list_cases() if c["claim_id"] in CLAIM_IDS]
    out(f"demo court cases ({len(cases)}):")
    for c in sorted(cases, key=lambda c: c["claim_id"]):
        out(f"  {c['case_id']}  {c['claim_id']}  {c['status']}  ({c['source']})")
    active = store.get_active()
    out(f"live version: {('v%s' % active['version']) if active else 'none'}")
    cands = candidates(store)
    for v in cands:
        out(f"candidate version: v{v['version']}{' (demo)' if v.get('demo') else ''} - {v.get('reason', '')[:100]}")
    if not cands:
        out("candidate version: none")
    return {"policy": bool(pol), "claims": len(claims), "cases": len(cases),
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
    except RuntimeError as e:
        print(f"demo: {e}", file=sys.stderr)
        return 1
    return 0
