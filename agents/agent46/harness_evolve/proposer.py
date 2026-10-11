"""Proposers: propose(manifest, failures, split_summary) -> {"ops": [...], "rationale": str}.

Proposers only ever see DEV failures. Held-out data is reserved for the gate in loop.py.
"""

import json
import os
import re
from typing import Any, Protocol

from evals.run_eval import AUTO, RANK, load_scenarios

from .patch import OP_SCHEMA, PatchError, validate_ops

# Phrases that signal fraud/severity. Matched against dev failure descriptions only.
RISK_LEXICON = [
    "arson", "submerged", "hit and run", "vandalism", "abandoned", "no witnesses",
    "owner away", "stripped", "cloned", "staged",
]


class Proposer(Protocol):
    name: str

    def propose(self, manifest: dict, failures: list[dict], split_summary: dict) -> dict: ...


class HeuristicProposer:
    """Offline, deterministic. Reads dev scenarios (by failure id) and nothing else."""

    name = "heuristic"

    def __init__(self) -> None:
        self._dev = {s["id"]: s for s in load_scenarios("dev")}  # never heldout

    def propose(self, manifest: dict, failures: list[dict], split_summary: dict) -> dict:
        ops: list[dict[str, Any]] = []
        notes: list[str] = []
        known = {lvl: {k.lower() for k in manifest["keywords"][lvl]} for lvl in ("high", "medium")}
        seen: set[tuple[str, str]] = set()

        # Under-escalated failures: add the lexicon phrases found in their descriptions.
        for f in sorted(failures, key=lambda f: f["id"]):
            sc = self._dev.get(f["id"])
            if sc is None or RANK.get(f["got"], 0) >= RANK[f["expected"]]:
                continue
            level = "high" if f["expected"] == "manual_review_required" else "medium"
            text = f"{sc['claim'].get('description', '')} {sc['claim'].get('claim_type', '')}".lower()
            for phrase in RISK_LEXICON:
                if phrase in text and (level, phrase) not in seen and not any(phrase in k for k in known[level] | known["high"]):
                    seen.add((level, phrase))
                    ops.append({"op": "add_keyword", "level": level, "keyword": phrase})
                    notes.append(f"{f['id']} under-escalated ({f['got']} vs {f['expected']}): keyword {phrase!r} -> {level}")

        # Over-escalated low-amount claims on elevated-score policies: raise medium_score.
        t = manifest["thresholds"]
        scores = []
        for f in failures:
            sc = self._dev.get(f["id"])
            if sc is None or f["expected"] != AUTO or f["got"] == AUTO:
                continue
            ps = sc["policy"].get("risk_score", 0.3)
            if sc["claim"]["claim_amount"] < t["auto_approve_limit"] and t["medium_score"] <= ps < t["high_score"]:
                scores.append((f["id"], ps))
        if scores:
            new = round(max(p for _, p in scores) + 0.05, 2)
            if t["medium_score"] < new < t["high_score"]:
                ops.append({"op": "set_threshold", "name": "medium_score", "value": new})
                notes.append(f"over-escalated {[i for i, _ in scores]} on elevated-score policies: medium_score -> {new}")

        validate_ops(ops)
        return {"ops": ops, "rationale": "; ".join(notes) or "no heuristic applies"}


class LLMProposer:
    """LLM-backed proposer. Network only when invoked; never used unless --proposer llm."""

    name = "llm"

    def __init__(self, model: str | None = None) -> None:
        self._model = model

    def build_prompt(self, manifest: dict, failures: list[dict], split_summary: dict) -> str:
        dev = {s["id"]: s for s in load_scenarios("dev")}
        cases = []
        for f in failures:
            sc = dev.get(f["id"])
            if sc:
                cases.append({"failure": f, "claim": sc["claim"], "policy": sc["policy"],
                              "previous_claims_count": sc["previous_claims_count"]})
        return (
            "You improve an insurance claim triage harness by patching its manifest.\n"
            f"Current manifest:\n{json.dumps(manifest, indent=1)}\n\n"
            f"Dev summary: {json.dumps(split_summary)}\n"
            f"Failing dev scenarios:\n{json.dumps(cases, indent=1)}\n\n"
            f"Allowed ops (exact fields, nothing else): {json.dumps({k: list(v) for k, v in OP_SCHEMA.items()})}\n"
            "level is 'high' or 'medium'. Never remove baseline keywords. Never ship critical claims as auto_approve.\n"
            'Reply with ONLY JSON: {"ops": [...], "rationale": "..."}'
        )

    def propose(self, manifest: dict, failures: list[dict], split_summary: dict) -> dict:
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        key = os.environ.get("LLM_API_KEY", "")
        if not key:
            raise RuntimeError("LLM_API_KEY is missing")
        from agent_agent46.llm import MODEL

        llm = ChatOpenAI(model=self._model or MODEL, api_key=SecretStr(key), use_responses_api=True)
        raw = llm.invoke(self.build_prompt(manifest, failures, split_summary)).text
        return self.parse(raw)

    @staticmethod
    def parse(raw: str) -> dict:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise PatchError("no JSON object in LLM reply")
        data = json.loads(m.group(0))
        if not isinstance(data, dict) or set(data) - {"ops", "rationale"} or "ops" not in data:
            raise PatchError("LLM reply must be exactly {ops, rationale}")
        validate_ops(data["ops"])
        return {"ops": data["ops"], "rationale": str(data.get("rationale", ""))}
