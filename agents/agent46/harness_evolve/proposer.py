"""Proposers: propose(manifest, failures, split_summary) -> {"ops": [...], "rationale": str}.

Proposers only ever see DEV failures. Held-out data is reserved for the gate in loop.py.

Each proposer has a `surface` ('risk', 'tools' or 'all'); a proposer without one is treated as
'risk'. Tool-surface proposers receive tool-eval dev failures instead of risk-eval failures
(see loop._collect_proposals). CompositeProposer groups several proposers via `members`.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

from evals.run_eval import AUTO, RANK, load_scenarios

from agent_agent46.tool_access import INTENTS, REQUIRED_TOOLS, effective_tool_access

from .patch import OP_SCHEMA, PatchError, validate_ops

TOOL_OPS = ("disable_tool", "enable_tool")
RISK_OPS = tuple(k for k in OP_SCHEMA if k not in TOOL_OPS)
SURFACES = ("risk", "tools", "all")

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
    surface = "risk"

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


class ToolHeuristicProposer:
    """Offline, deterministic. Narrows tool access using dev tool-eval failures only.

    It loads no scenario files at all: everything it needs is in the dev failures it is handed.
    Never proposes a required tool, an already-disabled tool, or enable_tool (widening is human-only).
    """

    name = "tool-heuristic"
    surface = "tools"

    def propose(self, manifest: dict, failures: list[dict], split_summary: dict) -> dict:
        allow = {i: set(e["allow"]) for i, e in effective_tool_access(manifest)["intents"].items()}
        ops: list[dict[str, Any]] = []
        notes: list[str] = []
        seen: set[tuple[str, str]] = set()
        for f in sorted(failures, key=lambda f: f["id"]):
            intent = f.get("intent")
            if intent not in INTENTS:
                continue
            for tool in sorted(f.get("exposed_forbidden") or []):
                if (intent, tool) in seen or tool not in allow[intent] or tool in REQUIRED_TOOLS[intent]:
                    continue
                seen.add((intent, tool))
                ops.append({"op": "disable_tool", "intent": intent, "tool": tool})
                notes.append(f"{f['id']}: {tool} exposed for {intent} -> disable")
        validate_ops(ops)
        return {"ops": ops, "rationale": "; ".join(notes) or "no tool heuristic applies"}


class CompositeProposer:
    """Runs several proposers, each on its own surface. The loop feeds each member its own failures."""

    surface = "all"

    def __init__(self, members: list) -> None:
        self.members = list(members)
        self.name = "+".join(m.name for m in self.members)

    def propose(self, manifest: dict, failures: list[dict], split_summary: dict, tool_failures: list[dict] | None = None) -> dict:
        ops: list[dict[str, Any]] = []
        notes: list[str] = []
        for m in self.members:
            fs = (tool_failures or []) if getattr(m, "surface", "risk") == "tools" else failures
            out = m.propose(manifest, fs, split_summary)
            ops += out["ops"]
            notes.append(out.get("rationale", ""))
        validate_ops(ops)
        return {"ops": ops, "rationale": "; ".join(n for n in notes if n)}


class BudgetExceeded(RuntimeError):
    """Raised when the per-process LLM call limit (EVOLVE_LLM_MAX_CALLS) is reached."""


_CALLS = 0  # real (uncached) LLM calls made by this process
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".evolve_cache"  # gitignored


def reset_call_count() -> None:
    global _CALLS
    _CALLS = 0


def estimate_tokens(text: str) -> int:
    """Rough estimate (~4 chars per token)."""
    return (len(text) + 3) // 4


def _severity_key(f: dict) -> tuple:
    gap = RANK.get(f.get("expected"), 0) - RANK.get(f.get("got"), 0)
    crit = bool(f.get("critical_violation")) or f.get("severity") == "critical"
    return (not crit, -gap, str(f.get("id")))


def _default_client(prompt: str, model: str, max_tokens: int) -> str:
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    key = os.environ.get("LLM_API_KEY", "")
    if not key:
        raise RuntimeError("LLM_API_KEY is missing")
    llm = ChatOpenAI(model=model, api_key=SecretStr(key), use_responses_api=True, max_tokens=max_tokens)
    return llm.invoke(prompt).text


class LLMProposer:
    """LLM-backed proposer: one call per cycle, capped prompt and output, cached, call-limited.

    Network only when invoked; never used unless --proposer llm. `client(prompt, model, max_tokens)`
    is injectable for tests. Parse/validation failures raise; there is no retry.
    """

    name = "llm"
    surface = "all"

    def __init__(self, model: str | None = None, *, max_tokens: int = 600, max_failures: int = 8,
                 cache_dir: Path | str | None = None, max_calls: int | None = None, client=None) -> None:
        self._model = model
        self.max_tokens = max_tokens
        self.max_failures = max_failures
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self._max_calls = max_calls
        self._client = client or _default_client

    @property
    def model(self) -> str:
        from agent_agent46.llm import MODEL

        return self._model or MODEL

    def _cap(self, failures: list[dict]) -> tuple[list[dict], int]:
        ordered = sorted(failures, key=_severity_key)
        return ordered[: self.max_failures], max(0, len(ordered) - self.max_failures)

    def build_prompt(self, manifest: dict, failures: list[dict], split_summary: dict, tool_failures: list[dict] | None = None) -> str:
        dev = {s["id"]: s for s in load_scenarios("dev")}  # never heldout
        shown, omitted = self._cap(failures)
        cases = []
        for f in shown:
            sc = dev.get(f["id"])
            if sc:
                cases.append({"id": f["id"], "expected": f["expected"], "got": f["got"], "severity": f.get("severity"),
                              "claim": sc["claim"], "policy": sc["policy"],
                              "previous_claims_count": sc["previous_claims_count"]})
        tshown, tomitted = self._cap(tool_failures or [])
        omit_note = f"({omitted} more failing dev scenarios omitted; worst {len(shown)} shown, critical first)\n" if omitted else ""
        tomit_note = f"({tomitted} more tool failures omitted)\n" if tomitted else ""
        summary = {k: v for k, v in split_summary.items() if k != "failed_ids"}
        return (
            "You improve an insurance claim triage harness by patching its manifest.\n"
            f"Current manifest:\n{json.dumps(manifest, separators=(',', ':'))}\n\n"
            f"Dev summary: {json.dumps(summary)}\n"
            f"Failing dev scenarios:\n{json.dumps(cases, separators=(',', ':'))}\n{omit_note}\n"
            f"Allowed ops (exact fields, nothing else): {json.dumps({k: list(v) for k, v in OP_SCHEMA.items()})}\n"
            "level is 'high' or 'medium'. Never remove baseline keywords. Never ship critical claims as auto_approve.\n"
            f"Tool access (manifest 'tool_access'; ops disable_tool/enable_tool, intent in {list(INTENTS)}). "
            "disable_tool narrows a tool out of an intent's allow-list; enable_tool re-widens it and needs human approval. "
            "Never disable a required tool: "
            f"{json.dumps({k: sorted(v) for k, v in REQUIRED_TOOLS.items() if k != 'general'})}. "
            "The 'general' intent may never be narrowed.\n"
            f"Failing dev tool-access cases:\n{json.dumps(tshown, separators=(',', ':'))}\n{tomit_note}"
            'Reply with ONLY JSON: {"ops": [...], "rationale": "..."}'
        )

    def _cache_path(self, prompt: str) -> Path:
        h = hashlib.sha256(json.dumps([prompt, self.model]).encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _limit(self) -> int:
        if self._max_calls is not None:
            return self._max_calls
        return int(os.environ.get("EVOLVE_LLM_MAX_CALLS", "3"))

    def dry_run(self, manifest: dict, failures: list[dict], split_summary: dict, tool_failures: list[dict] | None = None) -> dict:
        prompt = self.build_prompt(manifest, failures, split_summary, tool_failures)
        return {"prompt": prompt, "prompt_chars": len(prompt), "estimated_prompt_tokens": estimate_tokens(prompt),
                "max_output_tokens": self.max_tokens, "model": self.model,
                "cached": self._cache_path(prompt).exists(), "calls_allowed": self._limit()}

    def propose(self, manifest: dict, failures: list[dict], split_summary: dict, tool_failures: list[dict] | None = None) -> dict:
        global _CALLS
        prompt = self.build_prompt(manifest, failures, split_summary, tool_failures)
        path = self._cache_path(prompt)
        raw = None
        if path.exists():
            try:
                raw = json.loads(path.read_text())["raw"]
            except (OSError, ValueError, KeyError):
                raw = None
        if raw is None:
            limit = self._limit()
            if _CALLS >= limit:
                raise BudgetExceeded(f"LLM call limit reached ({_CALLS}/{limit}); raise EVOLVE_LLM_MAX_CALLS to allow more")
            _CALLS += 1
            raw = self._client(prompt, self.model, self.max_tokens)
            result = self.parse(raw)  # only cache replies that parse and validate
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": self.model, "raw": raw}))
            return result
        return self.parse(raw)

    @staticmethod
    def parse(raw: str) -> dict:
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        m = fence or re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise PatchError("no JSON object in LLM reply")
        text = m.group(1) if fence else m.group(0)
        try:
            data = json.loads(text)
        except ValueError as e:
            raise PatchError(f"LLM reply is not valid JSON: {e}") from e
        if not isinstance(data, dict) or set(data) - {"ops", "rationale"} or "ops" not in data:
            raise PatchError("LLM reply must be exactly {ops, rationale}")
        validate_ops(data["ops"])
        return {"ops": data["ops"], "rationale": str(data.get("rationale", ""))}
