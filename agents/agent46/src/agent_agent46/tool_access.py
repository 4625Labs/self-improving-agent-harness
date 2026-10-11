"""
Tool access - least-privilege tool allow-lists per intent, evolvable only inside human-owned limits.

What evolves (in the manifest, key "tool_access"): for each intent, which tools the model is
shown and may call. A proposer may NARROW a list freely (low risk) and may only re-widen it with
human approval.

What is fixed here, in code:
- ALL_TOOLS: the tools agent.yaml already places in sandboxes. The manifest can never add a tool
  that is not in this set, and a test keeps it in sync with agent.yaml.
- REQUIRED_TOOLS: per intent, the tools that must always stay allowed so a workflow cannot be
  stranded halfway (for example human_review and resolve_claim inside a claim).
- classify_intent: a deterministic classifier. Ambiguous or unknown messages map to "general",
  whose allow-list may never be narrowed, so a misclassification can only fail open.
"""

import copy
import re
from collections.abc import Iterable, Sequence
from typing import Any

ALL_TOOLS: frozenset[str] = frozenset(
    {
        "human_review",
        "send_notification",
        "lookup_policy",
        "get_quote",
        "create_policy",
        "list_customer_policies",
        "file_claim",
        "analyze_claim_risk",
        "check_claim_status",
        "list_customer_claims",
        "resolve_claim",
        "save_customer_info",
        "recall_customer_info",
        "save_conversation_summary",
        "recall_past_conversations",
        "explain_insurance_term",
        "get_coverage_options",
    }
)

INTENTS: tuple[str, ...] = ("quote", "claims", "info", "general")

_MEMORY_READ = {"recall_customer_info", "recall_past_conversations"}
_MEMORY_WRITE = {"save_customer_info", "save_conversation_summary"}

REQUIRED_TOOLS: dict[str, frozenset[str]] = {
    "claims": frozenset(
        {
            "list_customer_policies",
            "lookup_policy",
            "file_claim",
            "analyze_claim_risk",
            "human_review",
            "resolve_claim",
            "send_notification",
            "check_claim_status",
            "list_customer_claims",
        }
        | _MEMORY_READ
        | _MEMORY_WRITE
    ),
    "quote": frozenset(
        {
            "get_quote",
            "create_policy",
            "list_customer_policies",
            "lookup_policy",
            "get_coverage_options",
            "explain_insurance_term",
        }
        | _MEMORY_READ
        | _MEMORY_WRITE
    ),
    "info": frozenset({"explain_insurance_term", "get_coverage_options"} | _MEMORY_READ),
    "general": ALL_TOOLS,
}

# Tools whose presence outside their own workflow is a privilege risk: they change records or
# finalize decisions. Evaluators treat leaving these allowed in the wrong intent as critical.
SENSITIVE_TOOLS: frozenset[str] = frozenset(
    {"resolve_claim", "human_review", "create_policy", "file_claim", "send_notification"}
)

# Tools called earlier in a thread that show a claim is in progress.
CLAIM_PIPELINE_TOOLS: frozenset[str] = frozenset(
    {
        "file_claim",
        "analyze_claim_risk",
        "human_review",
        "resolve_claim",
        "check_claim_status",
        "list_customer_claims",
    }
)

# While a claim is in flight (filed, notification not yet sent) these stay allowed in EVERY
# intent, so an unrelated question mid-claim can never strand resolve_claim or human_review.
CLAIM_FLOW_TOOLS: frozenset[str] = frozenset(
    {
        "list_customer_policies",
        "lookup_policy",
        "analyze_claim_risk",
        "human_review",
        "resolve_claim",
        "send_notification",
        "check_claim_status",
    }
)

_CLAIMS_KEYWORDS = (
    "claim", "accident", "collision", "crash", "damage", "stolen", "theft", "fire", "flood",
    "vandal", "vandalism", "vandalized", "vandalised", "arson", "hit and run", "hit-and-run", "totaled", "total loss", "police report",
    "break-in", "windshield", "adjuster", "wreck", "wrecked", "rear-ended", "rear ended",
    "fender bender", "wildfire",
)
_QUOTE_KEYWORDS = (
    "quote", "buy", "buying", "purchase", "purchasing", "sign up", "new policy", "get insured",
    "insure my", "how much would", "price",
)
_INFO_KEYWORDS = (
    "what is", "what's", "what does", "what are", "explain", "meaning of", "define",
    "difference between", "how does", "deductible",
)


def _matches(text: str, keywords: tuple[str, ...]) -> bool:
    """Whole-word match with a simple plural or past-tense ending (claim, claims, damaged).

    A bare prefix match would turn "fireplace" into a claim and "priceless" into a quote.
    """
    return any(re.search(r"\b" + re.escape(k) + r"(?:s|es|d|ed)?\b", text) for k in keywords)


def classify_intent(latest_user_text: str, called_tools: Iterable[str] = ()) -> str:
    """Return one of INTENTS. Deterministic; ambiguous or unknown input returns "general".

    Exactly one intent group matching decides the intent. Matching several groups is ambiguous
    and fails open to "general". With no match, a claim already in progress (a claim-pipeline
    tool was called earlier in the thread) stays "claims"; otherwise the intent is "general".
    """
    text = (latest_user_text or "").lower()
    hits = [
        intent
        for intent, kws in (
            ("claims", _CLAIMS_KEYWORDS),
            ("quote", _QUOTE_KEYWORDS),
            ("info", _INFO_KEYWORDS),
        )
        if _matches(text, kws)
    ]
    if len(hits) == 1:
        return hits[0]
    if not hits and CLAIM_PIPELINE_TOOLS & set(called_tools):
        return "claims"
    return "general"


def claim_in_flight(called_tools: Sequence[str]) -> bool:
    """True when a claim was filed and its outcome notification has not been sent since.

    called_tools must be in call order (oldest first).
    """
    last_filed = max((i for i, t in enumerate(called_tools) if t == "file_claim"), default=-1)
    last_notified = max((i for i, t in enumerate(called_tools) if t == "send_notification"), default=-1)
    return last_filed > last_notified


def baseline_tool_access() -> dict[str, Any]:
    """Every intent allows every tool: the behavior the agent shipped with."""
    return {"intents": {intent: {"allow": sorted(ALL_TOOLS)} for intent in INTENTS}}


def validate_tool_access(tool_access: Any) -> list[str]:
    """Return invariant violations for a manifest's tool_access value (empty list = valid)."""
    if not isinstance(tool_access, dict) or set(tool_access) != {"intents"}:
        return ["tool_access must be an object with exactly the key 'intents'"]
    intents = tool_access["intents"]
    if not isinstance(intents, dict) or set(intents) != set(INTENTS):
        return [f"tool_access.intents must define exactly these intents: {list(INTENTS)}"]
    errors: list[str] = []
    for intent in INTENTS:
        entry = intents[intent]
        if not isinstance(entry, dict) or set(entry) != {"allow"}:
            errors.append(f"tool_access.intents.{intent} must be an object with only 'allow'")
            continue
        allow = entry["allow"]
        if not isinstance(allow, list) or not all(isinstance(t, str) for t in allow):
            errors.append(f"tool_access.intents.{intent}.allow must be a list of tool names")
            continue
        if len(set(allow)) != len(allow):
            errors.append(f"tool_access.intents.{intent}.allow has duplicates")
        unknown = set(allow) - ALL_TOOLS
        if unknown:
            errors.append(
                f"tool_access.intents.{intent}.allow names tools that do not exist: {sorted(unknown)}"
            )
        missing = REQUIRED_TOOLS[intent] - set(allow)
        if missing:
            errors.append(
                f"tool_access.intents.{intent}.allow may not remove required tools: {sorted(missing)}"
            )
    return errors


def effective_tool_access(manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest's tool_access, or the baseline when an older manifest has none."""
    ta = manifest.get("tool_access")
    return copy.deepcopy(ta) if ta is not None else baseline_tool_access()


def allowed_tools(manifest: dict[str, Any], intent: str) -> frozenset[str]:
    """Tools the model may see and call for this intent under this manifest.

    An unknown intent is treated as "general". The result is always a subset of ALL_TOOLS.
    """
    if intent not in INTENTS:
        intent = "general"
    allow = effective_tool_access(manifest)["intents"][intent]["allow"]
    return frozenset(allow) & ALL_TOOLS


def resolve_tool_access(
    manifest: dict[str, Any], latest_user_text: str, called_tools: Sequence[str] = ()
) -> tuple[str, frozenset[str]]:
    """The single place that decides (intent, allowed tools) for a conversation state.

    The agent, the runtime guard and the evaluator all call this, so they cannot disagree.
    """
    intent = classify_intent(latest_user_text, called_tools)
    allowed = allowed_tools(manifest, intent)
    if claim_in_flight(called_tools):
        allowed = allowed | CLAIM_FLOW_TOOLS
    return intent, allowed
