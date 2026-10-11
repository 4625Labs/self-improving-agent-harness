"""
Memory label policy - a FIXED invariant, never part of the evolvable harness manifest.

The memory service enforces label uniqueness per ORG, not per user. A semantic save with
upsert=True (the default) on a label that already exists overwrites that document and moves
it to the new caller, so a shared label lets one user's write erase another user's fact.
Every label this agent writes therefore goes through user_scoped_label().
"""

import re

SEPARATOR = "::"
_KEY_PATTERN = re.compile(r"[a-z0-9_-]{1,64}")


def normalize_key(key: str) -> str:
    """Lowercase a model-supplied key and turn spaces into underscores."""
    return key.strip().lower().replace(" ", "_")


def user_scoped_label(user_id: str, key: str) -> str:
    """Return a label that is unique to (user_id, key).

    The separator may not appear in either part, so two different (user, key) pairs can never
    produce the same label (e.g. user "a" + key "b_c" vs user "a_b" + key "c").

    Raises:
        ValueError: When user_id is empty, or either part is malformed.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id is required to build a memory label")
    if SEPARATOR in user_id:
        raise ValueError(f"user_id may not contain {SEPARATOR!r}")
    normalized = normalize_key(key)
    if not _KEY_PATTERN.fullmatch(normalized):
        raise ValueError("memory key must be 1-64 chars of a-z, 0-9, underscore, hyphen")
    return f"{user_id}{SEPARATOR}{normalized}"


def is_scoped_to(label: str, user_id: str) -> bool:
    """True when label was built for this user by user_scoped_label()."""
    return label.startswith(f"{user_id}{SEPARATOR}")
