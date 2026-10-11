"""Append-only audit trail plus the confirmation and remote-host guards for host-side evolve tooling.

Nothing here blocks outright: destructive actions need `confirmed=True` (CLI `--yes`, UI checkbox),
and a non-local Mongo URI needs an explicit override. Every attempt, refused or not, is recorded.
Auditing never raises into the caller. There are deliberately no update/delete helpers.
"""

import getpass
import logging
import os
import socket
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import unquote

log = logging.getLogger(__name__)

AUDIT_COLLECTION = "audit_log"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
ALLOW_REMOTE_ENV = "EVOLVE_ALLOW_REMOTE"


class ConfirmationRequired(ValueError):
    """A destructive action was attempted without explicit confirmation."""


class RemoteRefused(RuntimeError):
    """The Mongo URI is not local and no override was given."""


def os_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return "unknown"


def record(db, action: str, actor: str, target: Any, detail: dict | None, outcome: str) -> None:
    """Append one audit document. Never raises."""
    try:
        db[AUDIT_COLLECTION].insert_one({
            "ts": datetime.now(timezone.utc).isoformat(), "action": action, "actor": actor or "unknown",
            "target": target, "detail": dict(detail or {}), "outcome": outcome, "host": socket.gethostname()})
    except Exception as e:  # noqa: BLE001
        log.warning("audit write failed (%s %s %s): %s", action, target, outcome, e)


def recent(db, n: int = 10) -> list[dict]:
    """Newest-first audit rows for display. Never raises."""
    try:
        cur = db[AUDIT_COLLECTION].find({})
        if isinstance(cur, list):  # in-memory fakes return plain lists
            rows = sorted(cur, key=lambda r: r.get("ts", ""), reverse=True)[:n]
        else:
            rows = list(cur.sort("ts", -1).limit(n))
        return [{k: v for k, v in r.items() if k != "_id"} for r in rows]
    except Exception as e:  # noqa: BLE001
        log.warning("audit read failed: %s", e)
        return []


def run_confirmed(db, action: str, actor: str, target: Any, detail: dict | None, confirmed: bool,
                  fn: Callable[[], Any]) -> Any:
    """Refuse (audited) unless confirmed; otherwise run fn and audit ok / error."""
    if not confirmed:
        record(db, action, actor, target, detail, "refused: confirmation required")
        raise ConfirmationRequired(
            f"{action} changes live state and needs explicit confirmation (CLI --yes, UI confirm checkbox, "
            "library confirmed=True); nothing was changed")
    try:
        result = fn()
    except Exception as e:
        record(db, action, actor, target, detail, f"error: {e}")
        raise
    record(db, action, actor, target, detail, "ok")
    return result


def uri_hosts(uri: str) -> list[str]:
    if "://" not in uri:
        return [uri.split("/")[0].split(":")[0]]
    rest = uri.split("://", 1)[1].split("/", 1)[0].split("?", 1)[0]
    rest = rest.rsplit("@", 1)[-1]
    hosts = []
    for h in rest.split(","):
        h = unquote(h.strip())
        hosts.append(h[1:h.index("]")] if h.startswith("[") else h.split(":")[0])
    return hosts


def is_local_uri(uri: str) -> bool:
    if uri.startswith("mongodb+srv://"):
        return False
    hosts = uri_hosts(uri)
    return bool(hosts) and all(h.lower() in LOCAL_HOSTS for h in hosts)


def check_remote(uri: str, allow_remote: bool = False) -> None:
    """Call before any mutating command. Raises RemoteRefused for a non-local URI without override."""
    if is_local_uri(uri) or allow_remote or os.environ.get(ALLOW_REMOTE_ENV) == "1":
        return
    raise RemoteRefused(
        f"refusing to modify a non-local Mongo host ({', '.join(uri_hosts(uri))}); "
        f"pass --allow-remote or set {ALLOW_REMOTE_ENV}=1 to override. Read-only commands still work")
