"""Persistent daily cost guardrail for the LLM proposer (on top of the per-process EVOLVE_LLM_MAX_CALLS).

Ledger: <cache_dir>/ledger.json = {"days": {"YYYY-MM-DD": {"calls": int, "tokens": int}}}, keyed by UTC date.
A real call does: reservation = check_and_reserve(est) -> call -> record(actual, reservation), or
release(reservation) if the call raised. Cache hits never touch the ledger. Token counts are estimates.

Limits: EVOLVE_LLM_MAX_CALLS_PER_DAY (default 10), EVOLVE_LLM_MAX_TOKENS_PER_DAY (default 50000).
Writes are atomic (temp file + os.replace) and serialized with a thread lock plus an flock where available.
A missing or corrupt ledger starts fresh with a logged warning.
"""

import contextlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:  # POSIX only; the thread lock alone covers in-process use elsewhere.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

log = logging.getLogger(__name__)

DEFAULT_MAX_CALLS_PER_DAY = 10
DEFAULT_MAX_TOKENS_PER_DAY = 50_000
LEDGER_NAME = "ledger.json"
LOCK_NAME = ".ledger.lock"
KEEP_DAYS = 30

_THREAD_LOCK = threading.Lock()


class BudgetExceeded(RuntimeError):
    """Raised when an LLM call limit is reached: per-process (EVOLVE_LLM_MAX_CALLS) or per-day."""


@dataclass(frozen=True)
class Reservation:
    day: str
    tokens: int


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return value


def _valid_days(data) -> dict:
    days = data["days"]
    if not isinstance(days, dict):
        raise ValueError("'days' is not an object")
    for day, e in days.items():
        if not isinstance(e, dict) or not isinstance(e.get("calls"), int) or not isinstance(e.get("tokens"), int):
            raise ValueError(f"malformed entry for {day!r}")
    return days


class DailyBudget:
    def __init__(self, cache_dir: Path | str, *, max_calls: int | None = None, max_tokens: int | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.path = self.cache_dir / LEDGER_NAME
        self.max_calls = max_calls if max_calls is not None else _int_env("EVOLVE_LLM_MAX_CALLS_PER_DAY", DEFAULT_MAX_CALLS_PER_DAY)
        self.max_tokens = max_tokens if max_tokens is not None else _int_env("EVOLVE_LLM_MAX_TOKENS_PER_DAY", DEFAULT_MAX_TOKENS_PER_DAY)

    # --- ledger I/O -------------------------------------------------------

    def _load(self) -> dict:
        try:
            return _valid_days(json.loads(self.path.read_text()))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("LLM budget ledger %s is unreadable (%s); starting fresh", self.path, e)
            return {}

    def _save(self, days: dict) -> None:
        for old in sorted(days)[:-KEEP_DAYS]:
            del days[old]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, prefix=".ledger-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"days": days}, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    @contextlib.contextmanager
    def _locked(self):
        with _THREAD_LOCK:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_dir / LOCK_NAME, "a+") as lf:
                if fcntl is not None:
                    fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lf, fcntl.LOCK_UN)

    # --- public API -------------------------------------------------------

    def remaining(self, day: str | None = None) -> dict:
        day = day or utc_today()
        e = self._load().get(day, {"calls": 0, "tokens": 0})
        return {
            "date": day,
            "calls_used": e["calls"],
            "calls_remaining": max(0, self.max_calls - e["calls"]),
            "tokens_used": e["tokens"],
            "tokens_remaining": max(0, self.max_tokens - e["tokens"]),
            "max_calls_per_day": self.max_calls,
            "max_tokens_per_day": self.max_tokens,
        }

    def check_and_reserve(self, est_tokens: int, day: str | None = None) -> Reservation:
        """Count one call and est_tokens against today's limits, or raise BudgetExceeded and change nothing."""
        day = day or utc_today()
        est = max(0, int(est_tokens))
        with self._locked():
            days = self._load()
            e = dict(days.get(day, {"calls": 0, "tokens": 0}))
            if e["calls"] + 1 > self.max_calls:
                raise BudgetExceeded(
                    f"daily LLM call limit reached for {day} ({e['calls']}/{self.max_calls} calls); "
                    "raise EVOLVE_LLM_MAX_CALLS_PER_DAY to allow more"
                )
            if e["tokens"] + est > self.max_tokens:
                raise BudgetExceeded(
                    f"daily LLM token limit would be exceeded for {day} ({e['tokens']} used + {est} estimated "
                    f"> {self.max_tokens}); raise EVOLVE_LLM_MAX_TOKENS_PER_DAY to allow more"
                )
            e["calls"] += 1
            e["tokens"] += est
            days[day] = e
            self._save(days)
        return Reservation(day, est)

    def record(self, actual_tokens: int, reservation: Reservation) -> None:
        """A reply came back: replace the reserved estimate with the actual token count."""
        self._adjust(reservation.day, 0, int(actual_tokens) - reservation.tokens)

    def release(self, reservation: Reservation) -> None:
        """The call failed without a reply: give back the call and its reserved tokens."""
        self._adjust(reservation.day, -1, -reservation.tokens)

    def _adjust(self, day: str, d_calls: int, d_tokens: int) -> None:
        with self._locked():
            days = self._load()
            e = dict(days.get(day, {"calls": 0, "tokens": 0}))
            e["calls"] = max(0, e["calls"] + d_calls)
            e["tokens"] = max(0, e["tokens"] + d_tokens)
            days[day] = e
            self._save(days)
