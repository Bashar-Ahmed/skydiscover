"""Usage-limit detection and wait-until-reset coordination.

Subscription-based backends (Claude Code CLI) and metered APIs both reject
requests once a quota is exhausted. The default behaviour elsewhere in this
package is to retry a few times and then let the iteration fail, which silently
discards work: on a plan with a five-hour window, a run that trips the limit
loses every remaining iteration.

This module instead treats a usage limit as *scheduled downtime*. When a call is
rejected, the reset instant is parsed out of the error (or its response
headers), recorded in a process-wide gate, and every caller sleeps until it
passes rather than burning retries.

The gate deliberately uses ``threading.Lock`` plus timestamp polling rather than
``asyncio`` primitives. Parts of this codebase drive coroutines from worker
threads with their own event loops (see ``context_builder/evox/builder.py``),
and an ``asyncio.Event`` bound to one loop cannot be awaited from another.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional

logger = logging.getLogger("skydiscover.llm")

# Wait at most this long for a single limit before retrying the request anyway.
# A longer real limit (e.g. a weekly cap) is simply re-detected and re-waited,
# so this bounds a bad parse without capping legitimate downtime.
DEFAULT_MAX_WAIT_SECONDS = 24 * 60 * 60

# Used when a limit is recognised but carries no machine-readable reset time.
DEFAULT_FALLBACK_WAIT_SECONDS = 15 * 60

# Slack added after the parsed reset so we do not race the server's own clock.
RESET_GRACE_SECONDS = 5.0

# A bare number at or above this is an absolute epoch, not a relative offset
# (~2001 in seconds). The millis variant is the same instant expressed in ms.
_EPOCH_THRESHOLD_SECONDS = 1_000_000_000
_EPOCH_THRESHOLD_MILLIS = 1_000_000_000_000

# Two tiers, because the correct response differs.
#
# HARD quota: the plan or account allowance is spent. There is nothing to do
# but wait, so it is safe to park the process even when no reset time is given.
_HARD_QUOTA_PATTERNS = (
    r"usage limit",
    r"quota (?:exceeded|exhausted)",
    r"exceeded your current quota",
    r"insufficient_quota",
    r"you(?:'ve| have) (?:reached|hit) your .{0,40}limit",
    r"\b(?:weekly|daily|hourly|monthly|session|plan|account)\s+limit\b",
    r"\b\d+\s*-?\s*hour\s+limit\b",
    r"\blimit (?:will )?reset",
    r"\blimit resets\b",
    r"upgrade to claude (?:max|pro)\b",
)

# SOFT rate limit: a per-minute throughput cap or transient overload. These
# clear in seconds. Parking the whole process for the fallback window would be
# far worse than the ordinary retry path, so a soft limit engages the gate ONLY
# when the provider actually told us when to come back.
_SOFT_RATE_LIMIT_PATTERNS = (
    r"rate[_ ]limit",
    r"too many requests",
    r"requests per (?:min|minute)",
    r"tokens per (?:min|minute)",
)

# Note: `overloaded_error` (HTTP 529) is deliberately absent from both lists.
# It signals server capacity, not an exhausted quota, and belongs to the
# ordinary exponential-retry path.

_HARD_QUOTA_RE = re.compile("|".join(_HARD_QUOTA_PATTERNS), re.IGNORECASE)
_SOFT_RATE_LIMIT_RE = re.compile("|".join(_SOFT_RATE_LIMIT_PATTERNS), re.IGNORECASE)

# `Claude AI usage limit reached|1770000000` — the CLI's machine-readable form.
_PIPE_EPOCH_RE = re.compile(r"usage limit reached\s*\|\s*(\d{9,13})", re.IGNORECASE)

# Bare epoch seconds/millis appearing anywhere after a limit phrase.
_BARE_EPOCH_RE = re.compile(r"\b(1[6-9]\d{8}|2\d{9})(?:\d{3})?\b")

# Duration units, longest-first so that "minutes" cannot be consumed as "m"
# (which would then fail the trailing word boundary) and "ms" wins over "m".
# OpenAI's standard 429 body ends "Please try again in 120ms." and must parse
# as 0.12s, not 120 minutes.
_DURATION_UNITS = r"milliseconds?|millis|ms|seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|s|m|h|d"

# "in 42 minutes", "in 3 hours", "retry after 30 seconds", "try again in 120ms".
_RELATIVE_RE = re.compile(
    r"(?:in|after|for)\s+(\d+(?:\.\d+)?)\s*(" + _DURATION_UNITS + r")\b",
    re.IGNORECASE,
)

# Compound Go-style durations inside prose: "Please try again in 6m0s."
# The single-unit pattern above cannot match these because the unit is followed
# immediately by another digit rather than a word boundary.
_TEXT_GO_DURATION_RE = re.compile(
    r"(?:in|after)\s+((?:\d+(?:\.\d+)?\s*(?:ms|s|m|h|d)){2,})\b",
    re.IGNORECASE,
)

# "resets at 3pm", "resets 15:04", "try again at 9:30 AM".
#
# This branch is the most dangerous one in the module: it turns a bare integer
# into an hour-of-day, and if that hour has passed it rolls to tomorrow. Read
# loosely it would parse the "120" of "Please try again in 120ms." as 12:00 and
# stall the process for most of a day. Two guards prevent that:
#   * real time evidence is REQUIRED -- either an am/pm marker or a :MM group;
#   * a negative lookahead rejects a number followed by a duration unit.
# The relative-duration branch also runs first, so well-formed hints never
# reach here at all.
_CLOCK_RE = re.compile(
    r"(?:reset(?:s|ting)?|try again|available again|back)\D{0,20}?"
    r"(\d{1,2})"
    r"(?!\s*(?:" + _DURATION_UNITS + r")\b)"
    r"(?::(\d{2})\s*(am|pm)?|\s*(am|pm))",
    re.IGNORECASE,
)

_UNIT_SECONDS = {
    "ms": 0.001,
    "milli": 0.001,
    "millis": 0.001,
    "millisecond": 0.001,
    "s": 1,
    "sec": 1,
    "second": 1,
    "m": 60,
    "min": 60,
    "minute": 60,
    "h": 3600,
    "hr": 3600,
    "hour": 3600,
    "d": 86400,
    "day": 86400,
}

# Go-style durations used by OpenAI rate-limit headers, e.g. "6m0s", "1.5s".
_GO_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)")

_RESET_HEADERS = (
    "retry-after",
    "anthropic-ratelimit-unified-reset",
    "anthropic-ratelimit-requests-reset",
    "anthropic-ratelimit-tokens-reset",
    "anthropic-ratelimit-input-tokens-reset",
    "anthropic-ratelimit-output-tokens-reset",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "x-ratelimit-reset",
)


class UsageLimitError(Exception):
    """Raised when a provider rejects a call because a quota is exhausted.

    ``reset_at`` is an absolute ``time.time()``-style timestamp when known.
    """

    def __init__(
        self,
        message: str,
        reset_at: Optional[float] = None,
        source: Optional[str] = None,
    ):
        super().__init__(message)
        self.reset_at = reset_at
        self.source = source


# ──────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────


def is_hard_quota(text: Optional[str]) -> bool:
    """Whether *text* says the plan/account allowance itself is exhausted."""
    return bool(text) and bool(_HARD_QUOTA_RE.search(text))


def is_soft_rate_limit(text: Optional[str], status_code: Optional[int] = None) -> bool:
    """Whether this is a transient throughput cap rather than a spent quota."""
    if status_code == 429:
        return True
    return bool(text) and bool(_SOFT_RATE_LIMIT_RE.search(text))


def looks_like_usage_limit(text: Optional[str], status_code: Optional[int] = None) -> bool:
    """Whether *text* / *status_code* indicate a quota or rate rejection.

    Note this is broader than what engages the gate — see :func:`parse_usage_limit`,
    which only parks the process for a hard quota or a soft limit that carries a
    reset time.
    """
    return is_hard_quota(text) or is_soft_rate_limit(text, status_code)


def _lookup_unit(unit: str) -> Optional[float]:
    """Map a duration unit token (possibly plural) to seconds."""
    normalized = unit.lower()
    if normalized in _UNIT_SECONDS:
        return _UNIT_SECONDS[normalized]
    if normalized.endswith("s") and normalized[:-1] in _UNIT_SECONDS:
        return _UNIT_SECONDS[normalized[:-1]]
    return None


def _parse_go_duration(value: str) -> Optional[float]:
    matches = _GO_DURATION_RE.findall(value)
    if not matches:
        return None
    total = 0.0
    for amount, unit in matches:
        seconds = 0.001 if unit == "ms" else _UNIT_SECONDS.get(unit)
        if seconds is None:
            return None
        total += float(amount) * seconds
    return total


def _parse_header_value(value: str, now: float) -> Optional[float]:
    """Interpret one rate-limit header as an absolute reset timestamp."""
    value = (value or "").strip()
    if not value:
        return None

    # Plain number. Small values are a relative offset (Retry-After: 120);
    # large ones are an absolute epoch (X-RateLimit-Reset: 1800000000). Treating
    # an epoch as an offset would park the process for decades.
    try:
        number = float(value)
    except ValueError:
        pass
    else:
        # Millis first: an epoch in milliseconds also clears the seconds
        # threshold, so checking seconds first would mis-scale it by 1000x.
        if number >= _EPOCH_THRESHOLD_MILLIS:
            return number / 1000.0
        if number >= _EPOCH_THRESHOLD_SECONDS:
            return number
        return now + number

    # Go-style duration (x-ratelimit-reset-requests: 6m0s)
    duration = _parse_go_duration(value)
    if duration is not None:
        return now + duration

    # RFC 3339 / ISO 8601 (anthropic-ratelimit-*-reset)
    iso = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        pass

    # HTTP-date (Retry-After: Wed, 21 Oct 2026 07:28:00 GMT)
    try:
        return parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError):
        return None


def parse_reset_from_headers(
    headers: Optional[Dict[str, Any]], now: Optional[float] = None
) -> Optional[float]:
    """Return the earliest usable reset timestamp from response *headers*."""
    if not headers:
        return None
    now = now if now is not None else time.time()

    lowered = {}
    try:
        for key, value in dict(headers).items():
            lowered[str(key).lower()] = value
    except (TypeError, ValueError):
        return None

    candidates = []
    for name in _RESET_HEADERS:
        if name in lowered:
            parsed = _parse_header_value(str(lowered[name]), now)
            if parsed is not None and parsed > now:
                candidates.append(parsed)

    return min(candidates) if candidates else None


def parse_reset_from_text(text: Optional[str], now: Optional[float] = None) -> Optional[float]:
    """Extract an absolute reset timestamp from a provider error message."""
    if not text:
        return None
    now = now if now is not None else time.time()

    # 1. The CLI's explicit "usage limit reached|<epoch>" form.
    match = _PIPE_EPOCH_RE.search(text)
    if match:
        raw = match.group(1)
        epoch = float(raw) / 1000.0 if len(raw) > 10 else float(raw)
        if epoch > now:
            return epoch

    # 2. A relative duration: "try again in 12 minutes".
    match = _RELATIVE_RE.search(text)
    if match:
        unit = _lookup_unit(match.group(2))
        if unit:
            return now + float(match.group(1)) * unit

    # 3. A compound Go-style duration: "try again in 6m0s".
    match = _TEXT_GO_DURATION_RE.search(text)
    if match:
        duration = _parse_go_duration(match.group(1))
        if duration is not None:
            return now + duration

    # 4. A bare epoch anywhere in the message.
    match = _BARE_EPOCH_RE.search(text)
    if match:
        raw = match.group(0)
        epoch = float(raw) / 1000.0 if len(raw) > 10 else float(raw)
        if now < epoch < now + 30 * 86400:
            return epoch

    # 5. A wall-clock time: "resets at 3pm". Interpreted in local time; if the
    #    result is already past, assume it means the next occurrence.
    match = _CLOCK_RE.search(text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        # Group 3 is the meridiem of the "H:MM am" form, group 4 of a bare "Ham".
        meridiem = (match.group(3) or match.group(4) or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            local_now = datetime.fromtimestamp(now)
            target = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target.timestamp() <= now:
                target += timedelta(days=1)
            return target.timestamp()

    return None


def parse_usage_limit(
    text: Optional[str],
    status_code: Optional[int] = None,
    headers: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
) -> Optional[float]:
    """Return when to resume if the process should park, else None.

    Returning None does not mean "not a rate limit" — it means the ordinary
    retry path should handle it. The distinction matters:

    * **Hard quota** (plan allowance spent): park, and fall back to
      ``DEFAULT_FALLBACK_WAIT_SECONDS`` when no reset time is given, because
      retrying immediately cannot succeed.
    * **Soft rate limit** (per-minute cap, transient 429): park only when the
      provider said when to come back. A bare 429 with no reset clears in
      seconds, so parking every LLM call in the process for the fallback
      window would be far worse than one exponential-backoff retry.
    """
    now = now if now is not None else time.time()
    hard = is_hard_quota(text)

    if not hard and not is_soft_rate_limit(text, status_code):
        return None

    reset_at = parse_reset_from_headers(headers, now) or parse_reset_from_text(text, now)
    if reset_at is not None:
        return reset_at

    return now + DEFAULT_FALLBACK_WAIT_SECONDS if hard else None


def extract_error_details(error: BaseException) -> Dict[str, Any]:
    """Pull message text, HTTP status, and headers out of a provider exception."""
    parts = [str(error)]
    status_code = getattr(error, "status_code", None)
    headers = None

    body = getattr(error, "body", None)
    if body is not None:
        parts.append(str(body))

    response = getattr(error, "response", None)
    if response is not None:
        if status_code is None:
            status_code = getattr(response, "status_code", None)
        headers = getattr(response, "headers", None)
        text = getattr(response, "text", None)
        if text:
            parts.append(str(text))

    return {"text": " ".join(parts), "status_code": status_code, "headers": headers}


# ──────────────────────────────────────────────────────────────────────────
# Gate
# ──────────────────────────────────────────────────────────────────────────


class UsageLimitGate:
    """Process-wide barrier held while a provider quota is exhausted.

    One rejected call blocks every other caller too: on a subscription plan the
    quota is per-account, so letting siblings keep firing only produces more
    rejections and, on some providers, extends the lockout.
    """

    def __init__(self, max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS):
        self._lock = threading.Lock()
        self._blocked_until = 0.0
        self._reason = ""
        self._generation = 0
        self.max_wait_seconds = max_wait_seconds

    # -- state ---------------------------------------------------------

    def remaining(self) -> float:
        """Seconds left on the current block (0.0 when clear)."""
        with self._lock:
            return max(0.0, self._blocked_until - time.time())

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def note_limit(self, reset_at: Optional[float], reason: str = "") -> float:
        """Record a usage limit. Returns the seconds now left to wait.

        Extends an existing block but never shortens one — a stricter limit hit
        by a sibling call must not be cleared early by a laxer one.
        """
        now = time.time()
        if reset_at is None:
            reset_at = now + DEFAULT_FALLBACK_WAIT_SECONDS

        capped = min(reset_at, now + self.max_wait_seconds) + RESET_GRACE_SECONDS

        with self._lock:
            if capped > self._blocked_until:
                self._blocked_until = capped
                self._reason = reason
                self._generation += 1
                announce = True
            else:
                announce = False
            remaining = max(0.0, self._blocked_until - now)

        if announce:
            logger.warning(
                "Usage limit hit%s. Pausing all LLM calls for %s (until %s).",
                f": {reason}" if reason else "",
                _format_duration(remaining),
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now + remaining)),
            )
        return remaining

    def clear(self) -> None:
        """Drop any active block (used by tests and on explicit shutdown)."""
        with self._lock:
            self._blocked_until = 0.0
            self._reason = ""

    # -- waiting -------------------------------------------------------

    async def wait_until_clear(self, poll_interval: float = 5.0) -> float:
        """Sleep until the gate is clear. Returns the seconds actually waited.

        Safe to call from any event loop, including one running on a worker
        thread, because it holds no loop-bound state.
        """
        waited = 0.0
        next_log = 0.0

        while True:
            remaining = self.remaining()
            if remaining <= 0:
                break

            if waited >= next_log:
                logger.info(
                    "Waiting %s for usage limit to reset%s...",
                    _format_duration(remaining),
                    f" ({self.reason})" if self.reason else "",
                )
                next_log = waited + 60.0

            nap = min(remaining, poll_interval)
            await asyncio.sleep(nap)
            waited += nap

        if waited > 0:
            logger.info("Usage limit cleared after %s; resuming.", _format_duration(waited))
        return waited

    async def pause_for(self, reset_at: Optional[float], reason: str = "") -> float:
        """Record a limit and wait it out. Returns seconds waited."""
        self.note_limit(reset_at, reason)
        return await self.wait_until_clear()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


# Shared across every backend in the process: a quota is per-account, not
# per-client, so all pools must observe the same block.
GLOBAL_USAGE_LIMIT_GATE = UsageLimitGate()


def get_usage_limit_gate() -> UsageLimitGate:
    return GLOBAL_USAGE_LIMIT_GATE
