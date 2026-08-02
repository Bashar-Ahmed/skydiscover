"""Tests for usage-limit detection and the wait-until-reset gate."""

import asyncio
import time

import pytest

from skydiscover.llm.rate_limit import (
    DEFAULT_FALLBACK_WAIT_SECONDS,
    UsageLimitGate,
    extract_error_details,
    looks_like_usage_limit,
    parse_reset_from_headers,
    parse_reset_from_text,
    parse_usage_limit,
)

NOW = 1_800_000_000.0


class TestDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "Claude AI usage limit reached",
            "You've reached your usage limit",
            "rate_limit_error",
            "Rate limit exceeded",
            "429 Too Many Requests",
            "quota exceeded",
            "insufficient_quota",
            "You've hit your weekly limit for Opus",
            "5-hour limit reached",
            "Your limit will reset at 3pm",
            "Upgrade to Claude Max for more usage",
        ],
    )
    def test_positive(self, text):
        assert looks_like_usage_limit(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Model output limit reached: max_tokens",
            "SyntaxError: invalid syntax",
            "Connection reset by peer",
            "The evaluator timed out",
            "",
            None,
        ],
    )
    def test_negative(self, text):
        assert looks_like_usage_limit(text) is False

    def test_status_429_alone_is_enough(self):
        assert looks_like_usage_limit("anything", 429) is True

    def test_other_status_is_not(self):
        assert looks_like_usage_limit("anything", 500) is False


class TestTextParsing:
    def test_pipe_epoch(self):
        text = f"Claude AI usage limit reached|{int(NOW + 3600)}"
        assert parse_reset_from_text(text, now=NOW) == pytest.approx(NOW + 3600, abs=1)

    def test_pipe_epoch_millis(self):
        text = f"Claude AI usage limit reached|{int((NOW + 1800) * 1000)}"
        assert parse_reset_from_text(text, now=NOW) == pytest.approx(NOW + 1800, abs=1)

    @pytest.mark.parametrize(
        "text,seconds",
        [
            ("try again in 42 minutes", 42 * 60),
            ("try again in 1 minute", 60),
            ("retry after 30 seconds", 30),
            ("retry after 30s", 30),
            ("back in 2 hours", 7200),
            ("available again in 1 day", 86400),
        ],
    )
    def test_relative_durations(self, text, seconds):
        assert parse_reset_from_text(text, now=NOW) == pytest.approx(NOW + seconds, abs=1)

    def test_clock_time_resolves_to_future(self):
        got = parse_reset_from_text("Your limit will reset at 3pm", now=NOW)
        assert got is not None
        assert NOW < got <= NOW + 86400

    def test_unparseable_returns_none(self):
        assert parse_reset_from_text("no timing information here", now=NOW) is None

    def test_past_epoch_ignored(self):
        text = f"usage limit reached|{int(NOW - 5000)}"
        assert parse_reset_from_text(text, now=NOW) is None


class TestHeaderParsing:
    def test_retry_after_seconds(self):
        got = parse_reset_from_headers({"retry-after": "120"}, now=NOW)
        assert got == pytest.approx(NOW + 120)

    def test_go_duration(self):
        got = parse_reset_from_headers({"x-ratelimit-reset-requests": "6m0s"}, now=NOW)
        assert got == pytest.approx(NOW + 360)

    def test_iso8601(self):
        import datetime

        target = datetime.datetime.fromtimestamp(NOW + 900, datetime.timezone.utc)
        headers = {"anthropic-ratelimit-unified-reset": target.isoformat().replace("+00:00", "Z")}
        assert parse_reset_from_headers(headers, now=NOW) == pytest.approx(NOW + 900, abs=1)

    def test_earliest_of_several_wins(self):
        headers = {"retry-after": "600", "x-ratelimit-reset-tokens": "30s"}
        assert parse_reset_from_headers(headers, now=NOW) == pytest.approx(NOW + 30)

    def test_headers_in_the_past_ignored(self):
        assert parse_reset_from_headers({"retry-after": "0"}, now=NOW) is None

    def test_none_headers(self):
        assert parse_reset_from_headers(None, now=NOW) is None

    def test_case_insensitive(self):
        assert parse_reset_from_headers({"Retry-After": "45"}, now=NOW) == pytest.approx(NOW + 45)

    def test_epoch_seconds_header_is_absolute_not_relative(self):
        """X-RateLimit-Reset carries an epoch; adding it to now would park for decades."""
        headers = {"x-ratelimit-reset": str(int(NOW + 300))}
        assert parse_reset_from_headers(headers, now=NOW) == pytest.approx(NOW + 300)

    def test_epoch_millis_header_is_absolute(self):
        headers = {"x-ratelimit-reset": str(int((NOW + 300) * 1000))}
        assert parse_reset_from_headers(headers, now=NOW) == pytest.approx(NOW + 300, abs=1)

    def test_small_number_is_still_a_relative_offset(self):
        assert parse_reset_from_headers({"retry-after": "90"}, now=NOW) == pytest.approx(NOW + 90)


class TestParseUsageLimit:
    """Only a *hard* quota parks the process without a reset time.

    A transient 429 clears in seconds; parking every LLM call in the process
    for the fallback window would be far worse than one backoff retry.
    """

    def test_non_limit_returns_none(self):
        assert parse_usage_limit("segfault", None, None, now=NOW) is None

    def test_headers_take_priority_over_text(self):
        got = parse_usage_limit(
            "usage limit reached, try again in 9 hours", 429, {"retry-after": "60"}, now=NOW
        )
        assert got == pytest.approx(NOW + 60)

    def test_hard_quota_without_timing_uses_fallback(self):
        got = parse_usage_limit("Claude AI usage limit reached", None, None, now=NOW)
        assert got == pytest.approx(NOW + DEFAULT_FALLBACK_WAIT_SECONDS)

    def test_soft_rate_limit_without_timing_does_not_park(self):
        assert parse_usage_limit("rate_limit_error", 429, None, now=NOW) is None

    def test_bare_429_does_not_park(self):
        assert parse_usage_limit("", 429, None, now=NOW) is None
        assert parse_usage_limit("Too Many Requests", 429, None, now=NOW) is None

    def test_soft_rate_limit_with_timing_parks_briefly(self):
        got = parse_usage_limit("rate limit exceeded", 429, {"retry-after": "30"}, now=NOW)
        assert got == pytest.approx(NOW + 30)

    def test_overloaded_error_is_not_a_quota(self):
        """529 is server capacity, not an exhausted allowance."""
        text = "Error code: 529 - {'type': 'overloaded_error', 'message': 'Overloaded'}"
        assert parse_usage_limit(text, 529, None, now=NOW) is None

    @pytest.mark.parametrize(
        "text,seconds",
        [
            # The verbatim OpenAI RPM 429 body. Misreading "120ms" as an
            # hour-of-day previously stalled the process for ~21 hours.
            (
                "Rate limit reached for gpt-4o in organization org-abc on requests per "
                "min (RPM): Limit 500, Used 500, Requested 1. Please try again in 120ms.",
                0.12,
            ),
            ("Rate limit reached. Please try again in 20ms.", 0.02),
            ("Rate limit reached. Please try again in 6m0s.", 360),
        ],
    )
    def test_subsecond_and_compound_retry_hints(self, text, seconds):
        got = parse_usage_limit(text, 429, None, now=NOW)
        assert got is not None
        assert got - NOW == pytest.approx(seconds, abs=0.01)


class TestExtractErrorDetails:
    def test_pulls_status_and_headers(self):
        class FakeResponse:
            status_code = 429
            headers = {"retry-after": "12"}
            text = "rate limited"

        class FakeError(Exception):
            response = FakeResponse()
            body = {"error": {"type": "rate_limit_error"}}

        details = extract_error_details(FakeError("boom"))
        assert details["status_code"] == 429
        assert details["headers"] == {"retry-after": "12"}
        assert "rate_limit_error" in details["text"]

    def test_plain_exception(self):
        details = extract_error_details(ValueError("nope"))
        assert details["status_code"] is None
        assert "nope" in details["text"]


class TestGate:
    def test_starts_clear(self):
        assert UsageLimitGate().remaining() == 0.0

    def test_note_limit_blocks(self):
        gate = UsageLimitGate()
        remaining = gate.note_limit(time.time() + 30, "test")
        assert 25 < remaining <= 40
        assert gate.reason == "test"

    def test_later_reset_extends_block(self):
        gate = UsageLimitGate()
        gate.note_limit(time.time() + 30, "first")
        gate.note_limit(time.time() + 120, "second")
        assert gate.remaining() > 100
        assert gate.reason == "second"

    def test_earlier_reset_does_not_shorten_block(self):
        gate = UsageLimitGate()
        gate.note_limit(time.time() + 300, "strict")
        gate.note_limit(time.time() + 10, "lax")
        assert gate.remaining() > 250
        assert gate.reason == "strict"

    def test_max_wait_is_capped(self):
        gate = UsageLimitGate(max_wait_seconds=60)
        gate.note_limit(time.time() + 100_000, "weekly")
        assert gate.remaining() <= 70

    def test_none_reset_uses_fallback(self):
        gate = UsageLimitGate()
        remaining = gate.note_limit(None, "unknown")
        assert remaining == pytest.approx(DEFAULT_FALLBACK_WAIT_SECONDS, abs=30)

    def test_clear(self):
        gate = UsageLimitGate()
        gate.note_limit(time.time() + 300, "x")
        gate.clear()
        assert gate.remaining() == 0.0

    def test_wait_until_clear_returns_immediately_when_clear(self):
        gate = UsageLimitGate()
        waited = asyncio.run(gate.wait_until_clear())
        assert waited == 0.0

    def test_wait_until_clear_actually_waits(self):
        gate = UsageLimitGate()
        gate.note_limit(time.time() + 0.25 - 5.0, "brief")  # grace makes this ~0.25s

        async def run():
            start = time.monotonic()
            await gate.wait_until_clear(poll_interval=0.05)
            return time.monotonic() - start

        elapsed = asyncio.run(run())
        assert 0.1 <= elapsed < 3.0
        assert gate.remaining() == 0.0

    def test_concurrent_callers_all_wait_and_resume(self):
        gate = UsageLimitGate()
        gate.note_limit(time.time() + 0.2 - 5.0, "shared")

        async def run():
            results = await asyncio.gather(
                *(gate.wait_until_clear(poll_interval=0.05) for _ in range(5))
            )
            return results

        waits = asyncio.run(run())
        assert len(waits) == 5
        assert all(w > 0 for w in waits)
        assert gate.remaining() == 0.0

    def test_gate_is_usable_from_a_foreign_event_loop(self):
        """The gate must hold no loop-bound state (EvoX drives loops on threads)."""
        import threading

        gate = UsageLimitGate()
        gate.note_limit(time.time() + 0.15 - 5.0, "thread")
        result = {}

        def worker():
            result["waited"] = asyncio.run(gate.wait_until_clear(poll_interval=0.05))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert result["waited"] > 0

    def test_pause_for_records_and_waits(self):
        gate = UsageLimitGate()

        async def run():
            return await gate.pause_for(time.time() + 0.2 - 5.0, "combo")

        assert asyncio.run(run()) >= 0.0
        assert gate.remaining() == 0.0
