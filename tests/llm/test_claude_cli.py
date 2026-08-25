"""Tests for the Claude Code CLI LLM backend."""

import asyncio
import json
import os
import shutil

import pytest

from skydiscover.config import LLMModelConfig
from skydiscover.llm.claude_cli import (
    CLI_EFFORT_LEVELS,
    ClaudeCLILLM,
    CostTracker,
    _flatten_messages,
    _json_schema_from_response_format,
    _last_json_object,
    normalize_effort,
)
from skydiscover.llm.rate_limit import UsageLimitError, get_usage_limit_gate

CLI_AVAILABLE = shutil.which("claude") is not None


@pytest.fixture
def backend(monkeypatch):
    """A ClaudeCLILLM whose binary lookup is stubbed out."""
    monkeypatch.setattr(shutil, "which", lambda _b: "/usr/bin/claude")
    return ClaudeCLILLM(LLMModelConfig(name="claude_cli/sonnet", timeout=60, retries=1))


class TestEffortNormalization:
    @pytest.mark.parametrize("level", CLI_EFFORT_LEVELS)
    def test_valid_levels_pass_through(self, level):
        assert normalize_effort(level) == level

    @pytest.mark.parametrize(
        "given,expected",
        [("minimal", "low"), ("none", "low"), ("MEDIUM", "medium"), ("maximum", "max")],
    )
    def test_aliases(self, given, expected):
        assert normalize_effort(given) == expected

    def test_unknown_is_dropped(self):
        assert normalize_effort("turbo") is None

    def test_empty(self):
        assert normalize_effort(None) is None
        assert normalize_effort("") is None


class TestConstruction:
    def test_missing_binary_raises_actionable_error(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _b: None)
        with pytest.raises(RuntimeError, match="not found on PATH"):
            ClaudeCLILLM(LLMModelConfig(name="claude_cli/sonnet"))

    def test_provider_prefix_stripped(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _b: "/usr/bin/claude")
        assert ClaudeCLILLM(LLMModelConfig(name="claude_cli/opus")).model == "opus"
        assert ClaudeCLILLM(LLMModelConfig(name="claude-cli/opus")).model == "opus"
        assert ClaudeCLILLM(LLMModelConfig(name="haiku")).model == "haiku"

    def test_temperature_is_never_sent(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _b: "/usr/bin/claude")
        llm = ClaudeCLILLM(LLMModelConfig(name="claude_cli/sonnet", temperature=0.9))
        assert llm.temperature is None
        assert llm.top_p is None


class TestCommandBuilding:
    def test_core_flags_present(self, backend):
        cmd = backend._build_command(None)
        assert "--print" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "json"
        # Pure text generation: no tools, single turn.
        assert cmd[cmd.index("--tools") + 1] == ""
        assert cmd[cmd.index("--max-turns") + 1] == "1"
        # Hermetic: no ambient settings, MCP servers, or slash commands.
        assert "--no-session-persistence" in cmd
        assert "--strict-mcp-config" in cmd
        assert "--disable-slash-commands" in cmd
        assert cmd[cmd.index("--setting-sources") + 1] == ""

    def test_model_flag(self, backend):
        cmd = backend._build_command(None)
        assert cmd[cmd.index("--model") + 1] == "sonnet"

    def test_effort_flag_from_kwargs(self, backend):
        cmd = backend._build_command(None, reasoning_effort="xhigh")
        assert cmd[cmd.index("--effort") + 1] == "xhigh"

    def test_invalid_effort_is_omitted(self, backend):
        assert "--effort" not in backend._build_command(None, reasoning_effort="bogus")

    def test_no_effort_flag_by_default(self, backend):
        assert "--effort" not in backend._build_command(None)

    def test_system_prompt_file(self, backend):
        cmd = backend._build_command("/tmp/sys.txt")
        assert cmd[cmd.index("--system-prompt-file") + 1] == "/tmp/sys.txt"

    def test_json_schema_flag(self, backend):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        rf = {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}
        cmd = backend._build_command(None, response_format=rf)
        assert json.loads(cmd[cmd.index("--json-schema") + 1]) == schema

    def test_json_object_format_adds_no_schema_flag(self, backend):
        cmd = backend._build_command(None, response_format={"type": "json_object"})
        assert "--json-schema" not in cmd

    def test_budget_flag(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _b: "/usr/bin/claude")
        cfg = LLMModelConfig(name="claude_cli/sonnet")
        cfg.max_budget_usd = 5
        cmd = ClaudeCLILLM(cfg)._build_command(None)
        assert cmd[cmd.index("--max-budget-usd") + 1] == "5"


class TestAgenticCommandBuilding:
    def test_agentic_enables_read_only_tools(self, backend):
        cmd = backend._build_command(None, agentic=True, max_steps=5)
        tools = cmd[cmd.index("--tools") + 1]
        assert tools == "Read,Grep,Glob"
        # Pre-approved so a non-interactive run never blocks on a prompt.
        assert cmd[cmd.index("--allowed-tools") + 1] == tools

    def test_agentic_never_grants_write_or_bash(self, backend):
        tools = backend._build_command(None, agentic=True)[
            backend._build_command(None, agentic=True).index("--tools") + 1
        ]
        for dangerous in ("Edit", "Write", "Bash"):
            assert dangerous not in tools

    def test_agentic_turn_budget_leaves_room_for_tool_calls(self, backend):
        """max_turns == max_steps would be spent entirely on exploration and
        end in error_max_turns with no answer."""
        cmd = backend._build_command(None, agentic=True, max_steps=5)
        assert int(cmd[cmd.index("--max-turns") + 1]) > 5

    def test_agentic_turn_budget_has_a_floor(self, backend):
        cmd = backend._build_command(None, agentic=True, max_steps=0)
        assert int(cmd[cmd.index("--max-turns") + 1]) >= 4

    def test_add_dir_only_for_existing_directory(self, backend, tmp_path):
        cmd = backend._build_command(None, agentic=True, codebase_root=str(tmp_path))
        assert cmd[cmd.index("--add-dir") + 1] == str(tmp_path)
        assert "--add-dir" not in backend._build_command(
            None, agentic=True, codebase_root="/nonexistent/path/xyz"
        )
        assert "--add-dir" not in backend._build_command(None, agentic=True, codebase_root=None)

    def test_non_agentic_is_still_toolless_single_turn(self, backend):
        cmd = backend._build_command(None)
        assert cmd[cmd.index("--tools") + 1] == ""
        assert cmd[cmd.index("--max-turns") + 1] == "1"
        assert "--allowed-tools" not in cmd
        assert "--add-dir" not in cmd

    def test_backend_advertises_native_agentic(self, backend):
        assert backend.supports_native_agentic is True

    def test_agentic_runs_from_the_codebase_root(self, backend, tmp_path):
        """Regression: --add-dir widens the permission boundary but does not
        point the agent anywhere. Launched from an empty temp cwd, the agent's
        Glob/Grep searched that empty directory and found nothing."""
        cwd = backend._resolve_cwd("/tmp/workdir", agentic=True, codebase_root=str(tmp_path))
        assert cwd == str(tmp_path)

    def test_agentic_falls_back_to_workdir_without_a_valid_root(self, backend):
        assert backend._resolve_cwd("/tmp/workdir", agentic=True) == "/tmp/workdir"
        assert (
            backend._resolve_cwd("/tmp/workdir", agentic=True, codebase_root="/no/such/dir")
            == "/tmp/workdir"
        )

    def test_non_agentic_stays_in_the_isolated_workdir(self, backend, tmp_path):
        """A pure generation call must not pick up ambient CLAUDE.md files."""
        assert backend._resolve_cwd("/tmp/workdir", codebase_root=str(tmp_path)) == "/tmp/workdir"

    def test_exhausted_turns_gives_actionable_error(self, backend):
        payload = {"is_error": False, "result": None, "subtype": "error_max_turns"}
        with pytest.raises(RuntimeError, match="max_steps"):
            backend._text_from_payload(payload)


class TestPayloadHandling:
    def test_success(self, backend):
        payload = {"is_error": False, "result": "hello", "api_error_status": None}
        assert backend._text_from_payload(payload) == "hello"

    def test_api_error_raises(self, backend):
        payload = {
            "is_error": True,
            "result": "There's an issue with the selected model",
            "api_error_status": 404,
            "subtype": "success",
        }
        with pytest.raises(RuntimeError, match="status=404"):
            backend._text_from_payload(payload)

    def test_usage_limit_raises_usage_limit_error(self, backend):
        payload = {
            "is_error": True,
            "result": "Claude AI usage limit reached|1900000000",
            "api_error_status": 429,
        }
        with pytest.raises(UsageLimitError) as exc:
            backend._text_from_payload(payload)
        assert exc.value.reset_at == pytest.approx(1900000000, abs=1)

    def test_usage_limit_in_a_non_error_result(self, backend):
        payload = {"is_error": False, "result": "You've hit your weekly limit for Opus."}
        with pytest.raises(UsageLimitError):
            backend._text_from_payload(payload)

    def test_long_result_mentioning_limits_is_not_treated_as_one(self, backend):
        text = "Here is code that handles a rate limit. " + "x" * 500
        payload = {"is_error": False, "result": text}
        assert backend._text_from_payload(payload) == text

    def test_missing_result_raises(self, backend):
        with pytest.raises(RuntimeError, match="no text result"):
            backend._text_from_payload({"is_error": False, "result": None})

    def test_parse_empty_stdout_raises(self, backend):
        with pytest.raises(RuntimeError, match="no output"):
            backend._parse_payload(b"", b"segfault", 1)

    def test_parse_empty_stdout_with_limit_in_stderr(self, backend):
        with pytest.raises(UsageLimitError):
            backend._parse_payload(b"", b"Claude AI usage limit reached", 1)

    def test_parse_non_json_raises(self, backend):
        with pytest.raises(RuntimeError, match="non-JSON"):
            backend._parse_payload(b"not json at all", b"", 0)

    def test_parse_recovers_trailing_json_object(self, backend):
        raw = b'warning: something\n{"is_error": false, "result": "ok"}'
        assert backend._parse_payload(raw, b"", 0)["result"] == "ok"


class TestHelpers:
    def test_flatten_single_user_message_is_verbatim(self):
        assert _flatten_messages([{"role": "user", "content": "hi"}]) == "hi"

    def test_flatten_multiturn_is_labelled(self):
        out = _flatten_messages(
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ]
        )
        assert "[user]" in out and "[assistant]" in out

    def test_flatten_multipart_content(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
        assert _flatten_messages(msgs) == "x"

    def test_flatten_empty(self):
        assert _flatten_messages([]) == ""

    def test_schema_extraction_variants(self):
        assert _json_schema_from_response_format(None) is None
        assert _json_schema_from_response_format({"type": "json_object"}) is None
        schema = {"type": "object"}
        assert (
            _json_schema_from_response_format(
                {"type": "json_schema", "json_schema": {"schema": schema}}
            )
            == schema
        )

    def test_last_json_object(self):
        assert _last_json_object('noise {"a": 1} more {"b": 2}') == {"b": 2}
        assert _last_json_object("no objects here") is None


class TestCostTracker:
    def test_accumulates(self):
        tracker = CostTracker()
        tracker.record(
            {"total_cost_usd": 0.25, "usage": {"input_tokens": 100, "output_tokens": 20}}
        )
        tracker.record({"total_cost_usd": 0.5, "usage": {"input_tokens": 10, "output_tokens": 5}})
        summary = tracker.summary()
        assert summary["calls"] == 2
        assert summary["total_cost_usd"] == pytest.approx(0.75)
        assert summary["input_tokens"] == 110
        assert summary["output_tokens"] == 25

    def test_tolerates_missing_fields(self):
        tracker = CostTracker()
        tracker.record({})
        assert tracker.summary()["calls"] == 1


class TestGenerateContract:
    def test_image_output_is_rejected(self, backend):
        with pytest.raises(NotImplementedError, match="cannot generate images"):
            asyncio.run(
                backend.generate("sys", [{"role": "user", "content": "x"}], image_output=True)
            )

    def test_empty_prompt_is_rejected(self, backend):
        with pytest.raises(ValueError, match="empty prompt"):
            asyncio.run(backend.generate("sys", [{"role": "user", "content": "  "}]))

    def test_usage_limit_pauses_then_succeeds(self, backend, monkeypatch):
        """A usage limit must wait and retry, not consume the retry budget."""
        get_usage_limit_gate().clear()
        calls = {"n": 0}

        async def fake_invoke(system_message, prompt, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                # Reset already in the past: the gate waits ~0s.
                raise UsageLimitError("usage limit reached", reset_at=0.0)
            return {"is_error": False, "result": "recovered"}

        monkeypatch.setattr(backend, "_invoke_cli", fake_invoke)
        # retries=1; without special handling the limit would eat that budget.
        result = asyncio.run(backend.generate("sys", [{"role": "user", "content": "go"}]))
        assert result.text == "recovered"
        assert calls["n"] == 2
        get_usage_limit_gate().clear()

    def test_gives_up_after_too_many_limit_waits(self, backend, monkeypatch):
        get_usage_limit_gate().clear()
        backend.max_usage_limit_waits = 2

        async def always_limited(system_message, prompt, **kwargs):
            raise UsageLimitError("usage limit reached", reset_at=0.0)

        monkeypatch.setattr(backend, "_invoke_cli", always_limited)
        with pytest.raises(RuntimeError, match="still usage-limited"):
            asyncio.run(backend.generate("sys", [{"role": "user", "content": "go"}]))
        get_usage_limit_gate().clear()

    def test_timeout_retry_downgrades_effort(self, backend, monkeypatch):
        """Each timeout retry steps the effort down a rung instead of
        re-running the identical call that just thought past the deadline."""
        efforts = []

        async def fake_invoke(system_message, prompt, **kwargs):
            efforts.append(kwargs.get("reasoning_effort"))
            if len(efforts) < 3:
                raise asyncio.TimeoutError()
            return {"is_error": False, "result": "done"}

        monkeypatch.setattr(backend, "_invoke_cli", fake_invoke)
        result = asyncio.run(
            backend.generate(
                "sys",
                [{"role": "user", "content": "go"}],
                reasoning_effort="max",
                retries=2,
                retry_delay=0,
            )
        )
        assert result.text == "done"
        assert efforts == ["max", "xhigh", "high"]

    def test_timeout_at_lowest_effort_keeps_retrying_unchanged(self, backend, monkeypatch):
        efforts = []

        async def fake_invoke(system_message, prompt, **kwargs):
            efforts.append(kwargs.get("reasoning_effort"))
            if len(efforts) < 2:
                raise asyncio.TimeoutError()
            return {"is_error": False, "result": "done"}

        monkeypatch.setattr(backend, "_invoke_cli", fake_invoke)
        result = asyncio.run(
            backend.generate(
                "sys",
                [{"role": "user", "content": "go"}],
                reasoning_effort="low",
                retries=1,
                retry_delay=0,
            )
        )
        assert result.text == "done"
        assert efforts == ["low", "low"]


@pytest.mark.integration
@pytest.mark.skipif(not CLI_AVAILABLE, reason="claude CLI not installed")
class TestLiveCLI:
    """Exercises the real binary. Consumes a small amount of plan quota."""

    def test_round_trip(self):
        llm = ClaudeCLILLM(LLMModelConfig(name="claude_cli/haiku", timeout=120, retries=0))
        response = asyncio.run(
            llm.generate(
                "You are terse. Reply with exactly one word.",
                [{"role": "user", "content": "Reply with the word OK"}],
            )
        )
        assert "OK" in response.text.upper()
