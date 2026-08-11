"""Tests for the Codex CLI LLM backend."""

import json
import shutil

import pytest

from skydiscover.config import LLMModelConfig
from skydiscover.llm.codex_cli import (
    CLI_EFFORT_LEVELS,
    CodexCLILLM,
    UsageTracker,
    _event_error_text,
    _flatten_messages,
    _json_schema_from_response_format,
    _merge_system_prompt,
    _parse_jsonl,
    normalize_effort,
)
from skydiscover.llm.rate_limit import UsageLimitError


@pytest.fixture
def backend(monkeypatch):
    """A CodexCLILLM whose binary lookup is stubbed out."""
    monkeypatch.setattr(shutil, "which", lambda _b: "/usr/bin/codex")
    return CodexCLILLM(LLMModelConfig(name="codex_cli/gpt-5.6-terra", timeout=60, retries=1))


def jsonl(*events):
    return "\n".join(json.dumps(e) for e in events).encode("utf-8")


def agent_message(text):
    return {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": text}}


class TestEffortNormalization:
    @pytest.mark.parametrize("level", CLI_EFFORT_LEVELS)
    def test_valid_levels_pass_through(self, level):
        assert normalize_effort(level) == level

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("none", "low"),
            ("MEDIUM", "medium"),
            ("extra_high", "xhigh"),
            ("extra-high", "xhigh"),
            ("maximum", "max"),
        ],
    )
    def test_aliases(self, given, expected):
        assert normalize_effort(given) == expected

    def test_minimal_is_mapped_down_not_passed_through(self):
        """Verified against a live CLI: `minimal` is rejected with
        `unsupported_value`, despite still appearing in the published docs.
        Mapping it down to `low` keeps the intent; dropping it would silently
        promote the call to the model's default effort instead."""
        assert normalize_effort("minimal") == "low"

    def test_max_is_a_real_level(self):
        """Regression: `max` was briefly mapped to `xhigh` on the strength of a
        stale doc page. models_cache.json lists it as supported, so remapping it
        would quietly cap emulated temperature below its top rung."""
        assert normalize_effort("max") == "max"

    def test_every_shared_ladder_rung_is_usable(self):
        """The temperature emulator samples from DEFAULT_EFFORT_LADDER, which is
        written for the Claude CLI. Every rung must survive the trip to Codex,
        or emulated temperature would silently drop calls at the ends of its
        range."""
        from skydiscover.llm.temperature import DEFAULT_EFFORT_LADDER

        for rung in DEFAULT_EFFORT_LADDER:
            assert normalize_effort(rung) in CLI_EFFORT_LEVELS, rung

    def test_unknown_is_dropped(self):
        assert normalize_effort("turbo") is None

    def test_empty(self):
        assert normalize_effort(None) is None
        assert normalize_effort("") is None


class TestConstruction:
    def test_missing_binary_raises_actionable_error(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _b: None)
        with pytest.raises(RuntimeError, match="not found on PATH"):
            CodexCLILLM(LLMModelConfig(name="codex_cli/gpt-5.6-terra"))

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("codex_cli/gpt-5.6-terra", "gpt-5.6-terra"),
            ("codex-cli/gpt-5.6-terra", "gpt-5.6-terra"),
            ("codex/gpt-5.6-terra", "gpt-5.6-terra"),
            ("gpt-5.6-terra", "gpt-5.6-terra"),
        ],
    )
    def test_provider_prefix_stripped(self, monkeypatch, name, expected):
        monkeypatch.setattr(shutil, "which", lambda _b: "/usr/bin/codex")
        assert CodexCLILLM(LLMModelConfig(name=name)).model == expected

    def test_temperature_is_never_sent(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _b: "/usr/bin/codex")
        backend = CodexCLILLM(LLMModelConfig(name="codex_cli/gpt-5.6-terra", temperature=0.9))
        assert backend.temperature is None
        assert backend.top_p is None

    def test_binary_override_from_env(self, monkeypatch):
        monkeypatch.setenv("SKYDISCOVER_CODEX_BINARY", "/opt/codex")
        monkeypatch.setattr(shutil, "which", lambda b: b)
        assert CodexCLILLM(LLMModelConfig(name="codex_cli/gpt-5.6-terra")).binary == "/opt/codex"


class TestCommandBuilding:
    def test_core_flags_present(self, backend):
        cmd = backend._build_command(None)
        assert cmd[1] == "exec"
        for flag in ("--json", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config"):
            assert flag in cmd
        assert cmd[cmd.index("--sandbox") + 1] == "read-only"

    def test_approval_policy_is_a_config_key_not_a_flag(self, backend):
        """Regression: `codex exec` has no --ask-for-approval flag — that is an
        interactive-mode option — and clap rejects the whole invocation with
        'unexpected argument', so every call would have failed."""
        cmd = backend._build_command(None)
        assert "--ask-for-approval" not in cmd
        assert "approval_policy=never" in cmd
        assert cmd[cmd.index("approval_policy=never") - 1] == "--config"

    def test_prompt_is_read_from_stdin(self, backend):
        """A prompt carrying a whole program is not a comfortable argv entry."""
        assert backend._build_command(None)[-1] == "-"

    def test_model_flag(self, backend):
        cmd = backend._build_command(None)
        assert cmd[cmd.index("--model") + 1] == "gpt-5.6-terra"

    def test_effort_uses_a_config_override(self, backend):
        cmd = backend._build_command(None, reasoning_effort="high")
        assert "model_reasoning_effort=high" in cmd
        assert cmd[cmd.index("model_reasoning_effort=high") - 1] == "--config"

    def test_invalid_effort_is_omitted(self, backend):
        cmd = backend._build_command(None, reasoning_effort="turbo")
        assert not any(a.startswith("model_reasoning_effort=") for a in cmd)

    def test_no_effort_override_by_default(self, backend):
        cmd = backend._build_command(None)
        assert not any(a.startswith("model_reasoning_effort=") for a in cmd)

    def test_output_schema_flag(self, backend):
        cmd = backend._build_command("/tmp/schema.json")
        assert cmd[cmd.index("--output-schema") + 1] == "/tmp/schema.json"

    def test_extra_args_are_appended_before_the_stdin_sentinel(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _b: "/usr/bin/codex")
        backend = CodexCLILLM(
            LLMModelConfig(name="codex_cli/gpt-5.6-terra", cli_extra_args=["--color", "never"])
        )
        cmd = backend._build_command(None)
        assert cmd[-3:] == ["--color", "never", "-"]

    def test_sandbox_is_read_only_even_in_agentic_mode(self, backend):
        """The framework applies the model's answer itself; the agent explores."""
        cmd = backend._build_command(None, agentic=True)
        assert cmd[cmd.index("--sandbox") + 1] == "read-only"

    def test_backend_advertises_native_agentic(self, backend):
        assert backend.supports_native_agentic is True


class TestWorkingDirectory:
    def test_agentic_runs_from_the_codebase_root(self, backend, tmp_path):
        assert backend._resolve_cwd("/tmp/work", agentic=True, codebase_root=str(tmp_path)) == str(
            tmp_path
        )

    def test_agentic_falls_back_to_workdir_without_a_valid_root(self, backend):
        assert backend._resolve_cwd("/tmp/work", agentic=True, codebase_root="/no/such") == (
            "/tmp/work"
        )

    def test_non_agentic_stays_in_the_isolated_workdir(self, backend, tmp_path):
        """An empty cwd is what stops a non-agentic call reading project files —
        Codex has no way to disable its tools."""
        assert backend._resolve_cwd("/tmp/work", codebase_root=str(tmp_path)) == "/tmp/work"


class TestSystemPrompt:
    def test_system_message_is_folded_into_the_prompt(self):
        """codex exec has no system-prompt channel."""
        merged = _merge_system_prompt("Be terse.", "Say hi")
        assert "Be terse." in merged and "Say hi" in merged
        assert merged.index("Be terse.") < merged.index("Say hi")

    def test_empty_system_message_leaves_the_prompt_untouched(self):
        assert _merge_system_prompt("", "Say hi") == "Say hi"
        assert _merge_system_prompt("   ", "Say hi") == "Say hi"


class TestStreamHandling:
    def test_final_agent_message_is_returned(self, backend):
        out = jsonl(
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "item.completed", "item": {"type": "reasoning", "text": "hmm"}},
            agent_message("first"),
            agent_message("final answer"),
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 2}},
        )
        assert backend._text_from_stream(out, b"", 0) == "final answer"

    def test_non_agent_items_are_ignored(self, backend):
        out = jsonl(
            {"type": "item.completed", "item": {"type": "reasoning", "text": "thinking"}},
            {"type": "item.completed", "item": {"type": "command_execution", "text": "ls"}},
            agent_message("answer"),
        )
        assert backend._text_from_stream(out, b"", 0) == "answer"

    def test_turn_failed_raises(self, backend):
        out = jsonl(
            agent_message("partial"),
            {"type": "turn.failed", "error": {"message": "model exploded"}},
        )
        with pytest.raises(RuntimeError, match="model exploded"):
            backend._text_from_stream(out, b"", 0)

    def test_error_event_raises(self, backend):
        out = jsonl({"type": "error", "message": "bad request"})
        with pytest.raises(RuntimeError, match="bad request"):
            backend._text_from_stream(out, b"", 1)

    def test_no_agent_message_raises(self, backend):
        out = jsonl({"type": "turn.completed", "usage": {"input_tokens": 1}})
        with pytest.raises(RuntimeError, match="no agent message"):
            backend._text_from_stream(out, b"", 0)

    def test_empty_output_raises_with_stderr(self, backend):
        with pytest.raises(RuntimeError, match="no parseable events"):
            backend._text_from_stream(b"", b"codex: something broke", 1)

    def test_usage_is_recorded(self, backend, monkeypatch):
        tracker = UsageTracker()
        monkeypatch.setattr("skydiscover.llm.codex_cli.GLOBAL_USAGE_TRACKER", tracker)
        out = jsonl(
            agent_message("hi"),
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 80,
                    "output_tokens": 20,
                    "reasoning_output_tokens": 5,
                },
            },
        )
        backend._text_from_stream(out, b"", 0)
        assert tracker.summary() == {
            "calls": 1,
            "input_tokens": 100,
            "cached_input_tokens": 80,
            "output_tokens": 20,
            "reasoning_output_tokens": 5,
        }


class TestUsageLimits:
    def test_limit_in_a_failure_event_raises_usage_limit_error(self, backend):
        out = jsonl(
            {
                "type": "turn.failed",
                "error": {"message": "You've hit your usage limit. Try again at 3:30pm."},
            }
        )
        with pytest.raises(UsageLimitError):
            backend._text_from_stream(out, b"", 1)

    def test_limit_on_stderr_with_no_events_raises_usage_limit_error(self, backend):
        err = b"Error: usage limit reached. Your limit will reset at 11:45pm."
        with pytest.raises(UsageLimitError):
            backend._text_from_stream(b"", err, 1)

    def test_limit_disguised_as_an_answer_raises(self, backend):
        out = jsonl(agent_message("You have hit your usage limit; resets at 9:00am."))
        with pytest.raises(UsageLimitError):
            backend._text_from_stream(out, b"", 0)

    def test_ordinary_failure_is_not_a_usage_limit(self, backend):
        out = jsonl({"type": "turn.failed", "error": {"message": "connection refused"}})
        with pytest.raises(RuntimeError) as exc:
            backend._text_from_stream(out, b"", 1)
        assert not isinstance(exc.value, UsageLimitError)


class TestJsonlParsing:
    def test_skips_non_json_noise(self):
        text = 'warming up\n{"type":"a"}\nnot json\n{"type":"b"}\n'
        assert [e["type"] for e in _parse_jsonl(text)] == ["a", "b"]

    def test_skips_json_arrays_and_scalars(self):
        assert _parse_jsonl('[1,2]\n"str"\n{"type":"a"}') == [{"type": "a"}]

    def test_empty(self):
        assert _parse_jsonl("") == []


class TestErrorTextExtraction:
    @pytest.mark.parametrize(
        "event,expected",
        [
            ({"error": "flat"}, "flat"),
            ({"error": {"message": "nested"}}, "nested"),
            ({"message": "top level"}, "top level"),
            ({"type": "error"}, None),
        ],
    )
    def test_shapes(self, event, expected):
        assert _event_error_text(event) == expected


class TestHelpers:
    def test_single_user_message_passes_through(self):
        assert _flatten_messages([{"role": "user", "content": "hi"}]) == "hi"

    def test_multi_turn_is_labelled(self):
        out = _flatten_messages(
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        )
        assert "[user]" in out and "[assistant]" in out

    def test_json_schema_extracted(self):
        fmt = {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
        assert _json_schema_from_response_format(fmt) == {"type": "object"}

    def test_json_object_format_yields_no_schema(self):
        assert _json_schema_from_response_format({"type": "json_object"}) is None


class TestRejectedModes:
    @pytest.mark.asyncio
    async def test_image_output_is_rejected(self, backend):
        with pytest.raises(NotImplementedError, match="cannot generate images"):
            await backend.generate("s", [{"role": "user", "content": "x"}], image_output=True)

    @pytest.mark.asyncio
    async def test_empty_prompt_is_rejected(self, backend):
        with pytest.raises(ValueError, match="empty prompt"):
            await backend.generate("s", [{"role": "user", "content": "  "}])
