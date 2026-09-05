"""The per-call log of which model/effort each iteration actually used."""

import json

import pytest

from skydiscover.config import LLMModelConfig
from skydiscover.llm.base import LLMResponse
from skydiscover.llm.call_log import GLOBAL_CALL_LOG, LLMCallLog, set_call_context
from skydiscover.llm.llm_pool import LLMPool


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    log = LLMCallLog()
    log.configure(str(tmp_path))
    monkeypatch.setattr("skydiscover.llm.llm_pool.GLOBAL_CALL_LOG", log)
    set_call_context()
    return log


def rows(log):
    return [json.loads(line) for line in open(log.path)]


class FakeBackend:
    model = "fake-model-1"
    reasoning_effort = None

    def __init__(self):
        self.seen = []

    async def generate(self, system_message, messages, **kwargs):
        self.seen.append(kwargs)
        return LLMResponse(text="hello")


def pool_with(backend, **kw):
    pool = object.__new__(LLMPool)
    pool.models = [backend]
    pool.effective_weights = [1.0]
    pool.temperature_emulator = kw.get("emulator")
    pool.temperature = kw.get("temperature")
    import random

    pool.random_state = random.Random(0)
    return pool


class TestDisabledByDefault:
    def test_recording_without_configure_is_a_no_op(self):
        LLMCallLog().record("generate", model="x")  # must not raise

    def test_configure_with_no_dir_disables(self, tmp_path):
        log = LLMCallLog()
        log.configure(str(tmp_path))
        log.configure(None)
        assert log.path is None


class TestGenerateRecords:
    @pytest.mark.asyncio
    async def test_model_and_effort_are_recorded(self, isolated_log):
        backend = FakeBackend()
        await pool_with(backend).generate(
            "s", [{"role": "user", "content": "u"}], reasoning_effort="xhigh"
        )
        (row,) = rows(isolated_log)
        assert row["event"] == "generate"
        assert row["model"] == "fake-model-1"
        assert row["backend"] == "FakeBackend"
        assert row["reasoning_effort"] == "xhigh"
        assert row["response_chars"] == 5
        assert "duration_s" in row

    @pytest.mark.asyncio
    async def test_call_context_is_attached(self, isolated_log):
        set_call_context(iteration=7, phase="paradigm")
        await pool_with(FakeBackend()).generate("s", [{"role": "user", "content": "u"}])
        (row,) = rows(isolated_log)
        assert row["iteration"] == 7
        assert row["phase"] == "paradigm"

    @pytest.mark.asyncio
    async def test_backend_effort_is_used_when_the_call_sets_none(self, isolated_log):
        """Otherwise a run configured with a fixed effort would log nothing."""
        backend = FakeBackend()
        backend.reasoning_effort = "medium"
        await pool_with(backend).generate("s", [{"role": "user", "content": "u"}])
        assert rows(isolated_log)[0]["reasoning_effort"] == "medium"

    @pytest.mark.asyncio
    async def test_failure_is_recorded_and_the_error_still_propagates(self, isolated_log):
        class Boom(FakeBackend):
            async def generate(self, *a, **k):
                raise RuntimeError("backend exploded")

        with pytest.raises(RuntimeError, match="backend exploded"):
            await pool_with(Boom()).generate("s", [{"role": "user", "content": "u"}])
        (row,) = rows(isolated_log)
        assert row["event"] == "generate_failed"
        assert row["error_type"] == "RuntimeError"
        assert "backend exploded" in row["error"]


class TestRobustness:
    @pytest.mark.asyncio
    async def test_an_unwritable_log_does_not_break_generation(self, isolated_log, monkeypatch):
        """Logging is observability; it must never fail a run."""

        def boom(*a, **k):
            raise OSError("disk gone")

        monkeypatch.setattr("builtins.open", boom)
        result = await pool_with(FakeBackend()).generate("s", [{"role": "user", "content": "u"}])
        assert result.text == "hello"

    def test_unserialisable_values_do_not_raise(self, isolated_log):
        isolated_log.record("generate", model=object())
        assert len(rows(isolated_log)) == 1
