"""Usage-limit handling and native delegation on the agentic generation path."""

import asyncio
import random
import time

import pytest

from skydiscover.config import AgenticConfig
from skydiscover.llm.agentic_generator import AgenticGenerator
from skydiscover.llm.base import LLMResponse
from skydiscover.llm.rate_limit import get_usage_limit_gate


class FakePool:
    """Minimal stand-in for LLMPool."""

    def __init__(self, models):
        self.models = models
        self.weights = [1.0 / len(models)] * len(models)
        self.effective_weights = list(self.weights)
        self.random_state = random.Random(0)

    def _sample_model(self):
        return self.models[0]


class NativeAgenticModel:
    supports_native_agentic = True

    def __init__(self, text="native answer"):
        self.text = text
        self.calls = []

    async def generate(self, system_message, messages, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(text=self.text)


class ExplodingNativeModel:
    supports_native_agentic = True

    async def generate(self, system_message, messages, **kwargs):
        raise RuntimeError("cli blew up")


@pytest.fixture(autouse=True)
def clear_gate():
    get_usage_limit_gate().clear()
    yield
    get_usage_limit_gate().clear()


class TestNativeDelegation:
    def test_delegates_to_native_agentic_backend(self):
        model = NativeAgenticModel()
        gen = AgenticGenerator(
            FakePool([model]), AgenticConfig(enabled=True, max_steps=5, codebase_root=None)
        )
        out = asyncio.run(gen.generate("sys", "do the thing"))
        assert out == "native answer"
        assert model.calls[0]["agentic"] is True
        assert model.calls[0]["max_steps"] == 5

    def test_native_failure_returns_none_for_caller_fallback(self):
        gen = AgenticGenerator(FakePool([ExplodingNativeModel()]), AgenticConfig(enabled=True))
        assert asyncio.run(gen.generate("sys", "x")) is None

    def test_blank_native_output_returns_none(self):
        gen = AgenticGenerator(FakePool([NativeAgenticModel(text="   ")]), AgenticConfig())
        assert asyncio.run(gen.generate("sys", "x")) is None


class TestQuotaPauseOnAgenticPath:
    """A quota pause must not be cancelled by the per-step timeout, nor count
    against the agent's wall-clock budget, nor consume the step."""

    def _generator(self, per_step_timeout=0.5, overall_timeout=30.0):
        pool = FakePool([object()])  # never actually called
        return AgenticGenerator(
            pool,
            AgenticConfig(
                enabled=True,
                max_steps=3,
                per_step_timeout=per_step_timeout,
                overall_timeout=overall_timeout,
            ),
        )

    def test_pause_survives_a_shorter_per_step_timeout(self, monkeypatch):
        """The gate wait sits outside asyncio.wait_for; a pause longer than
        per_step_timeout must still complete rather than being cancelled."""
        gen = self._generator(per_step_timeout=0.2)
        calls = {"n": 0}

        class Limited(Exception):
            status_code = 429

        async def flaky(system_message, conversation):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Limited("rate_limit_error, try again in 1 seconds")
            return {"role": "assistant", "content": "done"}

        monkeypatch.setattr(gen, "_call_llm", flaky)

        async def run():
            start = time.monotonic()
            msg, paused = await gen._call_llm_with_limits("sys", [], 0.2)
            return msg, paused, time.monotonic() - start

        msg, paused, elapsed = asyncio.run(run())
        assert msg["content"] == "done"
        assert calls["n"] == 2
        # Waited out the ~1s reset despite a 0.2s per-step timeout.
        assert paused >= 0.5
        assert elapsed >= 0.5

    def test_real_timeout_still_propagates(self, monkeypatch):
        gen = self._generator()

        async def hang(system_message, conversation):
            await asyncio.sleep(5)

        monkeypatch.setattr(gen, "_call_llm", hang)
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(gen._call_llm_with_limits("sys", [], 0.2))

    def test_non_quota_error_propagates_immediately(self, monkeypatch):
        gen = self._generator()

        async def boom(system_message, conversation):
            raise ValueError("syntax error in tool schema")

        monkeypatch.setattr(gen, "_call_llm", boom)
        with pytest.raises(ValueError):
            asyncio.run(gen._call_llm_with_limits("sys", [], 5.0))

    def test_gives_up_after_repeated_limits(self, monkeypatch):
        gen = self._generator()

        class Limited(Exception):
            status_code = 429

        async def always(system_message, conversation):
            raise Limited("rate_limit_error, try again in 1 seconds")

        monkeypatch.setattr(gen, "_call_llm", always)
        monkeypatch.setattr("skydiscover.llm.agentic_generator._MAX_USAGE_LIMIT_WAITS", 2)
        with pytest.raises(Limited):
            asyncio.run(gen._call_llm_with_limits("sys", [], 5.0))


class TestSampling:
    def test_uses_pool_sampler_so_emulated_weights_apply(self):
        sentinel = NativeAgenticModel()

        class Pool(FakePool):
            def __init__(self):
                super().__init__([object(), sentinel])
                self.sampled = 0

            def _sample_model(self):
                self.sampled += 1
                return sentinel

        pool = Pool()
        gen = AgenticGenerator(pool, AgenticConfig())
        assert gen._sample_model() is sentinel
        assert pool.sampled == 1

    def test_falls_back_when_pool_has_no_sampler(self):
        target = object()

        class BarePool:
            models = [target]
            weights = [1.0]
            random_state = random.Random(0)

        gen = AgenticGenerator(BarePool(), AgenticConfig())
        assert gen._sample_model() is target
