"""Tests for temperature emulation via model selection and reasoning effort."""

import random

import pytest

from skydiscover.config import LLMModelConfig
from skydiscover.llm.temperature import (
    DEFAULT_EFFORT_LADDER,
    TemperatureEmulationConfig,
    TemperatureEmulator,
    model_supports_temperature,
)


class TestModelSupportsTemperature:
    """The cutoff is Opus 4.7, not "newer than 4.5".

    Sampling parameters were removed starting with Opus 4.7 and are rejected
    with a 400 on Opus 4.7/4.8, Opus 5, Sonnet 5, Fable 5, and Mythos 5. They
    are still accepted on Opus 4.6, Sonnet 4.6, and the whole 4.5 family.
    """

    @pytest.mark.parametrize(
        "name",
        [
            # Non-Claude providers are unaffected.
            "gpt-5",
            "gpt-4o",
            "gemini-3-pro-preview",
            "llama3",
            # Claude 3.x and 4.0.
            "claude-3-5-sonnet-20241022",
            "claude-sonnet-4-20250514",
            # The 4.5 family still accepts temperature.
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            # 4.6 is the last version that accepts it.
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            # The bare "haiku" alias resolves to claude-haiku-4-5.
            "haiku",
        ],
    )
    def test_supported(self, name):
        assert model_supports_temperature(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            # Removal starts at Opus 4.7.
            "claude-opus-4-7",
            "claude-opus-4-8",
            # Claude 5 generation.
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-mythos-5",
            "claude-mythos-preview",
            # CLI aliases resolving to latest models.
            "opus",
            "sonnet",
            "fable",
        ],
    )
    def test_unsupported(self, name):
        assert model_supports_temperature(name) is False

    def test_claude_cli_provider_never_supports_temperature(self):
        # The CLI exposes no sampling knob at all, whatever the model.
        assert model_supports_temperature("claude-3-5-sonnet", "claude_cli") is False

    def test_provider_prefix_is_stripped(self):
        assert model_supports_temperature("anthropic/claude-opus-4-8") is False
        assert model_supports_temperature("anthropic/claude-opus-4-6") is True


class TestModelWeightShaping:
    def _emulator(self, temperature, **kwargs):
        return TemperatureEmulator(
            config=TemperatureEmulationConfig(**kwargs),
            temperature=temperature,
            rng=random.Random(0),
        )

    def test_temperature_one_reproduces_configured_weights(self):
        emulator = self._emulator(1.0)
        got = emulator.model_weights([0.7, 0.3])
        assert got == pytest.approx([0.7, 0.3])

    def test_zero_temperature_is_deterministic_argmax(self):
        emulator = self._emulator(0.0)
        assert emulator.model_weights([0.7, 0.2, 0.1]) == pytest.approx([1.0, 0.0, 0.0])

    def test_zero_temperature_splits_ties(self):
        emulator = self._emulator(0.0)
        assert emulator.model_weights([0.5, 0.5]) == pytest.approx([0.5, 0.5])

    def test_high_temperature_flattens_towards_uniform(self):
        low = self._emulator(0.5).model_weights([0.7, 0.3])
        mid = self._emulator(1.0).model_weights([0.7, 0.3])
        high = self._emulator(2.0).model_weights([0.7, 0.3])
        # Gap between the two models shrinks monotonically as T rises.
        assert (low[0] - low[1]) > (mid[0] - mid[1]) > (high[0] - high[1])

    def test_monotonic_in_temperature(self):
        gaps = []
        for temp in (0.25, 0.5, 1.0, 1.5, 2.0):
            w = self._emulator(temp).model_weights([0.6, 0.4])
            gaps.append(w[0] - w[1])
        assert gaps == sorted(gaps, reverse=True)

    def test_zero_weight_model_stays_excluded(self):
        emulator = self._emulator(1.5)
        got = emulator.model_weights([0.5, 0.0, 0.5])
        assert got[1] == 0.0
        assert sum(got) == pytest.approx(1.0)

    def test_vary_model_disabled_returns_normalized_weights(self):
        emulator = self._emulator(0.0, vary_model=False)
        assert emulator.model_weights([2.0, 2.0]) == pytest.approx([0.5, 0.5])

    def test_single_model_pool_is_unchanged(self):
        assert self._emulator(0.0).model_weights([1.0]) == pytest.approx([1.0])

    def test_all_zero_weights_returned_as_is(self):
        assert self._emulator(1.0).model_weights([0.0, 0.0]) == [0.0, 0.0]


class TestEffortJitter:
    def _emulator(self, temperature, **kwargs):
        kwargs.setdefault("base_effort", "medium")
        return TemperatureEmulator(
            config=TemperatureEmulationConfig(**kwargs),
            temperature=temperature,
            rng=random.Random(1234),
        )

    def test_zero_temperature_pins_base_effort(self):
        emulator = self._emulator(0.0)
        assert {emulator.sample_effort() for _ in range(50)} == {"medium"}

    def test_distribution_is_centred_on_base(self):
        emulator = self._emulator(1.0)
        dist = emulator.effort_distribution()
        base = list(DEFAULT_EFFORT_LADDER).index("medium")
        assert dist[base] == max(dist)
        assert sum(dist) == pytest.approx(1.0)

    def test_distribution_is_symmetric_around_interior_base(self):
        emulator = self._emulator(1.0, base_effort="high")
        dist = emulator.effort_distribution()
        idx = list(DEFAULT_EFFORT_LADDER).index("high")
        assert dist[idx - 1] == pytest.approx(dist[idx + 1])

    def test_higher_temperature_spreads_wider(self):
        base = list(DEFAULT_EFFORT_LADDER).index("medium")
        peak_low = self._emulator(0.5).effort_distribution()[base]
        peak_high = self._emulator(2.0).effort_distribution()[base]
        assert peak_low > peak_high

    def test_sampling_actually_varies_at_high_temperature(self):
        emulator = self._emulator(2.0)
        seen = {emulator.sample_effort() for _ in range(200)}
        assert len(seen) > 1
        assert seen.issubset(set(DEFAULT_EFFORT_LADDER))

    def test_vary_effort_disabled_pins_base(self):
        emulator = self._emulator(2.0, vary_effort=False)
        assert {emulator.sample_effort() for _ in range(50)} == {"medium"}

    def test_temperature_is_clamped_to_max(self):
        emulator = self._emulator(99.0)
        assert emulator.effective_temperature() == 2.0

    def test_invalid_base_effort_falls_back_to_middle_rung(self):
        emulator = self._emulator(0.0, base_effort="not-a-level")
        assert emulator.sample_effort() == DEFAULT_EFFORT_LADDER[len(DEFAULT_EFFORT_LADDER) // 2]

    def test_custom_ladder_is_respected(self):
        emulator = TemperatureEmulator(
            config=TemperatureEmulationConfig(effort_ladder=["low", "high"], base_effort="low"),
            temperature=0.0,
            rng=random.Random(0),
        )
        assert emulator.sample_effort() == "low"


class TestConfigFromDict:
    def test_empty_gives_defaults(self):
        cfg = TemperatureEmulationConfig.from_dict(None)
        assert cfg.enabled == "auto"
        assert cfg.vary_model is True

    def test_unknown_keys_ignored(self):
        cfg = TemperatureEmulationConfig.from_dict({"base_effort": "high", "bogus": 1})
        assert cfg.base_effort == "high"

    def test_passthrough_of_instance(self):
        original = TemperatureEmulationConfig(base_effort="max")
        assert TemperatureEmulationConfig.from_dict(original) is original

    def test_non_mapping_falls_back_to_defaults(self):
        assert TemperatureEmulationConfig.from_dict("nonsense").enabled == "auto"


class TestPoolActivation:
    """LLMPool decides when emulation applies; check the auto rule."""

    def _active(self, names, enabled="auto", provider=None):
        from skydiscover.llm.llm_pool import LLMPool

        cfgs = [LLMModelConfig(name=n, provider=provider) for n in names]
        return LLMPool._emulation_active(TemperatureEmulationConfig(enabled=enabled), cfgs)

    def test_auto_off_for_temperature_capable_models(self):
        assert self._active(["gpt-5"]) is False

    def test_auto_off_for_claude_that_still_accepts_temperature(self):
        assert self._active(["claude-opus-4-6"]) is False
        assert self._active(["claude-sonnet-4-5"]) is False

    def test_auto_on_for_latest_claude(self):
        assert self._active(["claude-opus-4-8"]) is True
        assert self._active(["claude-opus-5"]) is True

    def test_auto_on_for_claude_cli_provider(self):
        assert self._active(["haiku"], provider="claude_cli") is True

    def test_auto_on_if_any_pooled_model_lacks_temperature(self):
        assert self._active(["gpt-5", "claude-sonnet-5"]) is True

    def test_explicit_true_forces_on(self):
        assert self._active(["gpt-5"], enabled=True) is True
        assert self._active(["gpt-5"], enabled="true") is True

    def test_explicit_false_forces_off(self):
        assert self._active(["claude-opus-5"], enabled=False) is False
        assert self._active(["claude-opus-5"], enabled="off") is False
