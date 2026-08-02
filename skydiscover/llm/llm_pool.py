"""LLM pool -- weighted sampling over one or more LLM backends."""

import asyncio
import logging
import random
from typing import Any, Dict, List, Optional

from skydiscover.config import LLMModelConfig, is_local_provider
from skydiscover.llm.base import LLMInterface, LLMResponse
from skydiscover.llm.openai import OpenAILLM
from skydiscover.llm.temperature import (
    TemperatureEmulationConfig,
    TemperatureEmulator,
    model_supports_temperature,
)

logger = logging.getLogger("skydiscover.llm")


def _detect_provider(model_cfg: LLMModelConfig) -> str:
    """Best-effort provider for a model config that predates provider tagging."""
    provider = (model_cfg.provider or "").lower()
    if provider:
        return provider
    name = model_cfg.name or ""
    for prefix in ("claude_cli/", "claude-cli/"):
        if name.startswith(prefix):
            return "claude_cli"
    return ""


def create_llm_backend(model_cfg: LLMModelConfig) -> LLMInterface:
    """Instantiate the LLM backend a model config asks for."""
    if model_cfg.init_client:
        return model_cfg.init_client(model_cfg)

    provider = _detect_provider(model_cfg)
    if is_local_provider(provider):
        # Imported lazily: constructing it probes for the `claude` binary, and
        # runs that never use the CLI should not pay that cost or that failure.
        from skydiscover.llm.claude_cli import ClaudeCLILLM

        return ClaudeCLILLM(model_cfg)

    return OpenAILLM(model_cfg)


class LLMPool:
    """Weighted pool of LLM backends. Samples one per generate() call.

    When the pooled models expose no sampling temperature (Claude Opus 4.7 and
    later, Sonnet 5, Fable 5, and the Claude Code CLI), the configured
    ``temperature`` is instead applied by reshaping the model-selection weights
    and jittering reasoning effort. See :mod:`skydiscover.llm.temperature`.
    """

    def __init__(
        self,
        models_cfg: List[LLMModelConfig],
        temperature_emulation: Any = None,
        temperature: Optional[float] = None,
    ):
        if not models_cfg:
            raise ValueError("LLMPool requires at least one model config")

        self.models_cfg = models_cfg

        # Validate weights before creating clients to fail fast on bad config.
        self.weights = [m.weight for m in models_cfg]
        if any(w < 0 for w in self.weights):
            raise ValueError("LLMPool model weights must be non-negative")
        total = sum(self.weights)
        if total <= 0:
            raise ValueError("LLMPool model weights must sum to a positive value")
        self.weights = [w / total for w in self.weights]

        self.models = [create_llm_backend(model_cfg) for model_cfg in models_cfg]
        self.random_state = random.Random()

        # ── Temperature emulation ─────────────────────────────────────
        if temperature is None:
            temperature = next(
                (m.temperature for m in models_cfg if m.temperature is not None), None
            )
        self.temperature = temperature

        self.temperature_emulator: Optional[TemperatureEmulator] = None
        emulation_cfg = TemperatureEmulationConfig.from_dict(temperature_emulation)
        if self._emulation_active(emulation_cfg, models_cfg):
            base_effort = emulation_cfg.base_effort
            if base_effort is None:
                base_effort = next(
                    (m.reasoning_effort for m in models_cfg if m.reasoning_effort), None
                )
            resolved = TemperatureEmulationConfig(
                enabled=emulation_cfg.enabled,
                vary_model=emulation_cfg.vary_model,
                vary_effort=emulation_cfg.vary_effort,
                effort_ladder=list(emulation_cfg.effort_ladder),
                base_effort=base_effort,
                effort_spread=emulation_cfg.effort_spread,
                min_temperature=emulation_cfg.min_temperature,
                max_temperature=emulation_cfg.max_temperature,
            )
            self.temperature_emulator = TemperatureEmulator(
                config=resolved,
                temperature=self.temperature,
                rng=self.random_state,
            )
            self.effective_weights = self.temperature_emulator.model_weights(self.weights)
            # A run builds several pools with identical settings; log once.
            summary = self.temperature_emulator.describe(self.weights)
            if not hasattr(logger, "_logged_emulation"):
                logger._logged_emulation = set()
            summary_key = repr(sorted(summary.items(), key=lambda kv: kv[0]))
            if summary_key not in logger._logged_emulation:
                logger.info("Temperature emulation active: %s", summary)
                logger._logged_emulation.add(summary_key)
        else:
            self.effective_weights = list(self.weights)

        # Logging
        if len(models_cfg) > 1:
            pool_key = tuple((c.name, w) for c, w in zip(models_cfg, self.effective_weights))
            if not hasattr(logger, "_logged_pools"):
                logger._logged_pools = set()
            if pool_key not in logger._logged_pools:
                parts = ", ".join(
                    f"{c.name}={w:.2f}" for c, w in zip(models_cfg, self.effective_weights)
                )
                logger.info(f"Pool weights: {parts}")
                logger._logged_pools.add(pool_key)

    @staticmethod
    def _emulation_active(
        cfg: TemperatureEmulationConfig, models_cfg: List[LLMModelConfig]
    ) -> bool:
        enabled = cfg.enabled
        if isinstance(enabled, str):
            enabled = enabled.strip().lower()
            if enabled in ("false", "off", "no", "0"):
                return False
            if enabled in ("true", "on", "yes", "1"):
                return True
            # "auto": emulate only when a real temperature cannot be sent.
            return any(
                not model_supports_temperature(m.name, _detect_provider(m)) for m in models_cfg
            )
        return bool(enabled)

    def _sample_model(self):
        """
        Simple weighted sampling mechanism. Override this to implement a more complex sampling mechanism.
        """
        idx = self.random_state.choices(
            range(len(self.models)), weights=self.effective_weights, k=1
        )[0]
        return self.models[idx]

    def _emulated_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Per-call parameters derived from the emulated temperature.

        An explicit caller-supplied value always wins; this only fills gaps.
        """
        if self.temperature_emulator is None:
            return kwargs
        if kwargs.get("reasoning_effort") is not None:
            return kwargs

        effort = self.temperature_emulator.sample_effort(kwargs.get("temperature"))
        if effort is None:
            return kwargs

        merged = dict(kwargs)
        merged["reasoning_effort"] = effort
        logger.debug("Temperature emulation sampled reasoning_effort=%s", effort)
        return merged

    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        """Sample a model and generate a response."""
        model = self._sample_model()
        return await model.generate(system_message, messages, **self._emulated_kwargs(kwargs))

    async def generate_all(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> List[LLMResponse]:
        """Generate using all models concurrently."""
        call_kwargs = self._emulated_kwargs(kwargs)
        return await asyncio.gather(
            *(model.generate(system_message, messages, **call_kwargs) for model in self.models)
        )
