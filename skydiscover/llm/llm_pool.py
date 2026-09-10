"""LLM pool -- weighted sampling over one or more LLM backends."""

import asyncio
import hashlib
import json
import logging
import random
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from skydiscover.config import LLMModelConfig, is_local_provider
from skydiscover.llm.base import LLMInterface, LLMResponse
from skydiscover.llm.call_log import GLOBAL_CALL_LOG, describe_backend
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
    for prefix in ("codex_cli/", "codex-cli/"):
        if name.startswith(prefix):
            return "codex_cli"
    return ""


def create_llm_backend(model_cfg: LLMModelConfig) -> LLMInterface:
    """Instantiate the LLM backend a model config asks for."""
    if model_cfg.init_client:
        return model_cfg.init_client(model_cfg)

    provider = _detect_provider(model_cfg)
    if is_local_provider(provider):
        # Imported lazily: constructing either backend probes for its binary,
        # and runs that never use a CLI should not pay that cost or that
        # failure.
        if provider.replace("-", "_") == "codex_cli":
            from skydiscover.llm.codex_cli import CodexCLILLM

            return CodexCLILLM(model_cfg)

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
            # Per-model effort overrides: a model that sets effort_ladder /
            # base_effort / effort_spread gets its own emulator; the rest
            # share the pool's. Models saturate at different effort rungs,
            # so one shared centre can badly mis-place individual models.
            self._model_emulators: List[Optional[TemperatureEmulator]] = []
            for m in models_cfg:
                if m.effort_ladder or m.base_effort or m.effort_spread is not None:
                    per_cfg = TemperatureEmulationConfig(
                        enabled=resolved.enabled,
                        vary_model=resolved.vary_model,
                        vary_effort=resolved.vary_effort,
                        effort_ladder=(
                            list(m.effort_ladder)
                            if m.effort_ladder
                            else list(resolved.effort_ladder)
                        ),
                        base_effort=m.base_effort or resolved.base_effort,
                        effort_spread=(
                            m.effort_spread
                            if m.effort_spread is not None
                            else resolved.effort_spread
                        ),
                        min_temperature=resolved.min_temperature,
                        max_temperature=resolved.max_temperature,
                    )
                    emulator = TemperatureEmulator(
                        config=per_cfg, temperature=self.temperature, rng=self.random_state
                    )
                    self._model_emulators.append(emulator)
                    logger.info(
                        "Per-model effort override for %s: ladder=%s, base=%s, spread=%s",
                        m.name,
                        emulator.effort_ladder,
                        emulator.base_effort,
                        per_cfg.effort_spread,
                    )
                else:
                    self._model_emulators.append(None)
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
            self._model_emulators = [None] * len(models_cfg)
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

    # ── No-repeat-model-per-prompt enforcement ────────────────────────
    #: history key -> set of model indices already used for that exact prompt.
    #: Near-deterministic backends (the Codex/Claude CLIs) replay themselves on
    #: an identical prompt, so redrawing the same model wastes the evaluation on
    #: a duplicate program. Class-level so the rule survives the several pools a
    #: run builds, and guarded by a lock because EvoX drives pools from worker
    #: threads with their own event loops.
    _prompt_history: "OrderedDict[str, set]" = OrderedDict()
    _prompt_history_lock = threading.Lock()
    _PROMPT_HISTORY_MAX = 4096

    @staticmethod
    def _prompt_key(system_message: str, messages: List[Dict[str, Any]]) -> str:
        try:
            payload = json.dumps([system_message, messages], sort_keys=True, default=str)
        except (TypeError, ValueError):
            payload = repr((system_message, messages))
        return hashlib.blake2b(payload.encode(), digest_size=12).hexdigest()

    def _history_key(self, prompt_key: str) -> str:
        """Namespace the prompt by this pool's roster.

        The history stores model *indices*, so it is only meaningful within one
        roster. A run builds several pools — llms, evaluator_llms, guide_llms —
        that may hold different models, and without this they share one entry:
        the shorter pool's cycle-reset wipes the longer pool's history, silently
        removing the guarantee from the pool that needed it.
        """
        roster = getattr(self, "_roster_key", None)
        if roster is None:
            roster = hashlib.blake2b(
                repr([c.name for c in self.models_cfg]).encode(), digest_size=6
            ).hexdigest()
            self._roster_key = roster
        return f"{prompt_key}:{roster}"

    def _sample_model(self, prompt_key: Optional[str] = None):
        """
        Weighted sampling. With ``prompt_key``, a model already drawn for that
        exact prompt is excluded until every pooled model has been tried once,
        then the cycle restarts. Override for custom sampling.
        """
        track = prompt_key is not None and len(self.models) > 1
        used: set = set()
        key = self._history_key(prompt_key) if track else None

        if track:
            with LLMPool._prompt_history_lock:
                # Copy: the stored set is mutated below, and an alias would make
                # the debug line name the model we just chose as "excluded".
                used = set(LLMPool._prompt_history.get(key, ()))
            if len(used) >= len(self.models):
                used = set()  # every model tried: cycle again

        weights = [0.0 if i in used else w for i, w in enumerate(self.effective_weights)]
        if sum(weights) <= 0:
            weights = list(self.effective_weights)
        idx = self.random_state.choices(range(len(self.models)), weights=weights, k=1)[0]

        if track:
            with LLMPool._prompt_history_lock:
                hist = LLMPool._prompt_history
                entry = hist.setdefault(key, set())
                if len(entry) >= len(self.models):
                    entry.clear()
                entry.add(idx)
                hist.move_to_end(key)
                while len(hist) > LLMPool._PROMPT_HISTORY_MAX:
                    hist.popitem(last=False)
            if used:
                logger.debug(
                    "No-repeat rule: prompt seen before; excluded %s, chose %s",
                    sorted(self.models_cfg[i].name for i in used),
                    self.models_cfg[idx].name,
                )
        return self.models[idx]

    def _emulated_kwargs(
        self, kwargs: Dict[str, Any], model_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """Per-call parameters derived from the emulated temperature.

        An explicit caller-supplied value always wins; this only fills gaps.
        With ``model_index``, a model carrying a per-model effort override
        draws from its own emulator instead of the pool's.
        """
        emulator = self.temperature_emulator
        # getattr: tests (and any embedder) may build a pool without __init__
        per_model = getattr(self, "_model_emulators", None) or []
        if model_index is not None and 0 <= model_index < len(per_model):
            emulator = per_model[model_index] or emulator
        if emulator is None:
            return kwargs
        if kwargs.get("reasoning_effort") is not None:
            return kwargs

        effort = emulator.sample_effort(kwargs.get("temperature"))
        if effort is None:
            return kwargs

        merged = dict(kwargs)
        merged["reasoning_effort"] = effort
        logger.debug("Temperature emulation sampled reasoning_effort=%s", effort)
        return merged

    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        """Sample a model and generate a response.

        The prompt is hashed so repeated identical prompts (same parent, same
        context) never redraw the same near-deterministic model before the pool
        has cycled.
        """
        model = self._sample_model(self._prompt_key(system_message, messages))
        try:
            model_index = self.models.index(model)
        except ValueError:
            model_index = None
        call_kwargs = self._emulated_kwargs(kwargs, model_index=model_index)
        # The only place where both the sampled model and the sampled effort are
        # known; neither is recoverable from the config afterwards.
        details = {
            **describe_backend(model),
            "reasoning_effort": call_kwargs.get("reasoning_effort")
            or getattr(model, "reasoning_effort", None),
            "emulated": self.temperature_emulator is not None,
            "temperature": self.temperature,
            "agentic": bool(call_kwargs.get("agentic")),
        }
        started = time.monotonic()
        try:
            response = await model.generate(system_message, messages, **call_kwargs)
        except BaseException as exc:
            GLOBAL_CALL_LOG.record(
                "generate_failed",
                duration_s=round(time.monotonic() - started, 3),
                error_type=type(exc).__name__,
                error=str(exc)[:300],
                **details,
            )
            raise
        GLOBAL_CALL_LOG.record(
            "generate",
            duration_s=round(time.monotonic() - started, 3),
            response_chars=len(response.text or ""),
            **details,
        )
        return response

    async def generate_all(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> List[LLMResponse]:
        """Generate using all models concurrently."""
        call_kwargs = self._emulated_kwargs(kwargs)
        GLOBAL_CALL_LOG.record(
            "generate_all",
            models=[getattr(m, "model", None) for m in self.models],
            reasoning_effort=call_kwargs.get("reasoning_effort"),
        )
        return await asyncio.gather(
            *(model.generate(system_message, messages, **call_kwargs) for model in self.models)
        )
