"""Temperature emulation for models that expose no sampling temperature.

The latest Claude models (and the Claude Code CLI, which exposes no sampling
knob at all) reject or ignore the ``temperature`` parameter. That matters here
because SkyDiscover is an evolutionary search: the diversity of candidate
programs per iteration is a first-class search parameter, and AdaEvolve and
EvoX both actively steer it. Losing temperature means losing the explore/exploit
dial the search algorithms assume they have.

This module reconstructs that dial from the two knobs that *are* available:

1. **Which model is sampled** from a multi-model pool. Pool weights ``w`` are
   re-normalised as ``p_i ∝ w_i**(1/T)``. This is the textbook temperature
   transform on a categorical distribution: ``T → 0`` collapses to the
   highest-weight model (deterministic), ``T = 1`` reproduces the configured
   weights exactly, and ``T → ∞`` flattens towards uniform.

2. **How much reasoning effort** the call is given. Effort is drawn from a
   discrete Gaussian over an ordered ladder (low → max) centred on the
   configured base effort, with standard deviation ``σ = T * effort_spread``.
   ``T = 0`` pins effort to the base level; higher ``T`` wanders further. Effort
   controls how much the model explores its own solution space before
   committing, which is the closest available analogue to token-level sampling
   noise.

Both reduce to exactly the configured behaviour at ``T = 1`` with a
single-model pool and no spread, so turning emulation on is safe by default.
"""

from __future__ import annotations

import logging
import math
import random
import re
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("skydiscover.llm")

# Ordered weakest → strongest. Matches `claude --effort` levels; the first four
# also line up with OpenAI's reasoning_effort values.
DEFAULT_EFFORT_LADDER: Tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

DEFAULT_BASE_EFFORT = "medium"

# Below this, temperature is treated as exactly zero (fully deterministic).
_TEMPERATURE_EPSILON = 1e-3

# Model families that no longer accept a `temperature` request parameter.
# Matched against the bare model name, lowercased, after any provider prefix
# has been stripped.
#
# The cutoff is NOT simply "newer than 4.5". Per the Anthropic model docs,
# sampling parameters (temperature/top_p/top_k) were removed starting with
# Opus 4.7 and are rejected with a 400 on:
#     Opus 4.7, Opus 4.8, Opus 5, Sonnet 5, Fable 5, Mythos 5
# They are still ACCEPTED on:
#     Opus 4.6, Sonnet 4.6, and the entire 4.5 family (Opus/Sonnet/Haiku 4.5),
#     plus every Claude 3.x model.
# Getting this wrong in either direction is silently harmful: too broad drops a
# knob the model would have honoured, too narrow sends a parameter that 400s.
_NO_TEMPERATURE_PATTERNS: Tuple[str, ...] = (
    # Claude 5 generation and beyond: opus/sonnet/fable/mythos 5+.
    r"^claude-(?:opus|sonnet|fable|mythos)-(?:[5-9]|\d{2,})\b",
    r"^claude-mythos-preview$",
    # Opus 4.7 / 4.8 (and any later 4.x Opus). Opus 4.6 and below still accept
    # temperature, so the minor version is matched explicitly.
    r"^claude-opus-4-(?:[7-9]|\d{2,})\b",
    # Bare CLI aliases resolve to the latest model in each family. "haiku" is
    # deliberately excluded: it resolves to claude-haiku-4-5, which still
    # accepts temperature.
    r"^(?:opus|sonnet|fable|mythos)(?:-latest)?$",
)

_NO_TEMPERATURE_RE = re.compile("|".join(_NO_TEMPERATURE_PATTERNS), re.IGNORECASE)


def model_supports_temperature(name: Optional[str], provider: Optional[str] = None) -> bool:
    """Whether *name* still accepts a ``temperature`` request parameter."""
    if (provider or "").lower() in ("claude_cli", "claude-cli", "codex_cli", "codex-cli"):
        # Neither CLI exposes any sampling parameter.
        return False
    if not name:
        return True
    bare = name.split("/")[-1].strip().lower()
    # A bare name still carrying a local-CLI prefix (provider not yet resolved).
    if name.strip().lower().startswith(("claude_cli/", "claude-cli/", "codex_cli/", "codex-cli/")):
        return False
    return not _NO_TEMPERATURE_RE.match(bare)


@dataclass
class TemperatureEmulationConfig:
    """Configuration for :class:`TemperatureEmulator`.

    ``enabled`` accepts "auto" (default), True, or False. Under "auto",
    emulation activates only when at least one pooled model rejects a real
    temperature parameter, so existing OpenAI-backed runs are untouched.
    """

    enabled: Any = "auto"

    # Reshape the pool's model weights by temperature.
    vary_model: bool = True

    # Jitter reasoning effort around base_effort by temperature.
    vary_effort: bool = True

    effort_ladder: List[str] = field(default_factory=lambda: list(DEFAULT_EFFORT_LADDER))
    base_effort: Optional[str] = DEFAULT_BASE_EFFORT

    # sigma = temperature * effort_spread, in ladder steps.
    effort_spread: float = 1.0

    # Clamp so a wild temperature cannot make weights explode or vanish.
    min_temperature: float = 0.0
    max_temperature: float = 2.0

    @classmethod
    def from_dict(cls, data: Any) -> "TemperatureEmulationConfig":
        """Build from a YAML mapping, ignoring unknown keys with a warning."""
        if isinstance(data, cls):
            return data
        if not data:
            return cls()
        if not isinstance(data, dict):
            logger.warning(
                "llm.temperature_emulation must be a mapping, got %s; using defaults.",
                type(data).__name__,
            )
            return cls()

        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            logger.warning(
                "Ignoring unknown llm.temperature_emulation keys: %s. Valid keys: %s",
                ", ".join(sorted(unknown)),
                ", ".join(sorted(known)),
            )
        return cls(**{k: v for k, v in data.items() if k in known})


class TemperatureEmulator:
    """Turns a scalar temperature into (model choice, reasoning effort).

    Stateless apart from the RNG, so it can be shared by every pool in a run.
    """

    def __init__(
        self,
        config: Optional[TemperatureEmulationConfig] = None,
        temperature: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ):
        self.config = config or TemperatureEmulationConfig()
        self.temperature = 1.0 if temperature is None else float(temperature)
        self.rng = rng or random.Random()

        ladder = [str(e).strip().lower() for e in (self.config.effort_ladder or []) if e]
        self.effort_ladder: List[str] = ladder or list(DEFAULT_EFFORT_LADDER)

        base = (self.config.base_effort or DEFAULT_BASE_EFFORT).strip().lower()
        if base not in self.effort_ladder:
            logger.warning(
                "base_effort %r is not in effort_ladder %s; using the middle rung instead.",
                self.config.base_effort,
                self.effort_ladder,
            )
            base = self.effort_ladder[len(self.effort_ladder) // 2]
        self.base_effort = base
        self._base_index = self.effort_ladder.index(base)

    # ------------------------------------------------------------------
    # Temperature handling
    # ------------------------------------------------------------------

    def effective_temperature(self, override: Optional[float] = None) -> float:
        value = self.temperature if override is None else float(override)
        return max(self.config.min_temperature, min(self.config.max_temperature, value))

    # ------------------------------------------------------------------
    # 1. Model distribution
    # ------------------------------------------------------------------

    def model_weights(
        self, weights: Sequence[float], temperature: Optional[float] = None
    ) -> List[float]:
        """Reshape pool *weights* by temperature.

        ``p_i ∝ w_i**(1/T)``. Zero-weight models stay at zero. At ``T`` near
        zero the mass collapses onto the maximum-weight entries (ties shared).
        """
        weights = [max(0.0, float(w)) for w in weights]
        total = sum(weights)
        if not weights or total <= 0:
            return list(weights)

        if not self.config.vary_model or len(weights) == 1:
            return [w / total for w in weights]

        temp = self.effective_temperature(temperature)

        if temp <= _TEMPERATURE_EPSILON:
            # Deterministic: all mass on the argmax (shared across ties).
            peak = max(weights)
            winners = [1.0 if w == peak else 0.0 for w in weights]
            count = sum(winners)
            return [w / count for w in winners]

        exponent = 1.0 / temp
        try:
            shaped = [(w / total) ** exponent if w > 0 else 0.0 for w in weights]
        except (OverflowError, ValueError):
            return [w / total for w in weights]

        shaped_total = sum(shaped)
        if shaped_total <= 0 or not math.isfinite(shaped_total):
            return [w / total for w in weights]
        return [w / shaped_total for w in shaped]

    # ------------------------------------------------------------------
    # 2. Reasoning effort
    # ------------------------------------------------------------------

    def effort_distribution(self, temperature: Optional[float] = None) -> List[float]:
        """Discrete Gaussian over the effort ladder, centred on base_effort."""
        n = len(self.effort_ladder)
        if not self.config.vary_effort or n == 1:
            return [1.0 if i == self._base_index else 0.0 for i in range(n)]

        temp = self.effective_temperature(temperature)
        sigma = temp * max(0.0, self.config.effort_spread)

        if sigma <= _TEMPERATURE_EPSILON:
            return [1.0 if i == self._base_index else 0.0 for i in range(n)]

        two_sigma_sq = 2.0 * sigma * sigma
        raw = [math.exp(-((i - self._base_index) ** 2) / two_sigma_sq) for i in range(n)]
        total = sum(raw)
        if total <= 0 or not math.isfinite(total):
            return [1.0 if i == self._base_index else 0.0 for i in range(n)]
        return [r / total for r in raw]

    def sample_effort(self, temperature: Optional[float] = None) -> str:
        """Draw one effort level for this call."""
        probabilities = self.effort_distribution(temperature)
        index = self.rng.choices(range(len(self.effort_ladder)), weights=probabilities, k=1)[0]
        return self.effort_ladder[index]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def describe(self, weights: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        temp = self.effective_temperature()
        info: Dict[str, Any] = {
            "temperature": temp,
            "base_effort": self.base_effort,
            "effort_ladder": list(self.effort_ladder),
            "effort_distribution": [round(p, 4) for p in self.effort_distribution()],
        }
        if weights is not None:
            info["model_weights"] = [round(w, 4) for w in self.model_weights(weights)]
        return info
