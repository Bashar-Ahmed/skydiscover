"""LLM module"""

from skydiscover.llm.base import LLMInterface, LLMResponse
from skydiscover.llm.llm_pool import LLMPool, create_llm_backend
from skydiscover.llm.openai import OpenAILLM
from skydiscover.llm.rate_limit import (
    UsageLimitError,
    UsageLimitGate,
    get_usage_limit_gate,
    parse_usage_limit,
)
from skydiscover.llm.temperature import (
    TemperatureEmulationConfig,
    TemperatureEmulator,
    model_supports_temperature,
)

__all__ = [
    "LLMInterface",
    "LLMResponse",
    "OpenAILLM",
    "LLMPool",
    "create_llm_backend",
    "UsageLimitError",
    "UsageLimitGate",
    "get_usage_limit_gate",
    "parse_usage_limit",
    "TemperatureEmulationConfig",
    "TemperatureEmulator",
    "model_supports_temperature",
    "ClaudeCLILLM",
    "CodexCLILLM",
]


def __getattr__(name: str):
    # Lazy so importing skydiscover.llm does not probe for a CLI binary.
    if name == "ClaudeCLILLM":
        from skydiscover.llm.claude_cli import ClaudeCLILLM

        return ClaudeCLILLM
    if name == "CodexCLILLM":
        from skydiscover.llm.codex_cli import CodexCLILLM

        return CodexCLILLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
