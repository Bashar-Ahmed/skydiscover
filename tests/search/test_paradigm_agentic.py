"""paradigm_agentic: the guide pool's paradigm calls may run the backend's
native agent loop (research library + web search) independently of the
solution-generation `agentic` block."""

import asyncio
from types import SimpleNamespace

from skydiscover.config import AgenticConfig, Config
from skydiscover.search.adaevolve.paradigm.generator import ParadigmGenerator


class _Backend:
    def __init__(self, native):
        self.supports_native_agentic = native


class _Pool:
    def __init__(self, native=True):
        self.models = [_Backend(native), _Backend(native)]
        self.calls = []

    async def generate(self, system_message, messages, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text='{"ideas": [{"idea": "x", "description": "d", '
                               '"what_to_optimize": "w", "cautions": "c", '
                               '"approach_type": "a"}]}')


def _gen(pool, agentic):
    return ParadigmGenerator(llm_pool=pool, system_message="sys", num_paradigms=1,
                             agentic=agentic)


def test_enabled_agentic_passes_native_loop_kwargs():
    pool = _Pool(native=True)
    cfg = AgenticConfig(enabled=True, codebase_root="/tmp/lib", max_steps=7,
                        overall_timeout=1234.0)
    ideas = asyncio.run(_gen(pool, cfg).generate("prog", 0.1))
    assert len(ideas) == 1
    kw = pool.calls[0]
    assert kw["agentic"] is True
    assert kw["codebase_root"] == "/tmp/lib"
    assert kw["max_steps"] == 7
    assert kw["timeout"] == 1234.0
    assert kw["response_format"]["type"] == "json_schema"   # schema still enforced


def test_disabled_or_absent_agentic_keeps_plain_generation():
    for agentic in (None, AgenticConfig(enabled=False, codebase_root="/tmp/lib")):
        pool = _Pool(native=True)
        asyncio.run(_gen(pool, agentic).generate("prog", 0.1))
        assert "agentic" not in pool.calls[0]


def test_pool_with_a_non_native_backend_falls_back_to_plain():
    pool = _Pool(native=False)
    cfg = AgenticConfig(enabled=True, codebase_root="/tmp/lib")
    asyncio.run(_gen(pool, cfg).generate("prog", 0.1))
    assert "agentic" not in pool.calls[0]


def test_config_parses_paradigm_agentic_block():
    cfg = Config.from_dict({
        "paradigm_agentic": {"enabled": True, "codebase_root": "/tmp/lib",
                             "max_steps": 8, "allowed_extensions": [".md"]},
    })
    assert cfg.paradigm_agentic.enabled is True
    assert cfg.paradigm_agentic.codebase_root == "/tmp/lib"
    assert cfg.paradigm_agentic.allowed_extensions == (".md",)
    assert cfg.agentic.enabled is False          # independent of the solution block
    assert cfg.to_dict()["paradigm_agentic"]["max_steps"] == 8
