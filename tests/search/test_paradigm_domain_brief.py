"""Domain-brief override and call-context hygiene in ParadigmGenerator.

The built-in paradigm prompt was written for algorithmic benchmarks: it
demands single-function library-call ideas in "library.function" format,
which is meaningless in domains where a paradigm is a mechanism, not a
function name. `domain_brief` replaces that guidance while keeping the
JSON output contract. Separately, generate() must restore the caller's
call-log phase so the iteration call after a paradigm generation is not
logged as phase="paradigm".
"""

import asyncio
import json
import types

import pytest

import skydiscover.search.adaevolve.paradigm.generator as gen_mod
from skydiscover.llm.call_log import get_call_context, set_call_context
from skydiscover.search.adaevolve.paradigm.generator import ParadigmGenerator


class _StubPool:
    """Minimal pool: returns a fixed response; never agentic."""

    models = []

    def __init__(self, text):
        self._text = text
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        if isinstance(self._text, Exception):
            raise self._text
        return types.SimpleNamespace(text=self._text)


def _ideas_json(n=3):
    return json.dumps(
        {
            "ideas": [
                {
                    "idea": f"mechanism {i}",
                    "description": "detail",
                    "what_to_optimize": "score",
                    "cautions": "none",
                    "approach_type": f"family.mech{i}",
                }
                for i in range(n)
            ]
        }
    )


def test_default_prompt_keeps_library_guidance():
    g = ParadigmGenerator(llm_pool=_StubPool(""), num_paradigms=3)
    tech = g._build_techniques_section()
    fmt = g._build_output_format_section()
    assert "single-function library calls" in tech
    assert '"library.function" format' in fmt
    assert "scipy.optimize.minimize" in fmt


def test_domain_brief_replaces_library_guidance():
    brief = "Ideas must be documented economic mechanisms, cited in one line."
    g = ParadigmGenerator(llm_pool=_StubPool(""), num_paradigms=3, domain_brief=brief)
    tech = g._build_techniques_section()
    fmt = g._build_output_format_section()

    assert brief in tech
    assert "single-function" not in tech
    assert "scipy" not in tech

    # Library-centric idea format is gone...
    assert '"library.function"' not in fmt
    assert "scipy.optimize.minimize" not in fmt
    assert "MECHANISM" in fmt
    # ...but the JSON output contract is intact.
    assert '"ideas"' in fmt
    assert "exactly 3 idea objects" in fmt
    assert "approach_type" in fmt


def test_domain_brief_is_stripped_and_optional():
    g = ParadigmGenerator(llm_pool=_StubPool(""), domain_brief="   \n")
    assert g.domain_brief == ""
    assert "Technique Guidance" in g._build_techniques_section()


def test_generate_restores_call_phase_on_success():
    set_call_context(iteration=7, phase="iteration")
    g = ParadigmGenerator(llm_pool=_StubPool(_ideas_json()), num_paradigms=3)
    paradigms = asyncio.run(g.generate("solution", 0.0, []))
    assert len(paradigms) == 3
    assert get_call_context().get("phase") == "iteration"
    assert get_call_context().get("iteration") == 7


def test_generate_restores_call_phase_on_failure(monkeypatch):
    monkeypatch.setattr(gen_mod, "MAX_RETRIES", 1)  # no backoff sleeps
    set_call_context(iteration=9, phase="iteration")
    g = ParadigmGenerator(llm_pool=_StubPool(RuntimeError("boom")), num_paradigms=3)
    paradigms = asyncio.run(g.generate("solution", 0.0, []))
    assert paradigms == []
    assert get_call_context().get("phase") == "iteration"
