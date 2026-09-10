"""Per-model effort-ladder overrides in LLMPool.

Models saturate at different reasoning-effort levels, so a single
pool-level ladder centred on one rung can badly mis-place individual
models. A model config may set effort_ladder / base_effort /
effort_spread; when pool-level emulation is active, that model draws
its per-call effort from its own emulator while the others keep the
pool's. Unset fields inherit the pool settings.
"""

import pytest

import skydiscover.llm.llm_pool as pool_mod
from skydiscover.config import LLMModelConfig
from skydiscover.llm.llm_pool import LLMPool


class _DummyBackend:
    supports_native_agentic = False

    def __init__(self, cfg):
        self.model = cfg.name
        self.reasoning_effort = cfg.reasoning_effort


@pytest.fixture
def make_pool(monkeypatch):
    monkeypatch.setattr(pool_mod, "create_llm_backend", lambda cfg: _DummyBackend(cfg))

    def _make(models_cfg):
        return LLMPool(
            models_cfg,
            temperature_emulation={
                "enabled": True,
                "vary_model": False,
                "vary_effort": True,
                "effort_ladder": ["medium", "high", "xhigh", "max"],
                "base_effort": "xhigh",
                "effort_spread": 1.0,
            },
            temperature=0.7,
        )

    return _make


def _cfgs(**overrides_for_first):
    return [
        LLMModelConfig(name="codex_cli/model-a", weight=1.0, **overrides_for_first),
        LLMModelConfig(name="codex_cli/model-b", weight=1.0),
    ]


def test_override_builds_a_per_model_emulator(make_pool):
    pool = make_pool(_cfgs(effort_ladder=["medium"], base_effort="medium"))
    assert pool._model_emulators[0] is not None
    assert pool._model_emulators[1] is None
    assert pool._model_emulators[0].effort_ladder == ["medium"]


def test_overridden_model_draws_from_its_own_ladder(make_pool):
    pool = make_pool(_cfgs(effort_ladder=["medium"], base_effort="medium"))
    for _ in range(30):
        assert pool._emulated_kwargs({}, model_index=0)["reasoning_effort"] == "medium"
    # The other model keeps the pool ladder (xhigh-centred: never "medium"-only)
    seen = {pool._emulated_kwargs({}, model_index=1)["reasoning_effort"] for _ in range(60)}
    assert seen <= {"medium", "high", "xhigh", "max"}
    assert seen != {"medium"}


def test_no_index_falls_back_to_pool_emulator(make_pool):
    pool = make_pool(_cfgs(effort_ladder=["medium"], base_effort="medium"))
    seen = {pool._emulated_kwargs({})["reasoning_effort"] for _ in range(60)}
    assert seen != {"medium"}


def test_partial_override_inherits_pool_settings(make_pool):
    # Only base_effort set: ladder comes from the pool, centre moves.
    pool = make_pool(_cfgs(base_effort="medium"))
    emu = pool._model_emulators[0]
    assert emu is not None
    assert emu.effort_ladder == ["medium", "high", "xhigh", "max"]
    assert emu.base_effort == "medium"


def test_explicit_caller_effort_always_wins(make_pool):
    pool = make_pool(_cfgs(effort_ladder=["medium"], base_effort="medium"))
    out = pool._emulated_kwargs({"reasoning_effort": "max"}, model_index=0)
    assert out["reasoning_effort"] == "max"


def test_no_overrides_means_no_per_model_emulators(make_pool):
    pool = make_pool(_cfgs())
    assert pool._model_emulators == [None, None]


def test_emulation_disabled_still_defines_the_attribute(monkeypatch):
    monkeypatch.setattr(pool_mod, "create_llm_backend", lambda cfg: _DummyBackend(cfg))
    pool = LLMPool(
        _cfgs(effort_ladder=["medium"]),
        temperature_emulation={"enabled": False},
        temperature=0.7,
    )
    assert pool._model_emulators == [None, None]
    assert pool._emulated_kwargs({}, model_index=0) == {}
