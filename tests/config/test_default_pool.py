"""The default model pool (no config, or configs/default.yaml)."""

from skydiscover.config import LLMConfig, LLMModelConfig, load_config

ITERATION = ["gpt-5.6-sol", "gpt-5.6-terra", "claude-sonnet-5"]
PARADIGM = ["gpt-6-astra", "claude-fable-5-1", "claude-opus-5"]


def test_code_default_is_the_calibrated_pool():
    cfg = LLMConfig()
    assert [m.name for m in cfg.models] == ITERATION
    assert [m.weight for m in cfg.models] == [0.3, 0.4, 0.3]
    assert [m.base_effort for m in cfg.models] == ["medium", "xhigh", "high"]
    assert [m.name for m in cfg.guide_models] == PARADIGM
    assert all(m.timeout == 5400 for m in cfg.guide_models)
    assert cfg.timeout == 3600
    emu = cfg.temperature_emulation
    assert emu["effort_ladder"] == ["medium", "high", "xhigh"]
    assert emu["base_effort"] == "high" and emu["effort_spread"] == 1.5
    assert emu["vary_model"] is False


def test_default_yaml_matches_code_default():
    cfg = load_config("configs/default.yaml").llm
    assert [m.name for m in cfg.models] == ITERATION
    assert [m.name for m in cfg.guide_models] == PARADIGM


def test_user_pool_guides_itself():
    """A user-supplied `models` list without `guide_models` must not pick up
    the default paradigm pool's other backends."""
    cfg = LLMConfig(models=[LLMModelConfig(name="claude_cli/opus")])
    assert [m.name for m in cfg.guide_models] == ["opus"]
