"""Provider resolution and backend routing for the claude_cli provider."""

import shutil

import pytest

from skydiscover.config import (
    Config,
    LLMConfig,
    LLMModelConfig,
    apply_overrides,
    is_local_provider,
    load_config,
)
from skydiscover.llm.llm_pool import _detect_provider, create_llm_backend


class TestIsLocalProvider:
    @pytest.mark.parametrize("provider", ["claude_cli", "claude-cli", "CLAUDE_CLI"])
    def test_local(self, provider):
        assert is_local_provider(provider) is True

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "ollama", None, ""])
    def test_not_local(self, provider):
        assert is_local_provider(provider) is False


class TestConfigResolution:
    def test_prefix_sets_provider_and_strips_name(self):
        cfg = LLMConfig(models=[LLMModelConfig(name="claude_cli/sonnet")])
        model = cfg.models[0]
        assert model.provider == "claude_cli"
        assert model.name == "sonnet"

    def test_no_api_base_or_key_is_invented(self):
        """Regression: shared copies of `models` used to be re-parsed from the
        already-stripped bare name and given the OpenAI default endpoint."""
        cfg = LLMConfig(models=[LLMModelConfig(name="claude_cli/sonnet")])
        for model in cfg.models + cfg.evaluator_models + cfg.guide_models:
            assert model.api_base is None, "local provider must not inherit an endpoint"
            assert model.api_key is None, "local provider must not inherit an API key"

    def test_evaluator_and_guide_models_inherit_provider(self):
        cfg = LLMConfig(models=[LLMModelConfig(name="claude_cli/opus")])
        assert cfg.evaluator_models[0].provider == "claude_cli"
        assert cfg.guide_models[0].provider == "claude_cli"

    def test_mixed_pool_resolves_each_model_independently(self):
        cfg = LLMConfig(
            models=[
                LLMModelConfig(name="claude_cli/opus", weight=0.5),
                LLMModelConfig(name="gpt-5", weight=0.5),
            ]
        )
        local, remote = cfg.models
        assert local.provider == "claude_cli" and local.api_base is None
        assert remote.provider == "openai"
        assert remote.api_base == "https://api.openai.com/v1"

    def test_shared_generation_params_still_propagate(self):
        cfg = LLMConfig(
            models=[LLMModelConfig(name="claude_cli/sonnet")],
            max_tokens=1234,
            timeout=99,
        )
        assert cfg.models[0].max_tokens == 1234
        assert cfg.models[0].timeout == 99

    def test_cli_model_override_does_not_require_api_base(self):
        config = load_config(None)
        apply_overrides(config, model="claude_cli/haiku")
        model = config.llm.models[0]
        assert model.provider == "claude_cli"
        assert model.name == "haiku"
        assert model.api_base is None

    def test_unknown_local_style_provider_still_requires_api_base(self):
        config = load_config(None)
        with pytest.raises(ValueError, match="requires an explicit api_base"):
            apply_overrides(config, model="ollama/llama3")

    def test_yaml_template_loads(self):
        config = Config.from_yaml("configs/claude_cli.yaml")
        model = config.llm.models[0]
        assert model.provider == "claude_cli"
        assert model.api_base is None
        assert config.llm.temperature_emulation["enabled"] == "auto"

    @pytest.mark.parametrize(
        "key,value",
        [
            ("cli_binary", "/usr/bin/claude"),
            ("cli_extra_args", ["--verbose"]),
            ("max_budget_usd", 25),
            ("max_usage_limit_waits", 12),
            ("fallback_model", "haiku"),
        ],
    )
    def test_documented_per_model_options_load(self, key, value, tmp_path):
        """Regression: these are documented in configs/claude_cli.yaml and
        configs/README.md but were not fields on LLMModelConfig, so putting any
        of them in YAML raised TypeError before the run started."""
        import yaml

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {"llm": {"models": [{"name": "claude_cli/sonnet", "weight": 1.0, key: value}]}}
            )
        )
        config = Config.from_yaml(str(cfg_path))
        assert getattr(config.llm.models[0], key) == value

    def test_cli_options_reach_the_backend(self, monkeypatch):
        from skydiscover.llm.claude_cli import ClaudeCLILLM

        monkeypatch.setattr(shutil, "which", lambda _b: "/opt/claude")
        cfg = LLMModelConfig(
            name="claude_cli/sonnet",
            provider="claude_cli",
            cli_binary="/opt/claude",
            cli_extra_args=["--verbose"],
            max_budget_usd=25,
            max_usage_limit_waits=12,
            fallback_model="haiku",
        )
        backend = ClaudeCLILLM(cfg)
        assert backend.max_usage_limit_waits == 12
        cmd = backend._build_command(None)
        assert cmd[0] == "/opt/claude"
        assert cmd[cmd.index("--max-budget-usd") + 1] == "25"
        assert cmd[cmd.index("--fallback-model") + 1] == "haiku"
        assert cmd[-1] == "--verbose"


class TestBackendRouting:
    def test_detect_provider_from_field(self):
        assert _detect_provider(LLMModelConfig(name="x", provider="claude_cli")) == "claude_cli"

    def test_detect_provider_from_unstripped_name(self):
        assert _detect_provider(LLMModelConfig(name="claude_cli/sonnet")) == "claude_cli"

    def test_detect_provider_default_empty(self):
        assert _detect_provider(LLMModelConfig(name="gpt-5")) == ""

    def test_routes_to_claude_cli_backend(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _b: "/usr/bin/claude")
        backend = create_llm_backend(LLMModelConfig(name="sonnet", provider="claude_cli"))
        assert type(backend).__name__ == "ClaudeCLILLM"

    def test_routes_to_openai_backend(self):
        backend = create_llm_backend(LLMModelConfig(name="gpt-5", api_base="http://x", api_key="k"))
        assert type(backend).__name__ == "OpenAILLM"

    def test_init_client_hook_wins(self):
        sentinel = object()
        cfg = LLMModelConfig(name="whatever", init_client=lambda _c: sentinel)
        assert create_llm_backend(cfg) is sentinel
