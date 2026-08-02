# Configuration

Everything lives in `skydiscover/config.py`. Annotated templates are in
`configs/` (see `configs/README.md`).

## Layering, in order

1. **Dataclass defaults** (`config.py`).
2. **YAML file** — `Config.from_yaml`: expands `${VAR}`, then treats a short
   single-line `prompt.system_message` as a **file path relative to the config
   file's directory** and replaces it with the file's contents.
3. **Environment** — `load_config()`: `OPENAI_API_BASE` / `OPENAI_BASE_URL`
   (with `overwrite=True`), provider-aware key resolution, pushes
   `context_builder.system_message` into every model config, then
   `bridge_provider_env()` `os.environ.setdefault`s resolved keys so external
   backends (ShinkaEvolve etc.) can find them.
4. **CLI / API flags** — `apply_overrides()`.

⚠️ `load_dotenv` is **never called in the library run path**
(`cli.py` / `api.py` / `runner.py`) — only in `extras/monitor/viewer.py` and in
`benchmarks/frontier-cs-eval/run_all_frontiercs.py`. A `.env` file will **not**
be picked up by an ordinary `skydiscover-run`; export the vars yourself.

## Deserialization is inconsistent — this matters

`Config.from_dict` treats sections differently:

| Section | Behaviour on an unknown key |
|---|---|
| Top-level keys | applied only `if hasattr(config, key)` → **silently dropped** |
| `search.database`, `benchmark` | known fields split out; extras `setattr`'d / funnelled into `params` → **works**, so a strategy can take knobs before it has a typed config class |
| `llm`, `prompt`, `evaluator`, `agentic`, `monitor`, `search` | splatted with `**dict` → **any typo raises `TypeError`** |

So a misspelled top-level key fails silently, while a misspelled `llm` key
crashes at startup. `random_seed: 42` appears at the top level of four shipped
configs and is silently dropped — `Config` has no such field.

## Top-level keys

| Key | Default | Effect |
|---|---|---|
| `max_iterations` | 100 | CLI `-i` overrides |
| `checkpoint_interval` | **1** | `checkpoints/checkpoint_<n>/` cadence. Defaults to every iteration so an interrupted run loses nothing; each checkpoint is a full snapshot, so raise it if disk matters |
| `log_level` | `"INFO"` | |
| `log_dir` | `None` | defaults under the output dir |
| `language` | inferred from seed | `"image"` and `{text, prompt, text/plain}` change template selection, evaluator input, and AdaEvolve label sets |
| `file_suffix` | `".py"` | auto-set from the seed's extension |
| `seed_programs_dir` | `None` | directory of **extra** seed programs, evaluated and added before iteration 1. Relative paths resolve against the primary seed's directory. Only files matching the primary's extension are read; whitespace-identical duplicates (including a copy of the primary) are dropped. On an island database they are spread round-robin over islands 1..N-1, leaving island 0 to the primary. `None` = historical single-seed behaviour |
| `max_seed_programs` | 16 | cap on the pool — each seed costs one evaluator run *before* the loop starts, with no monitor yet, so an uncapped directory looks like a hang |
| `diff_based_generation` | `true` | **the single most behaviour-defining flag** — SEARCH/REPLACE diffs vs. full rewrite |
| `max_solution_length` | 60000 | over-length becomes a *parse error*, not a truncation |
| `max_parallel_iterations` | 1 | **only the base controller reads it** — inert for adaevolve / evox / gepa_native / claude_code. Documented nowhere else. |
| `human_feedback_enabled` | **`true`** | `human_feedback_mode: "replace"` overwrites `prompt["system"]` wholesale |
| `system_prompt_override` | `None` | runtime-only, set by `apply_overrides` |

## `llm`

| Key | Default | Notes |
|---|---|---|
| `models` | **`[{name: claude_cli/claude-opus-5}]`** | list of `{name, weight}`. The default runs on the local `claude` binary (`config.py::DEFAULT_MODEL`), so a run with no `-m` and no `claude` installed raises `RuntimeError` at pool construction. `-m <model>` replaces the list outright |
| `evaluator_models` / `guide_models` | `[]` | `__post_init__` copies from `models` when empty (**shallow copy — same objects**) |
| `temperature` | 0.7 | dropped automatically for models that reject it; see `llm-backends.md` |
| `top_p` | **`None`** | load-bearing: Anthropic/Bedrock reject temperature and top_p together. `None` params are omitted entirely |
| `max_tokens` | 32000 | |
| `timeout` / `retries` / `retry_delay` | 600 / 3 / 5 | worst case ≈ 4×600 + 3×5 ≈ 40 min per generation |
| `reasoning_effort` | `None` | |
| `temperature_emulation` | `{}` | see `llm-backends.md` |

Per-model extras (also settable per entry in `models`): `api_base`, `api_key`,
`provider`, `weight`, `max_usage_limit_waits`, and the Claude-CLI-only
`cli_binary`, `cli_extra_args`, `max_budget_usd`, `fallback_model`.

`LLMConfig.__post_init__` resolves each model's provider from the name prefix,
fills `api_base`/`api_key`, and strips the `provider/` prefix from the name —
**except `openai/`, which is sent verbatim** — and skips the whole resolution
block for any model that already has an explicit `api_base`. The CLI path
(`apply_overrides`) *does* strip `openai/`. Local providers (`claude_cli`) are
skipped entirely: no endpoint, no key.

## `prompt` → `config.context_builder`

The YAML section is `prompt:` but the dataclass is `ContextBuilderConfig`.

| Key | Default |
|---|---|
| `template` | `"default"` (or `"evox"`) |
| `template_dir` | `None` — extra template dir layered over the built-ins |
| `system_message` | `"system_message"` (a template key; a short single-line value is resolved as a **file path**) |
| `evaluator_system_message` | `"evaluator_system_message"` |
| `suggest_simplification_after_chars` | 500 |

## `evaluator`

| Key | Dataclass default | Note |
|---|---|---|
| `timeout` | **360** | but **10000** in `configs/default.yaml`, which effectively disables it |
| `max_retries` | 3 | |
| `cascade_evaluation` | **`true`** | `false` in most shipped configs; only `cascade_thresholds[0]` is read |
| `cascade_thresholds` | `[0.3, 0.6]` | second element is dead |
| `inject_evaluator_context` | `false` | prepends evaluator source / Harbor `instruction.md` as `# Task Description` |
| `llm_as_judge` | `false` | |
| `evaluation_file`, `file_suffix`, `is_image_mode` | set at runtime | mutated in place by `DiscoveryController.__init__` |

## `search`

| Key | Default |
|---|---|
| `type` | `"topk"` — selects the controller **and** the DatabaseConfig class via `_DB_CONFIG_BY_TYPE` |
| `database` | per-type config; unknown keys become untyped extras |
| `num_context_programs` | 4 |
| `switch_interval` | `None` (EvoX: stagnation iters before strategy switch; auto if None) |
| `share_llm` | `false` (EvoX: meta-level evolution reuses the main LLM config) |

## `agentic`

| Key | Default |
|---|---|
| `enabled` | `false` (CLI `--agentic`) |
| `codebase_root` | `None` → the seed program's directory |
| `max_steps` | 5 |
| `per_step_timeout` / `overall_timeout` | 60.0 / 300.0 |
| `max_context_chars` / `max_file_chars` / `max_search_results` | 400000 / 50000 / 50 |
| `max_files_read` | 20 |
| `regex_timeout` / `max_regex_length` | 2.0 / 200 |
| `repo_map_max_depth` | 4 |
| `allowed_extensions` / `excluded_dirs` | tuples used to build the repo map |

## `monitor`

| Key | Default |
|---|---|
| `enabled` | **`true`** |
| `port` / `host` | 8765 / `"127.0.0.1"` — `_serve` auto-increments the port up to 10 times if taken |
| `summary_model` | `"claude_cli/claude-sonnet-5"` — a `claude_cli/` model needs no API key; anything else is called over HTTP and does |
| `summary_interval` / `summary_top_k` | 0 (manual refresh only) / 3 |
| `max_solution_length` | 10000 — truncation for what the dashboard broadcasts |

⚠️ The monitor is **on by default**, so every run binds a port and starts a daemon
thread. Enabling it **also silently enables human-feedback file polling**,
regardless of `human_feedback_enabled` (`Runner._setup_human_feedback` gates on
`... or monitor_server`).

## Providers

`_PROVIDERS` in `config.py`:

| Provider | Default base URL | Env vars |
|---|---|---|
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `azure` | (same) | `AZURE_API_KEY`, `OPENAI_API_KEY` |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta/openai/` | `GEMINI_API_KEY`, `GOOGLE_API_KEY` |
| `anthropic` | `https://api.anthropic.com/v1/` | `ANTHROPIC_API_KEY` |
| `deepseek` / `mistral` / `cohere` | per-provider | per-provider |
| `huggingface` | **none** → `api_base` required | `HF_TOKEN`, `HUGGINGFACE_API_KEY` |
| `ollama` / `vllm` | **none** → `api_base` is required, else `apply_overrides` raises | — |
| `claude_cli` / `claude-cli` | **none needed** | none — auth comes from `claude auth` |

All key lookups fall back to `OPENAI_API_KEY`.

`_BARE_PREFIX_MAP` routes bare names: `gpt-`/`o1`/`o3`/`o4` → openai,
`gemini-` → gemini, `claude-` → anthropic, `deepseek-`, `mistral-`, `command-`.

## Config keys documented but broken

`configs/README.md` documents five keys that raise `TypeError`:
`evaluator.use_llm_feedback`, `evaluator.llm_feedback_weight`,
`llm.random_seed`, `llm.primary_model`, `llm.primary_model_weight`.
