# Architecture

Subsystem map for `skydiscover/`. Symbols are cited rather than line numbers
(line numbers drift; symbol names do not).

## Entry points — `cli.py`, `api.py`, `runner.py`

Two front doors converge on `Runner`.

- **CLI** `skydiscover-run` (`pyproject.toml [project.scripts]`) → `cli.py::main`.
  Positional `initial_program` (`nargs="?"`, genuinely optional) + `evaluation_file`.
- **API** `api.py::run_discovery(evaluator, initial_program=None, ...)`.
  Materializes inline strings and Python callables into files via `utils/prepare.py`.
  `api.py::discover_solution` is a convenience wrapper.

Both do: load config → `apply_overrides` → optional `resolve_benchmark_problem`
→ branch to an external backend if `is_external(search.type)` → else build a `Runner`.

`Runner.__init__` loads config, builds the output dir
(`config.py::build_output_dir` → `outputs/<search_type>/<problem>_<MMDD_HHMM>`),
reads the seed, infers `config.language` via
`utils/code_utils.py::extract_solution_language`, and creates the database via
`search/registry.py::create_database`.

`Runner.run` builds a `DiscoveryControllerInput` and calls
`search/route.py::get_discovery_controller`.

### Teardown

Final checkpoint → re-evaluate the best program with `mode="test"` and merge the
result as `test_*` keys into a **deep copy** (not the live database object, so
the checkpoint and the reported score stay consistent) → `controller.close()` →
stop monitor → write `outputs/…/best/`.

## Configuration — `config.py`

Single source of truth; 18 dataclasses. See `references/configuration.md`.

- `_PROVIDERS` maps provider → (default base URL, env var list).
- `_BARE_PREFIX_MAP` routes bare model names by prefix.
- `_DB_CONFIG_BY_TYPE` maps `search.type` → the DatabaseConfig subclass.
- `Config.from_yaml` expands `${VAR}` and treats a short single-line
  `prompt.system_message` as a **file path** relative to the config dir.
- `Config.from_dict` maps YAML sections onto dataclasses. `prompt:` maps to
  `config.context_builder`.

## Search core — `skydiscover/search/`

**`base_database.py`**
- `Program` dataclass: `id`, `solution`, `language`, `metrics`, `iteration_found`,
  `parent_id`, `other_context_ids`, `parent_info`, `context_info`, `timestamp`,
  `metadata`, `artifacts`, `prompts`, `generation`. Plus `to_dict` / `from_dict`
  (`from_dict` silently drops unknown keys with only a `logger.debug`).
- Useful concrete helpers you should call rather than reimplement:
  `_update_best_program`, `_is_better`, `get_best_program`, `get_top_programs`,
  `get`, `_save_program`, `log_prompt`, `log_status`, `get_statistics`.
- `ProgramDatabase` ABC — exactly two abstract methods:
  - `add(program, iteration=None, **kwargs) -> str`
  - `sample(num_context_programs) -> (parent, context)`
  Plus concrete save/load, best tracking, top-N, prompt logging, and
  `get_statistics()` (score tiers, execution trace, score trajectory, parent and
  context reuse ratios) — **which is prompt input, not telemetry**.

**`default_discovery_controller.py`** — `DiscoveryController` owns the async loop.
- `__init__` builds three `LLMPool`s (`models` / `evaluator_models` /
  `guide_models`), picks the context builder, **mutates the shared Config in
  place** (`config.evaluator.evaluation_file` etc.), and calls `create_evaluator`
  — which for a containerized benchmark runs `docker build` + `docker run` right
  there in the constructor.
- `run_discovery` dispatches on `config.max_parallel_iterations`
  (1 → `_run_discovery_sequential`, >1 → `_run_discovery_parallel`).
- Key methods: `_run_iteration`, `_build_prompt`, `_parse_llm_response`,
  `_create_child_program`, `_process_iteration_result`.

**`registry.py`** holds three dicts + factories; **`route.py` is the only module
that populates them**, at import time. In practice they are always populated:
importing anything under `skydiscover` runs `skydiscover/__init__.py` →
`runner.py`, which imports `search/route.py`. `create_database` raises
`ValueError` (listing available types) only for a genuinely unregistered type.

**`utils/`** — `checkpoint_manager.py`, `discovery_utils.py` (`SerializableResult`),
`logging_utils.py`.

## Context / prompt construction — `skydiscover/context_builder/`

- `base.py::ContextBuilder` — a ~40-line ABC with one method,
  `build_prompt(current_program, context, **kwargs) -> {"system", "user"}`.
  `self.config` is the **top-level `Config`**; `self.context_config` is
  `config.context_builder`. (Confusing these two was a real bug — see gotchas.)
- `utils.py::TemplateManager` layers `*.txt` template directories so later dirs
  override earlier by filename stem.
- `default/builder.py::DefaultContextBuilder` — six section formatters.
  `_select_template_key` picks: no parent → `from_scratch_user_message`;
  `language == "image"` → `image_user_message`; `diff_based_generation` →
  `diff_user_message`; text/prompt language → `full_rewrite_prompt_opt_user_message`;
  else `full_rewrite_user_message`.
- Templates are raw `str.format` strings — **no template engine**, so literal
  braces must be doubled.
- Ships: `DefaultContextBuilder`, `AdaEvolveContextBuilder` (adds
  `{search_guidance}` + Pareto wording), `GEPANativeContextBuilder` (adds
  reflective rejection history), `EvoxContextBuilder` (replaces `build_prompt`
  entirely; fires guide-LLM calls).
- `skydiscover/prompt/__init__.py` is a re-export shim for the legacy name.

## LLM — `skydiscover/llm/`

See `references/llm-backends.md` for detail.

- `base.py::LLMInterface` — one abstract coroutine returning
  `LLMResponse(text, image_path)`. Class attribute `supports_native_agentic`.
- `openai.py::OpenAILLM` — reaches every HTTP provider (Anthropic, Gemini,
  DeepSeek, Mistral, Ollama, vLLM) through OpenAI-compatible base URLs.
- `claude_cli.py::ClaudeCLILLM` — drives the local `claude` binary
  (subscription auth, no API key, no Docker).
- `llm_pool.py::LLMPool` — weighted sampling; `create_llm_backend` dispatches on
  provider. `generate_all()` fans out to all models (sole caller: `LLMJudge`).
- `rate_limit.py` — usage-limit detection + process-wide wait-until-reset gate.
- `temperature.py` — temperature emulation for models with no sampling knob.
- `agentic_generator.py::AgenticGenerator` — bounded ReAct loop with two
  sandboxed tools (`read_file`, `search`); delegates to backends that advertise
  `supports_native_agentic`.

## Evaluation — `skydiscover/evaluation/`

Three duck-typed backends behind `__init__.py::create_evaluator`, detected
most-specific-first. See `references/evaluation.md`.

Every failure mode returns a sentinel metric dict instead of raising, so a bad
generated program can never kill the loop.

## Extras — `skydiscover/extras/`

**`monitor/`** — a dependency-free single-port HTTP + WebSocket server
(hand-rolled RFC 6455 framing) on a daemon thread with its own event loop,
streaming every evaluated program to a Plotly dashboard (`dashboard.html`).
Full solution text is never broadcast — it is fetched lazily per selection.
`viewer.py::main` (`skydiscover-viewer`) replays a finished checkpoint dir
through the same server.

**`external/`** — adapters wrapping upstream packages as drop-in strategies
(`openevolve_backend.py`, `gepa_backend.py`, `shinkaevolve_backend.py`), sharing
one signature:

```python
async def run(program_path, evaluator_path, config_obj, iterations,
              output_dir, monitor_callback=None, feedback_reader=None) -> DiscoveryResult
```

Registration fails soft on `ImportError`; `KNOWN_EXTERNAL` turns "not installed"
into a precise `pip install` message.

## Benchmarks

`benchmarks/` holds ~200 self-describing task directories (seed + `config.yaml`
+ evaluator). `skydiscover/benchmarks/` (~90 lines) is separate: a
`BenchmarkResolver` ABC and `resolution.py::resolve_benchmark_problem`, which
dynamically imports a resolver module and calls its module-level `resolver`
object — enabling config-only problem selection (used by KernelBench).

## Concurrency model

Everything is one asyncio loop. **Every blocking call in the hot path** — both
LLM paths and both evaluator backends — uses `run_in_executor(None, …)`, i.e.
the *same* default `ThreadPoolExecutor(min(32, cpu+4))`. There is no
`set_default_executor` anywhere. `asyncio.wait_for` cannot cancel a running
thread. The Claude CLI backend is the exception: it uses
`asyncio.create_subprocess_exec`, which is genuinely async.
