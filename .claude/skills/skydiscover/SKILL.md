---
name: skydiscover
description: Working in the SkyDiscover repository — running discovery experiments, adding a search strategy / benchmark / evaluator / LLM backend, debugging a run, or answering how the framework works. Use this instead of exploring the repo from scratch.
---

# SkyDiscover

An LLM-driven algorithmic-discovery framework: you supply a scoring function and
(optionally) a seed program; an LLM rewrites the program, an evaluator scores it,
and a pluggable **search strategy** decides what to mutate next. Berkeley Sky
Computing, CAIS '26 (`README.md` citation block). ~27k LOC in `skydiscover/`,
~200 benchmark tasks under `benchmarks/`.

The architectural bet: *"what to mutate next"* is factored out of *"how to
prompt, generate, and evaluate."* That is why competitor algorithms
(OpenEvolve, GEPA) are reimplemented in-tree — so they can be A/B'd under
identical prompting machinery.

## Read this first

**`ProgramDatabase` has exactly two abstract methods — `add` and `sample`**
(`skydiscover/search/base_database.py`). That is the whole search extension
surface. `TopKDatabase` implements the full contract in 83 lines and is the
best file to read before writing anything.

**`get_statistics()` on the database is not telemetry — it is prompt input.**
Score tiers, execution traces, and parent-reuse ratios are fed back to the LLM.

## The core loop

Each entry point opens one asyncio event loop (`asyncio.run` in `cli.py::main`
and `api.py::run_discovery`). **EvoX is the exception**: it calls `asyncio.run`
again on a worker thread (`context_builder/evox/builder.py::run_async_safely`,
`search/evox/utils/variation_operator_generator.py`), which blocks the main loop
while a nested loop runs — see `references/gotchas.md`.

```
cli.py / api.py
  └─ Runner.__init__          load config, build output dir, create database
  └─ Runner.run
       └─ get_discovery_controller()        search/route.py — the entire dispatch rule
            └─ DiscoveryController.run_discovery
                 └─ _run_iteration(i)       ← the loop body
                      1. database.sample(n)            → (parent, context)
                      2. _build_prompt()               → {"system", "user"}
                      3. [human feedback]              → mutates prompt["system"]
                      4. _call_llm()                   → agentic or plain
                      5. _parse_llm_response()         → diff apply or full rewrite
                      6. evaluator.evaluate_program()  → EvaluationResult
                      7. failure screen                → retry feeds errors back in
                      8. _process_iteration_result()   → database.add(), checkpoint
       └─ teardown: final checkpoint → mode="test" re-eval → write best/
```

Two non-obvious details that explain a lot of the code:

- **`sample()` may return a dict-wrapped parent, and the dict key is a prompt
  payload, not a label.** It is injected verbatim under `# Current Solution`.
  AdaEvolve uses it for explore/exploit guidance; EvoX uses it so an evolved
  database can pick a mutation operator by choosing a dict key.
- **A failed iteration is not discarded** — the error text is re-injected as
  `context["errors"]` on the next attempt, making the retry loop a self-repair
  loop.

Scoring is uniform: `utils/metrics.py::get_score` returns `combined_score` if
present, else the mean of numeric metrics — with `bool` deliberately excluded so
`timeout: True` cannot count as fitness.

## Running it

```bash
uv sync                                    # base install
uv sync --extra math                       # per-benchmark extras (see below)

uv run skydiscover-run <initial_program> <evaluator> \
  -c config.yaml -s adaevolve -m gpt-5 -i 100 -o outputs/run1
```

- `<initial_program>` is **optional** (`nargs="?"`) — omit it to start from scratch.
- `<evaluator>` is a `.py` file, a containerized `evaluator/` **directory**, or a
  Harbor task directory. Auto-detected.
- Flags: `-c/--config -o/--output -i/--iterations -m/--model -s/--search
  -l/--log-level --api-base --checkpoint --agentic`.

Python API: `run_discovery(evaluator, initial_program=None, ...) -> DiscoveryResult`
— **note `evaluator` is first and required**. `discover_solution()` is a thin
wrapper accepting inline strings and callables.

**Using a Claude subscription instead of an API key:**

```bash
claude auth login                                    # once (bare `claude auth` only prints help)
uv run skydiscover-run prog.py evaluator.py -c configs/claude_cli.yaml
# or:  -m claude_cli/sonnet
```

Outputs land in `outputs/<search_type>/<problem>_<MMDD_HHMM>/`:
`logs/`, `checkpoints/checkpoint_<n>/`, `best/best_program*`. Resume with
`--checkpoint <dir>` — the only *CLI* path (`run_discovery()` has no checkpoint
parameter). A second, implicit path exists: setting `search.database.db_path`
makes `ProgramDatabase.__init__` load it. The two disagree on the start
iteration — see `references/gotchas.md`.

## Where to look

| Task | Read |
|---|---|
| Understand the whole system | `references/architecture.md` |
| Add or pick a search strategy | `references/search-strategies.md` |
| Add or run a benchmark | `references/benchmarks.md` |
| Write an evaluator | `references/evaluation.md` |
| Any config question | `references/configuration.md` |
| LLM backends, Claude CLI, rate limits, temperature | `references/llm-backends.md` |
| Something is behaving oddly | `references/gotchas.md` |

## Newcomer reading order

1. `skydiscover/search/README.md` — the authored extension contract.
2. `skydiscover/search/base_database.py` — `Program` + the two-method ABC.
3. `skydiscover/search/default_discovery_controller.py` — `_run_iteration`,
   `_build_prompt`, `_parse_llm_response`, `_process_iteration_result`.
   Everything else in the framework is a variation on these four.
4. `skydiscover/search/topk/database.py` — the whole Level-1 contract, 83 lines.
5. `skydiscover/config.py` — `_PROVIDERS`, `LLMConfig.__post_init__`,
   `_DB_CONFIG_BY_TYPE`, `Config.from_dict`, `apply_overrides`. Nearly every
   "why didn't my setting take effect?" is answered in those five places.
6. `skydiscover/search/adaevolve/adaptation.py` — 569 lines of pure math, no
   I/O; the clearest expression of the project's central idea.
7. `tests/test_smoke.py` — the only end-to-end test, and the fastest way to see
   how to stub the LLM.

## Extension points at a glance

| Axis | Interface | Register via |
|---|---|---|
| Search strategy | `ProgramDatabase.add` / `.sample`; optionally subclass `DiscoveryController` | `register_database` / `register_controller` in `search/route.py` |
| Context builder | `ContextBuilder.build_prompt(current_program, context, **kw) -> {"system","user"}` | `config.context_builder.template`, or hardcode in a controller `__init__` |
| Evaluator | duck-typed `evaluate_program(solution, program_id, mode)` | auto-detected by `evaluation/__init__.py::create_evaluator` |
| LLM backend | `LLMInterface.generate` | provider prefix → `llm/llm_pool.py::create_llm_backend` |
| Benchmark resolver | `BenchmarkResolver.resolve()` + module-level `resolver` | `config.benchmark.resolver` |

## Hard-won facts

- **CI** (`.github/workflows/ci.yml`) runs black + isort **scoped to
  `skydiscover/` only**, then `pytest tests/`. `mypy` is configured strictly in
  `pyproject.toml` and **never run**. Run `uv run black skydiscover/ && uv run
  isort skydiscover/` before pushing.
- **Tests:** `uv sync --extra dev` first — plain `uv sync` does **not** install
  pytest. Then `uv run python -m pytest tests/ -q -m "not integration"`.
  Integration tests hit real services and consume quota, and **nothing deselects
  them automatically** — `addopts` is only `--strict-markers` and there is no
  `conftest.py`, so you must pass `-m "not integration"` yourself. CI does not.
- There is **no cost model or budget cap** for API backends; `response.usage` is
  never read. The Claude CLI backend is the exception — it records cost and
  tokens (`llm/claude_cli.py::GLOBAL_COST_TRACKER`).
- **No sandboxing on evaluation.** Containerized evaluation is
  `docker run -d --rm --entrypoint sleep <tag> infinity` — no `--memory`,
  `--cpus`, `--network`, or `--gpus` anywhere in the repo. LLM-generated code
  runs with full default container privileges and unrestricted network.
