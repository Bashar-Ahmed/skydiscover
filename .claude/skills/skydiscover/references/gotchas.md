# Gotchas and sharp edges

Things that will cost you an afternoon if you don't know them.

## Config

- **A misspelled top-level key is silently dropped** (`hasattr` guard); a
  misspelled `llm`/`prompt`/`evaluator`/`agentic`/`monitor`/`search` key raises
  `TypeError`. Unknown `search.database` keys are accepted as untyped extras by
  design.
- `evaluator.timeout` is **360** in the dataclass but **10000** in
  `configs/default.yaml` — the template effectively disables the eval timeout.
- `evaluator.cascade_evaluation` is **`true`** in the dataclass but `false` in
  most shipped configs. Only `cascade_thresholds[0]` is read.
- `max_parallel_iterations` is read **only by the base controller** — it is inert
  for adaevolve, evox, gepa_native, and claude_code. It is documented nowhere.
- `random_seed: 42` at the top level of four shipped configs does nothing.
- `configs/README.md` documents five keys that raise `TypeError`:
  `evaluator.use_llm_feedback`, `evaluator.llm_feedback_weight`,
  `llm.random_seed`, `llm.primary_model`, `llm.primary_model_weight`.
- Enabling `monitor` **also silently enables human-feedback file polling**.
- `load_dotenv` is never called in the run path — a `.env` will not be read.
- `evaluator_models` / `guide_models` default to a **shallow copy** of `models`,
  so they hold the *same objects*. Mutating one mutates all three.

## Generation and parsing

- **Diff application skips non-matching blocks.** `apply_diff` replaces the
  first exact line-list match and moves on. Use `apply_diff_detailed`, which
  returns `(result, applied, total)`, and reject `applied == 0` — otherwise the
  child is byte-identical to its parent and gets stored as a genuine candidate.
  Both controllers do this; **a custom controller must too.**
- `extract_diffs` uses a non-greedy regex: a literal `=======` inside the
  **SEARCH block** ends it early and folds the remainder into the replacement (a
  `=======` inside the replacement is preserved). Both halves are `rstrip()`ed,
  so meaningful trailing whitespace can never match.
- `parse_full_rewrite` is **language-aware**: for text/prompt/image languages an
  unfenced response is returned verbatim (prompt-optimization depends on this);
  for code languages it requires a fence or code-like content, else returns
  `None`.
- `max_solution_length` (60000) turns an over-long solution into a **parse
  error**, not a truncation.

## Search and resume

- **Controllers have no `save`/`load`.** Only the database is checkpointed. Any
  state on a controller is lost on resume — put resumable state on the database.
  `gepa_native` does this via `_controller_state_to_dict` / `_from_dict`; copy
  that pattern.
- **EvoX resume is structurally broken and nowhere declared supported.** A
  resumed run reloads the *seed* random sampler, pours checkpointed programs into
  it, and discards the evolved strategies, the meta-population, the scorer
  window, and the generated DIVERGE/REFINE labels. It runs without error.
- **Two disagreeing resume mechanisms:** explicit `--checkpoint` yields
  `start_iteration = last_iteration + 1`; the implicit load inside
  `ProgramDatabase.__init__` from `config.search.database.db_path` yields
  `last_iteration` with **no `+1`**.
- **The final checkpoint is written twice** whenever the last iteration lands on
  an interval boundary — i.e. *always*, with `checkpoint_interval: 1`. The
  interval trigger in `_process_iteration_result` fires for iteration N, then
  `Runner.run` unconditionally re-saves `final_iteration = discovery_start +
  max_iterations - 1` (`runner.py:187-189`). Same directory, same content,
  written twice. Harmless but doubles the largest snapshot's write cost.
- `CheckpointManager.load` always rebuilds with base `Program.from_dict`,
  ignoring `db._program_class`, and `from_dict` drops unknown keys with only a
  `logger.debug`. Saves are full snapshots — O(programs × checkpoints) disk.
- `api.py::run_discovery` has **no `checkpoint` parameter**, and `cleanup=True`
  (the default) deletes the temp output dir containing the checkpoints. Resume
  is CLI-only.
- **Resume is effectively untested.** Nothing in `tests/` touches
  `CheckpointManager.load`, `Runner._load_checkpoint`, or `db_path`. The one
  checkpoint test, `tests/cli/test_checkpoint_discovery.py`, covers only
  `cli.py::_find_latest_checkpoint` — the helper that picks the
  highest-numbered `checkpoint_<n>` dir — not the restore path itself.
- `registry.py::register_program` is never called; `_PROGRAM_REGISTRY` is
  permanently empty.

## Evaluation

- The eval-failure predicate is **duplicated** in
  `default_discovery_controller.py` and `adaevolve/controller.py`. Change both.
- `mode="test"` is only forwarded by `ContainerizedEvaluator`; the Python
  `Evaluator` ignores it, and every shipped math/ADRS `evaluate.sh` carries
  `# MODE ($2) accepted but ignored`. The "authoritative test score" is usually a
  bit-identical re-run.
- `HarborEvaluator` is **not concurrency-safe** (fixed solution path + fixed
  reward file in a shared container), despite the base docstring.
- The Python `Evaluator` has **no inner timeout** on the user's `evaluate()`, and
  `asyncio.wait_for` cannot cancel a running thread — a hung evaluation
  permanently consumes a thread-pool slot shared with LLM generation.
- `container_evaluator.py::_inject_file` / `_remove_file` call `subprocess.run`
  with **no timeout at all**.
- **No resource sandboxing on the evaluator**:
  `container_evaluator.py::_start_container` runs
  `docker run -d --rm [-e …] --entrypoint sleep <tag> infinity` — no `--memory`,
  `--cpus`, `--network`, `--pids-limit`, `--ulimit`, or `--gpus`. The candidate
  is piped in over stdin rather than bind-mounted.
- **The `claude_code` baseline is less contained, not more**:
  `claude_code/controller.py::_build_docker_cmd` bind-mounts a host workspace
  (`-v {workspace}:/workspace`), passes `--dangerously-skip-permissions`, and
  with a containerized evaluator additionally runs `--privileged` (docker-in-docker).

## Concurrency and teardown

- Every blocking hot-path call shares the **same default `ThreadPoolExecutor`**
  (`run_in_executor(None, …)`): both LLM paths and both evaluator backends.
- **Parallel-mode bookkeeping is subtly wrong.**
  `task.add_done_callback(pending.discard)` binds the set live at creation, but
  `done, pending = await asyncio.wait(...)` rebinds `pending` to a fresh set — so
  earlier callbacks mutate orphaned sets and the backpressure check is
  inaccurate. Checkpointing fires out of order; a failed iteration on a multiple
  of the interval skips its checkpoint entirely.
- **The event loop freezes during EvoX prompt construction.**
  `EvoxContextBuilder.build_prompt` is synchronous but needs guide-LLM calls, so
  it routes through `run_async_safely`, which does
  `executor.submit(asyncio.run, coro); future.result()` — **with no timeout**.
  Worst case with the shipped meta config: ~1215 s of frozen loop, ×3 controller
  retries. The calls *are* concurrent and the problem-context summary is
  sha256-memoized, so the typical cost is one round-trip.
- **Teardown is unowned.** `DiscoveryController.close()` closes only the
  evaluator, and no controller overrides it. **No `LLMPool`/`OpenAILLM` client is
  ever closed** — each holds an httpx connection pool, and a run creates 3N–8N of
  them (AdaEvolve orphans an extra N by reassigning `self.llms` after
  `super().__init__()` already built one). EvoX's second controller and its
  evaluator are never closed.
- Every hot-swapped evolved database leaves a permanent `custom_database_<md5>`
  entry in `sys.modules` whose backing temp file was unlinked — so its tracebacks
  have no source.
- `HumanFeedbackReader` is an **unlocked shared blackboard** between the
  monitor's daemon thread and the discovery thread. `_write_feedback` truncates
  then writes; a concurrent `read()` sees an empty file, the `if feedback:` guard
  skips application, and the guidance is dropped silently. The controller also
  reads the file **twice** per prompt build, so logged feedback can differ from
  applied feedback.

## Testing and CI

- Run: `uv sync --extra dev` (pytest lives in the `dev` extra; a plain `uv sync`
  leaves you with "No module named pytest"), then
  `uv run python -m pytest tests/ -q -m "not integration"`.
- CI is three jobs: `lint` (black + isort, **scoped to `skydiscover/` only** —
  `tests/`, `benchmarks/`, `examples/`, `scripts/` are unlinted), `test`, `build`
  (`uv build`, publishes nothing). Single Python 3.10, no matrix.
- **mypy is configured strictly in `pyproject.toml` and never run.**
- `tests/` has no `conftest.py`. The dominant idiom is
  `object.__new__(Evaluator)` to test pure logic without Docker.
  `tests/utils/test_code_utils.py` side-loads via `importlib` so it passes with
  zero deps installed.
- `tests/test_smoke.py` is the only end-to-end test: it monkeypatches a
  `FakeLLMPool` into `default_discovery_controller`, runs 2 iterations, and
  asserts `best_score >= 0.8`. ⚠️ If you change `LLMPool.__init__`'s signature,
  update `FakeLLMPool` too.
- **Untested areas:** either Responses-API path, parallel mode, the checkpoint
  load/restore path, external backends. Agentic mode has partial coverage in
  `tests/llm/test_agentic_limits.py` (native delegation, usage-limit pauses,
  per-step timeout).

## Doc drift

- README's *"Any LiteLLM-compatible model works"* is **false** for the native
  path — there is no litellm import in `skydiscover/`.
- `search/README.md` omits `claude_code` entirely, and its "Directory structure"
  section omits both `claude_code/` and `openevolve_native/` (the latter does
  appear in its registration example).
- `context_builder/README.md` claims the default templates include
  `{search_guidance}` — only the adaevolve/ and gepa_native/ templates have it
  (`str.format` silently ignores the unused kwarg).
- AdaEvolve/GEPA templates print `# Current Solution` above `{current_program}`,
  which itself emits `# Current Solution` — the rendered prompt contains the
  heading twice.
- `frontier-cs-eval/config.yaml` contains `{problem_statement}` /
  `{problem_constraints}` placeholders that nothing substitutes — the LLM
  receives literal braces for all 172 problems.
- The `.gitmodules` entry for `benchmarks/ale_bench/ALE-Bench` is **orphaned** —
  no gitlink in the index; `git submodule update --init` is a no-op.
- `LICENSE` says `Copyright [2025] [SkyDiscover Team]` while every commit is 2026.

## Dead code

- `register_program` / `_PROGRAM_REGISTRY` — never called.
- `ClaudeCodeConfig.max_turns` — never read (the controller uses `max_iterations`).
- `Evaluator.evaluate_batch` / `ContainerizedEvaluator.evaluate_batch` — **zero
  call sites**. `TaskPool` *is* constructed by both evaluators (`evaluator.py:49`,
  `container_evaluator.py:81`) but is only ever driven from inside those dead
  methods, so `max_parallel_iterations`' second effect (evaluator
  `max_concurrent`) does nothing. (Unrelated: the module-level `evaluate_batch`
  in `evox/database/search_strategy_evaluator.py` *is* live.)
- `cascade_thresholds[1]` — never read.
- AdaEvolve: `stagnation_threshold`, `stagnation_multi_child_count`,
  `sibling_context_limit`, `archive_size` are dead config keys;
  `seed_all_islands` has no caller; `force_exploration` is plumbed through three
  layers and never passed `True`.

## Known-fixed (do not "re-fix")

These were live bugs and have been repaired — if you see them described in older
notes, they are stale:

- `llm_judge.py` read `context_builder.config` instead of `.context_config`,
  making the judge silently always return `None`.
- The non-agentic Responses-API fallback called two deleted `self._` helpers.
- AdaEvolve had no zero-applied-diff guard.
- `parse_full_rewrite` returned raw prose as a program for code languages.
- Image-mode `mode="test"` re-evaluation passed the model's prose instead of the
  image path.
- `test_*` metrics were merged into the live database object after the final
  checkpoint, so checkpoint and reported score disagreed.
- The monitor's Code tab called `showTab('code')` while the handler tested
  `'solution'`.
- GEPA reset its merge budget and tried-pairs set on every resume.
- `scripts/reproduce/*.sh` hardcoded `evaluator.py`, used wrong ADRS dataset
  paths, and reported success unconditionally via a bare `wait`.
