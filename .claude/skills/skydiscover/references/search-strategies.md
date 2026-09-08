# Search strategies

## Comparison

| Strategy | Selection policy | Distinguishing idea | Key file |
|---|---|---|---|
| `topk` | parent = rank 1 by score, context = ranks 2..K+1; deterministic, **no population cap** | the control condition — any delta vs. another strategy is attributable purely to selection | `search/topk/database.py` (83 lines) |
| `best_of_n` | sticky parent reused for N adds (default 5), then re-commit to global argmax; context re-randomized each iteration | N independent mutation attempts from one fixed code state | `search/best_of_n/database.py` |
| `beam_search` | fixed-width beam pruned on every `add`; 4 strategies (`best`, `stochastic` softmax, `round_robin`, default `diversity_weighted`) | greedy max-min diverse pruning via 1 − Jaccard over char 3-grams; optional exponential depth penalty; own save/load with BFS depth reconstruction | `search/beam_search/database.py` |
| `adaevolve` ⭐ | per-island adaptive intensity + UCB island bandit + QD archive sampling | four closed feedback loops on observed improvement | `search/adaevolve/` |
| `evox` ⭐ | whatever the *currently evolved* `EvolvedProgramDatabase.sample()` does | a nested controller evolves the search algorithm's own **source code**, hot-swapped mid-run | `search/evox/controller.py` |
| `gepa_native` | `epsilon_greedy` (default) / `best` / `pareto` over an elite pool | strict acceptance gating + LLM-mediated merge | `search/gepa_native/` |
| `openevolve_native` | per-island MAP-Elites: exploration / exploitation / uniform random | faithful port of codelion/openevolve | `search/openevolve_native/database.py` |
| `claude_code` | none — `sample()` returns `(best, [])` and the controller never calls it | not a search algorithm: shells the `claude` CLI into Docker as a single-agent baseline | `search/claude_code/controller.py` |
| `openevolve` / `gepa` / `shinkaevolve` | delegated upstream | external-backend adapters | `extras/external/*_backend.py` |

Registration is in `search/route.py` (bottom of file, `AUTO-REGISTRATION`).

Notes:
- `evox` has **no `_DATABASE_REGISTRY` entry** — `create_database` intercepts it
  and dynamically imports `EvolvedProgramDatabase` from
  `config.search.database.database_file_path`, which `EvoxDatabaseConfig.__post_init__`
  always fills with `search/evox/database/initial_search_strategy.py`.
- `claude_code` is registered and CLI-accessible (`--search claude_code`, listed
  in `cli.py::_SEARCH_CHOICES`) and is mentioned once in the README, but has no
  dedicated docs page and no config in `configs/`.
  It is **distinct from the `claude_cli/` LLM provider** —
  `claude_code` is a *search type* running the Claude Code agent in Docker;
  `claude_cli/` is an *LLM backend* usable by every strategy.

## AdaEvolve (the flagship, ~6.2k LOC)

Three levels of adaptation, all ablatable from config alone.

**Intensity** (`adaevolve/adaptation.py`) — each island tracks
`G_t = ρ·G_{t-1} + (1−ρ)·δ²` where δ is a *locally*-normalized improvement.
Intensity is `I_min + (I_max−I_min)/(1+√(G+ε))`: high G → exploit, decayed G → explore.

**Island scheduling** — UCB over decayed rewards using *globally*-normalized
deltas, with decayed visits in the denominator so `reward_avg` doesn't drift to
zero. Warm-up picks *randomly* among under-visited islands.

**Paradigms** (`adaevolve/paradigm/{tracker,generator}.py`) — when the global
improvement rate over a sliding window drops below threshold, the **guide LLM
pool** (deliberately separate from the mutation pool) is asked via a strict
`json_schema` for N code-free breakthrough ideas. These are injected as a
mandatory `## BREAKTHROUGH IDEA - IMPLEMENT THIS` block and rotate round-robin
until exhausted. The built-in idea guidance is library-centric (written for
algorithmic benchmarks: "single-function library call", `library.function`
approach types); `search.database.paradigm_domain_brief` replaces it with
domain-supplied text while keeping the JSON contract — set it whenever
"pick a library function" is not what a breakthrough means in your domain.
A top-level `paradigm_agentic:` block (same shape as
`agentic:`) makes those guide-pool calls run the backend's **native agent
loop** — read-only file tools rooted at its `codebase_root` and, on the CLI
backends, web search — independently of whether solution generation is
agentic. Honoured only when every pooled guide backend has
`supports_native_agentic`; otherwise the generator warns once and calls plainly.
The Codex CLI needs `cli_extra_args: ["--config", "tools.web_search=true"]` on
the model for web search; the Claude CLI's agentic tool list already includes
WebSearch/WebFetch. `llm_calls.jsonl` records `agentic: true` on such calls.

**Storage** — a flat quality-diversity `UnifiedArchive` per island with
deterministic-crowding eviction. **Explicitly not MAP-Elites** (that's
`openevolve_native`).

**Multi-objective** — activates by setting `search.database.pareto_objectives`
non-empty. The adaptation math still runs on a scalar proxy
(`utils/metrics.py::compute_proxy_score`) while selection and reporting go Pareto.

Config knobs: `AdaEvolveDatabaseConfig` in `config.py` (~40 fields).
Dead keys that look real but are never read: `stagnation_threshold`,
`stagnation_multi_child_count`, `sibling_context_limit`, `archive_size`.

## EvoX (co-evolution)

Two nested populations. The inner (solution) database is an **LLM-written Python
class**; the outer meta-population is `SearchStrategyDatabase`, whose
`Program.solution` *is the source of a database class*.

- `evox/database/search_strategy_evaluator.py` is a structural validator: class
  name, inheritance, signatures, metric immutability (1e-10 tolerance),
  `sample()` shape, and a purpose-built regression test that detects
  `isinstance(p, EvolvedProgram)` filtering bugs — emitting a fix-it message
  engineered to be read by the next generation's LLM.
- Meta-fitness: `improvement · (1+log(1+start)) / √horizon`
  (`evox/utils/search_scorer.py`).
- Before the loop, one guide-LLM call generates problem-specific
  `DIVERGE_LABEL` / `REFINE_LABEL` prompt fragments — grounded in the actual
  installed package list read from `requirements.txt` / `pyproject.toml` /
  `uv pip list` — stamped onto the database as attributes so an evolved
  `sample()` can pick a *prompt-level mutation operator* by choosing a dict key.
- `evox/config/evox_search_sys_prompt.txt` is the contract it enforces.

⚠️ **EvoX is the most dangerous code in the repo**: LLM-authored Python is
`exec`'d in-process with no sandbox and hot-swapped mid-run.

## Adding a search strategy

**Level 1 — database only** (most cases):

```python
# skydiscover/search/mystrategy/database.py
from skydiscover.search.base_database import Program, ProgramDatabase

class MyDatabase(ProgramDatabase):
    def add(self, program: Program, iteration=None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = iteration
        self._update_best_program(program)     # ← required by convention
        if self.config.db_path:
            self._save_program(program)
        return program.id

    def sample(self, num_context_programs: int):
        parent = self.get_best_program()
        context = self.get_top_programs(num_context_programs)
        return parent, context
```

The `add` contract is **convention, not enforcement**: store into
`self.programs`, bump `last_iteration`, call `_save_program` when `db_path` is
set, and **call `_update_best_program`** (forgetting this silently breaks best
tracking).

Register in `search/route.py`:

```python
from skydiscover.search.mystrategy.database import MyDatabase
register_database("mystrategy", MyDatabase)
```

Optionally add a `MyDatabaseConfig(DatabaseConfig)` in `config.py` and an entry
in `_DB_CONFIG_BY_TYPE` so YAML keys are typed. Unknown keys under
`search.database` are `setattr`'d as untyped extras either way, so a strategy
can take knobs before it has a typed config class.
(AdaEvolve does **not** rely on this — its ~38 knobs are declared in
`AdaEvolveDatabaseConfig`. A stale comment in `config.py` says otherwise.)

**Level 2 — custom controller**: subclass `DiscoveryController` and override
`run_discovery`, reusing `_run_iteration()` and `_process_iteration_result()` as
primitives. Register with `register_controller("mystrategy", MyController)`.

⚠️ If you override the controller and re-implement generation, **you must
replicate the diff-application and eval-failure guards** — they are duplicated,
not shared, between `default_discovery_controller.py` and
`adaevolve/controller.py`. Prefer calling the base `_run_iteration`.

⚠️ **Controllers have no `save`/`load`.** Any state you keep on the controller
is lost on `--checkpoint` resume. Put resumable state on the *database* and
persist it in the database's `save`/`load` (see `gepa_native/database.py::_controller_state_to_dict`
for the pattern).

**Dead axis:** `registry.py::register_program` is never called anywhere, so
`_PROGRAM_REGISTRY` is permanently empty and `get_program` always falls back to
base `Program` (except on the EvoX dynamic-file path).
