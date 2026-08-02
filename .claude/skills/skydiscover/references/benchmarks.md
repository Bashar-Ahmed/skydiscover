# Benchmarks

## Suites

| Suite | Domain | What's optimized | Eval mechanism | Infra |
|---|---|---|---|---|
| `benchmarks/math/` (14) | AlphaEvolve Appendix A/B constants: circle/hexagon packing, autocorrelation inequalities, Erdős min-overlap, Heilbronn, matmul tensor rank, signal processing | a Python function returning a construction | containerized; independently recomputes the objective in numpy and rejects mismatches (>1e-6, or exact `array_equal` for matmul). most tasks score `combined_score = BENCHMARK/achieved` or its inverse, so **> 1.0 means a new record**. Two exceptions: `signal_processing` returns a weighted 0–1 composite, and the *containerized* `circle_packing` evaluator returns raw `sum_radii` | Docker |
| `benchmarks/ADRS/` (5) | Berkeley systems research: `cloudcast` (multi-cloud broadcast), `eplb` (MoE expert load balancing), `prism` (GPU model placement), `llm_sql` (prefix-cache column ordering), `txn_scheduling` | one named function or class per task | containerized; simulator-based, with hard anti-reward-hacking validation. **Absolute scores, no in-loop baseline comparison** | Docker + dataset download |
| `benchmarks/gpu_mode/` (4) | Triton/CUDA kernels: `vecadd`, `grayscale`, `trimul` (AlphaFold3), `mla_decode` (DeepSeek) | `custom_kernel(data)` speed subject to correctness | plain Python `shared_eval.py`; adaptive timing loop with early exit on standard error; `combined_score = 3000 / geom_mean_us` | CUDA GPU (`mla_decode` needs H200/141 GB) or Modal (**`modal` is not a declared dep**) |
| `benchmarks/kernelbench/` (250+) | GPU kernels from HF `ScalingIntelligence/KernelBench` | a `ModelNew` nn.Module | resolver-driven; shells to KernelBench's `run_and_check.py` (pinned commit) and **regex-scrapes stdout**. Failures −100.0, timeouts −1.0 | Docker *or* native GPU host — ⚠️ shipped default has **no GPU passthrough** |
| `benchmarks/frontier-cs-eval/` (172) | competitive-programming C++ | one shared `initial_program.cpp`; problem chosen by `FRONTIER_CS_PROBLEM` env var | thin client posting to a Docker judge server | `git clone` Frontier-CS + `docker compose up -d` |
| `benchmarks/arc_benchmark/` | ARC-AGI-2 grid reasoning | **two** independent transforms `transform_grid_attempt_1/2` | containerized; `0.6·pass@2 + 0.4·best cell accuracy` on train grids | cloned ARC-AGI-2 repo + data-prep + per-task config generation |
| `benchmarks/ale_bench/` (10) | AtCoder Heuristic Contest C++ | `initial_program.cpp` | delegates to the external `ale_bench` package (compiles cpp20, runs 50 cases) | `ale_bench` + `ale_bench_eval` (manual install); the declared git submodule is **orphaned** |
| `benchmarks/prompt_optimization/hotpot_qa` | prompt evolution (**not code**) | a plain `.txt` prompt, no EVOLVE-BLOCK markers | host-side DSPy + BM25 over 300 HotPotQA examples; **~300 LLM calls per evaluation** | `--extra prompt-optimization`; ~1.3 GB auto-download |
| `benchmarks/image_gen/sky_festival` | VLM image generation | a PNG — **no seed file at all** | GPT-5 vision judge against a 100-point 7-category rubric | OpenAI vision access |

## Dependency extras

```bash
uv sync                              # base
uv sync --extra math                 # SciPy, JAX, PyWavelets, …
uv sync --extra adrs
uv sync --extra frontier-cs
uv sync --extra external             # OpenEvolve / GEPA backends
uv sync --extra prompt-optimization
uv sync --extra dev                  # pytest, black, isort, mypy
```

Combine freely. If a benchmark ships its own `requirements.txt`, also run
`uv pip install -r <path>`.

Not covered by any extra: `shinka` (manual install), `harbor`, `modal`.

## Anatomy of a task directory

```
benchmarks/<suite>/<task>/
├── initial_program.py        # or .cpp / initial_prompt.txt — OPTIONAL
├── config.yaml               # per-task config; system_message states the record to beat
└── evaluator.py              # plain Python evaluator
    └── OR evaluator/         # containerized: Dockerfile + evaluate.sh (+ data, download_dataset.sh)
```

Some tasks nest a size parameter (`hexagon_packing/11`, `minimizing_max_min_dist/2`).
Exactly one task ships **both** `evaluator.py` and `evaluator/` —
`benchmarks/math/circle_packing` — and the two are non-equivalent, on different
score scales. The reproduce scripts prefer `evaluator.py` when both exist.

## EVOLVE-BLOCK markers

⚠️ **The core framework does not parse or enforce these.** The whole seed file is
shown to the LLM, and a full rewrite replaces the whole file. In `skydiscover/`
the markers are only *written* by `utils/prepare.py` (wrapping an inline string)
and only *split on* by `extras/external/gepa_backend.py`. The README's "everything
outside is left untouched" describes a convention the native path does not
guarantee — if you need a frozen region, assert it in the evaluator.

By convention the seed marks the mutable region:

```python
# EVOLVE-BLOCK-START
def solve(input_data):
    return input_data
# EVOLVE-BLOCK-END
```

If no markers are present, the **entire file** is treated as mutable.

## Adding a benchmark

1. Create `benchmarks/<suite>/<task>/`.
2. Write the evaluator (see `references/evaluation.md`) — return
   `{"combined_score": ...}` plus any extra numeric metrics.
3. Write `initial_program.py` with EVOLVE-BLOCK markers (optional — omit to have
   the LLM start from scratch).
4. Write `config.yaml`. Put the target/record in `prompt.system_message` so the
   model knows what to beat.
5. Run:
   ```bash
   uv run skydiscover-run benchmarks/<suite>/<task>/initial_program.py \
     benchmarks/<suite>/<task>/evaluator.py \
     -c benchmarks/<suite>/<task>/config.yaml -s adaevolve -i 100
   ```

See `benchmarks/README.md` for the user-facing JSON protocol.

## Benchmark resolvers (config-only problem selection)

For suites where one task directory covers many problems (KernelBench):

```python
# benchmarks/<suite>/resolver.py
from pathlib import Path
from typing import Any, Dict

from skydiscover.benchmarks.base import BenchmarkResolver
from skydiscover.benchmarks.resolution import BenchmarkResolution

class MyResolver(BenchmarkResolver):
    def resolve(self, config: Dict[str, Any], output_dir: Path) -> BenchmarkResolution:
        return BenchmarkResolution(
            initial_program_path=...,
            evaluator_path=...,
            evaluator_env_vars={"MY_PROBLEM": "..."},
        )

resolver = MyResolver()      # module-level instance is required
```

`BenchmarkResolver` is in `skydiscover/benchmarks/base.py`; `BenchmarkResolution`
is defined in `skydiscover/benchmarks/resolution.py` (and re-exported through
`base`). `resolve()` receives the benchmark's `params` dict **and** an
`output_dir` for generated files.

Enable it in config:

```yaml
benchmark:
  enabled: true
  name: "kernelbench"
  resolver: "benchmarks.kernelbench.resolver"   # Python import path to the module
  params:                                        # passed through as `config`
    level: 1
    problem_id: 3
```

`resolve_benchmark_problem` inserts the CWD into `sys.path` before importing, so
the resolver path is relative to where you launch the run.

`evaluator_env_vars` is how per-run problem selection reaches the evaluator
without polluting the process environment (`-e` flags for Docker; a `_scoped_env`
context manager under an RLock for the native path).

Only in-repo implementation: `benchmarks/kernelbench/resolver.py`.

## Reproducing paper results

`scripts/reproduce/*.sh` — one per suite for 7 of the 9 suites (no script for
`kernelbench` or `image_gen`), plus `run_all.sh`. Each resolves the
evaluator path automatically (`evaluator.py` if present, else `evaluator/`),
tracks background PIDs, and exits non-zero if any run fails.

⚠️ `run_all.sh` launches **78 concurrent `gpt-5` runs at `-i 100` each**, all
internally sequential (every one uses `-s adaevolve`/`-s evox`, for which
`max_parallel_iterations` is inert). Set `ITERATIONS=2` in each script for a
smoke test first.

## Cost and runtime expectations

The repo provides **no cost model, no token accounting, and no budget cap** for
API backends. The only wall-clock datapoint in the tree is
`docs/…/quick-start.mdx` — 5 iterations in ~2 minutes (~24 s/iter) on the
`text_similarity` example.

Sizing anything else requires arithmetic the repo doesn't do: each iteration is
1–3 LLM calls plus 1–3 evaluations, serialized. `hotpot_qa` alone adds ~300
evaluator-side LLM calls **per evaluation**.

AdaEvolve is the exception — it writes per-iteration timings to
`adaevolve_iteration_stats_<ts>.jsonl` in the output dir.
