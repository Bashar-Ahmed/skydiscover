# Evaluation

## The three formats

`evaluation/__init__.py::create_evaluator` auto-detects, most-specific-first:

| Order | Detected by | Class | Use when |
|---|---|---|---|
| 1 | dir containing `instruction.md` + `tests/test.sh` + `environment/Dockerfile` | `HarborEvaluator` | external benchmark suites (AlgoTune, BigCodeBench, LiveCodeBench, USACO, …) |
| 2 | dir containing `Dockerfile` + `evaluate.sh` | `ContainerizedEvaluator` | custom deps, data files, isolation |
| 3 | anything else (a `.py` file) | `Evaluator` | simple tasks, no system deps |

All three are **duck-typed, not an ABC**. The shared surface is
`evaluate_program(solution, program_id, mode)`, `evaluate_batch`, `close()`,
`llm_judge`.

## Python evaluator

A module exposing `evaluate(program_path) -> dict`:

```python
def evaluate(program_path):
    score = run_and_grade(program_path)
    return {
        "combined_score": score,        # primary target, MAXIMIZED
        "runtime_s": elapsed,           # any extra numeric metrics are recorded
    }
```

- **`combined_score` drives evolution.** If omitted, `get_score` averages all
  numeric non-bool values in the dict.
- To signal failure, return a metric dict — do **not** raise. See the failure
  screen below.

⚠️ **An `"artifacts"` key in a plain returned dict is NOT unpacked.**
`Evaluator._normalize_result` calls `EvaluationResult.from_dict(result)`, which
assigns the *entire* dict to `metrics` — so `"artifacts"` lands inside `metrics`
as a non-numeric entry and `EvaluationResult.artifacts` stays empty. The README
documents the dict form as if it worked; it does not. To get artifacts to the
LLM from a Python evaluator, return the dataclass:

```python
from skydiscover.evaluation.evaluation_result import EvaluationResult

def evaluate(program_path):
    return EvaluationResult(
        metrics={"combined_score": score},
        artifacts={"feedback": "Off by one in the loop boundary"},
    )
```

The containerized path *does* unpack a top-level `"artifacts"` key from the
JSON, so only plain Python evaluators are affected.

## Containerized evaluator

A directory with `Dockerfile` + `evaluate.sh`. The script must print **one JSON
object** to stdout:

```json
{"status": "success", "combined_score": 0.87, "metrics": {...}, "artifacts": {...}}
```

`status` must be one of `"success"` / `"error"` / `"timeout"`. Anything other
than `"success"` is recorded as a `status` artifact (and trips the failure
screen when paired with a zero score).

Mechanics:
- The image is built and a container started in `DiscoveryController.__init__`
  (`docker run -d --rm [-e K=V]* --entrypoint sleep <tag> infinity`).
- The candidate is piped over stdin (`docker exec -i … tee /tmp/<uuid>`) rather
  than bind-mounted, so no path translation is needed and concurrent
  evaluations don't collide.
- The subclassing seam is `ContainerizedEvaluator._run_container` (plain,
  non-async, dispatched through `run_in_executor`). `HarborEvaluator` overrides
  `_run_container` and `_build_image`; it bypasses the lower-level
  `_run_single_in_container` helper entirely.

⚠️ **No resource limits or sandbox hardening exist.** There is no `--memory`,
`--cpus`, `--network`, `--pids-limit`, `--ulimit`, or `--gpus` on the evaluator
container. LLM-generated code runs with full default container privileges and
unrestricted network access. (The separate `--search claude_code` baseline goes
further still — see `references/gotchas.md`.)

## Harbor task

An unmodified [Harbor](https://harborframework.com/) task directory
(`instruction.md`, `environment/Dockerfile`, `tests/test.sh`). Requires
`pip install harbor` (**not a declared dependency**).

```bash
harbor datasets download algotune@1.0 -o /tmp/algotune
uv run skydiscover-run /tmp/algotune/<id>/algotune-set-cover -m gpt-5 -s best_of_n -i 10
```

⚠️ `HarborEvaluator` uses one fixed solution path and one fixed reward file in a
shared container, so it is **not concurrency-safe** — despite the base class
docstring claiming otherwise. Keep `max_parallel_iterations: 1` with Harbor.

## The failure screen

After evaluation, `_run_iteration` rejects the result when **any** of:

- `metrics["validity"] in (0, -1)`, or
- `metrics["timeout"] is True and metrics["validity"] is None`, or
- `metrics["combined_score"] == 0` **and** an error appears in `metrics` **or**
  `artifacts`.

A rejected result appends to `failed_attempts`, which is re-injected as
`context["errors"]` on the next attempt (self-repair). After the last retry the
iteration returns a `SerializableResult` with `error` set, and
`_process_iteration_result` early-returns — so **the program never enters the
database**.

This predicate is **duplicated** in `default_discovery_controller.py` and
`adaevolve/controller.py`. Change both.

## `mode="train"` / `mode="test"`

Threaded end to end, but **only `ContainerizedEvaluator` forwards it**. The
Python `Evaluator` ignores it, and every shipped `evaluate.sh` in math/ADRS
carries the comment `# MODE ($2) accepted but ignored`. So the post-loop
"authoritative test score" is, for most tasks, a bit-identical re-run of the
train evaluation.

## Cascade evaluation

`evaluator.cascade_evaluation` defaults to `True` in the dataclass but `false`
in most shipped configs. Only `cascade_thresholds[0]` is ever read — the second
element is dead.

## LLM-as-a-judge

`evaluation/llm_judge.py::LLMJudge` scores programs via `evaluator_models` and
appends metrics. Enable with `evaluator.llm_as_judge: true`; see
`configs/llm_judge.yaml`.

- `_parse_response` extracts a JSON dict (fenced ```json block first, then the
  outermost `{...}`). Numeric values become metrics; everything else becomes
  artifacts. Override it for XML/YAML formats.
- Scores are weighted by `llm_pool.weights` across all models
  (`generate_all` — the only caller of that method).

## Writing a new evaluator backend

Match the duck-typed surface used by `DiscoveryController`:

```python
class MyEvaluator:
    async def evaluate_program(self, solution, program_id="", mode="train") -> EvaluationResult: ...
    async def evaluate_batch(self, items) -> list: ...
    def close(self) -> None: ...
    llm_judge = None
```

Return `EvaluationResult(metrics: dict, artifacts: dict)` from
`evaluation/evaluation_result.py`. **Never raise for a bad candidate** — return
a metric dict the failure screen recognises, so the loop can retry with feedback
rather than dying.

Wire detection into `evaluation/__init__.py::create_evaluator`.
