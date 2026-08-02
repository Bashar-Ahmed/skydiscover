#!/usr/bin/env bash
# Reproduce ALE-Bench benchmarks (10 problems x 2 search methods).
# All benchmarks launch in parallel.
set -euo pipefail

# ── Settings ─────────────────────────────────────────────────────────────────
# Only two things to change:

MODEL="gpt-5"                        # main generation model
# MODEL="gemini/gemini-3.0-pro-preview"  # alternative
ITERATIONS=100

# -m sets all models (main + guide/paradigm) to the same MODEL.
# API keys: export OPENAI_API_KEY="sk-..." (and/or GEMINI_API_KEY for Gemini)

# ── Install ──────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/../.."
uv sync --extra external

# ── Helper ───────────────────────────────────────────────────────────────────

# Resolve a task's evaluator: a plain evaluator.py when the task ships one,
# otherwise the containerized evaluator/ directory (which most tasks use).
_eval_path() {
  local d=$1
  if [[ -f "$d/evaluator.py" ]]; then
    echo "$d/evaluator.py"
  elif [[ -d "$d/evaluator" ]]; then
    echo "$d/evaluator"
  else
    echo "ERROR: no evaluator found in $d" >&2
    return 1
  fi
}

run() {
  local dir=$1 search=$2
  local init="$dir/initial_program.py"
  [[ -f "$dir/initial_program.cpp" ]] && init="$dir/initial_program.cpp"
  [[ -f "$dir/initial_prompt.txt" ]] && init="$dir/initial_prompt.txt"
  local cfg="$dir/config.yaml"
  [[ -f "$dir/config_${search}.yaml" ]] && cfg="$dir/config_${search}.yaml"
  echo "== $search: ${dir#benchmarks/} =="
  uv run skydiscover-run "$init" "$(_eval_path "$dir")" \
    -c "$cfg" -s "$search" -m "$MODEL" -i "$ITERATIONS" \
    -o "outputs/reproduce/$search/${dir#benchmarks/}"
}

# ── AdaEvolve ────────────────────────────────────────────────────────────────
pids=()

run benchmarks/ale_bench/ale-bench-lite-problems/ahc008 adaevolve & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc011 adaevolve & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc015 adaevolve & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc016 adaevolve & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc024 adaevolve & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc025 adaevolve & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc026 adaevolve & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc027 adaevolve & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc039 adaevolve & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc046 adaevolve & pids+=($!)

# ── EvoX ─────────────────────────────────────────────────────────────────────

run benchmarks/ale_bench/ale-bench-lite-problems/ahc008 evox & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc011 evox & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc015 evox & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc016 evox & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc024 evox & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc025 evox & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc026 evox & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc027 evox & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc039 evox & pids+=($!)
run benchmarks/ale_bench/ale-bench-lite-problems/ahc046 evox & pids+=($!)

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
if (( fail > 0 )); then
  echo "ale_bench.sh: $fail of ${#pids[@]} runs FAILED." >&2
  exit 1
fi
echo "ale_bench.sh: all ${#pids[@]} runs finished."
