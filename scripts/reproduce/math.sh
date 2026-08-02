#!/usr/bin/env bash
# Reproduce math benchmarks (17 problems x 2 search methods).
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
uv sync --extra math

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

run benchmarks/math/circle_packing           adaevolve & pids+=($!)
run benchmarks/math/circle_packing_rect      adaevolve & pids+=($!)
run benchmarks/math/erdos_min_overlap        adaevolve & pids+=($!)
run benchmarks/math/first_autocorr_ineq      adaevolve & pids+=($!)
run benchmarks/math/second_autocorr_ineq     adaevolve & pids+=($!)
run benchmarks/math/third_autocorr_ineq      adaevolve & pids+=($!)
run benchmarks/math/uncertainty_ineq         adaevolve & pids+=($!)
run benchmarks/math/hexagon_packing/11       adaevolve & pids+=($!)
run benchmarks/math/hexagon_packing/12       adaevolve & pids+=($!)
run benchmarks/math/heilbronn_convex/13      adaevolve & pids+=($!)
run benchmarks/math/heilbronn_convex/14      adaevolve & pids+=($!)
run benchmarks/math/heilbronn_triangle       adaevolve & pids+=($!)
run benchmarks/math/minimizing_max_min_dist/2 adaevolve & pids+=($!)
run benchmarks/math/minimizing_max_min_dist/3 adaevolve & pids+=($!)
run benchmarks/math/matmul                   adaevolve & pids+=($!)
run benchmarks/math/signal_processing        adaevolve & pids+=($!)
run benchmarks/math/sums_diffs_finite_sets   adaevolve & pids+=($!)

# ── EvoX ─────────────────────────────────────────────────────────────────────

run benchmarks/math/circle_packing           evox & pids+=($!)
run benchmarks/math/circle_packing_rect      evox & pids+=($!)
run benchmarks/math/erdos_min_overlap        evox & pids+=($!)
run benchmarks/math/first_autocorr_ineq      evox & pids+=($!)
run benchmarks/math/second_autocorr_ineq     evox & pids+=($!)
run benchmarks/math/third_autocorr_ineq      evox & pids+=($!)
run benchmarks/math/uncertainty_ineq         evox & pids+=($!)
run benchmarks/math/hexagon_packing/11       evox & pids+=($!)
run benchmarks/math/hexagon_packing/12       evox & pids+=($!)
run benchmarks/math/heilbronn_convex/13      evox & pids+=($!)
run benchmarks/math/heilbronn_convex/14      evox & pids+=($!)
run benchmarks/math/heilbronn_triangle       evox & pids+=($!)
run benchmarks/math/minimizing_max_min_dist/2 evox & pids+=($!)
run benchmarks/math/minimizing_max_min_dist/3 evox & pids+=($!)
run benchmarks/math/matmul                   evox & pids+=($!)
run benchmarks/math/signal_processing        evox & pids+=($!)
run benchmarks/math/sums_diffs_finite_sets   evox & pids+=($!)

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
if (( fail > 0 )); then
  echo "math.sh: $fail of ${#pids[@]} runs FAILED." >&2
  exit 1
fi
echo "math.sh: all ${#pids[@]} runs finished."
