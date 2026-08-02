#!/usr/bin/env bash
# Run all reproduce scripts in parallel.
# Each category launches in the background; we wait for all to finish.
# Tip: set ITERATIONS=2 in each script for a quick smoke test.
set -euo pipefail

DIR="$(dirname "$0")"

names=()
pids=()

launch() {
  bash "$DIR/$1" &
  pids+=($!)
  names+=("$1")
}

launch math.sh
launch adrs.sh
launch ale_bench.sh
launch frontier_cs.sh
launch gpu.sh
launch arc.sh
launch prompt_opt.sh

failed=()
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || failed+=("${names[$i]}")
done

if (( ${#failed[@]} > 0 )); then
  echo "run_all.sh: ${#failed[@]} of ${#pids[@]} scripts FAILED: ${failed[*]}" >&2
  exit 1
fi
echo "All reproduce scripts finished."
