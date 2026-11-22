#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 1 ]]; then
  out_dir="$1"
elif [[ $# -eq 2 && "${1:-}" == "--" ]]; then
  out_dir="$2"
else
  echo "usage: $0 [--] <out-dir>" >&2
  exit 2
fi
binary="${CELLERATOR_DEVELOPMENTAL_TIME_BENCH_BIN:-./build/developmentalTimeABBench}"
scenario="${DT_BENCH_SCENARIO_ID:-large-compute}"
mode="${DT_BENCH_MODE:-train}"
rows="${DT_BENCH_ROWS:-2048}"
cols="${DT_BENCH_COLS:-4096}"
nnz_row="${DT_BENCH_NNZ_ROW:-32}"
stem_dim="${DT_BENCH_STEM_DIM:-256}"
hidden_dim="${DT_BENCH_HIDDEN_DIM:-128}"
time_bins="${DT_BENCH_TIME_BINS:-8}"
warmup="${DT_BENCH_WARMUP:-2}"
iters="${DT_BENCH_ITERS:-10}"

"${binary}" \
  --impl cuda \
  --mode "${mode}" \
  --scenario-id "${scenario}" \
  --rows "${rows}" \
  --cols "${cols}" \
  --nnz-row "${nnz_row}" \
  --stem-dim "${stem_dim}" \
  --hidden-dim "${hidden_dim}" \
  --time-bins "${time_bins}" \
  --warmup "${warmup}" \
  --iters "${iters}" \
  --out-dir "${out_dir}"
