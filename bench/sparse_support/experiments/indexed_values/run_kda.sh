#!/bin/bash
set -euo pipefail
cd /home/tumlinson/Cellerator
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED=1
export CELLERATOR_SPARSE_PACKAGE_ROOT=/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v11/package
export PYTHONPATH=/home/tumlinson/kda:/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v11/package
OUT=/home/tumlinson/Cellerator/bench/sparse_support/experiments/indexed_values/evidence
mkdir -p "$OUT"
/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -m kda_experiments \
  --task /home/tumlinson/Cellerator/bench/sparse_support/experiments/indexed_values/task.json \
  --workspace /home/tumlinson/Cellerator/bench/sparse_support/experiments/indexed_values/kda_workspace \
  evaluate --candidates full_scan indexed_union --incumbent full_scan \
  > "$OUT/kda_stdout.jsonl" 2> "$OUT/kda_stderr.txt"
