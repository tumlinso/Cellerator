#!/usr/bin/env bash
set -euo pipefail

CELLERATOR_ROOT=/home/tumlinson/Cellerator
EXPERIMENT_ROOT="$CELLERATOR_ROOT/bench/sparse_support/experiments/shared_conditions/scaled"
RUN_DIR="$EXPERIMENT_ROOT/evidence/runs/gpu1_20261027_01"
BIN="$RUN_DIR/scaled_condition_probe"
COMPUTE_SANITIZER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer

mkdir -p "$RUN_DIR"
cd "$CELLERATOR_ROOT"

{
    date --utc --iso-8601=seconds
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
} | tee "$RUN_DIR/hardware.txt"

SMALL="$EXPERIMENT_ROOT/evidence/input/v2/small"
EVENTS="$EXPERIMENT_ROOT/evidence/input/v2/large-events"
RANKED="$EXPERIMENT_ROOT/evidence/input/v3/large-ranked"

run_case() {
    "$BIN" "$@"
}

for item in \
    "small|$SMALL|96|8|64|8|" \
    "large-ranked|$RANKED|262144|32|4096|128|--ranked-presence" \
    "large-events|$EVENTS|1280000|8|20000|256|"
do
    IFS='|' read -r LABEL DIRECTORY EDGES CELLS GENES MODULES VIEW_FLAG <<< "$item"
    VIEW_ARGS=()
    if [[ -n "$VIEW_FLAG" ]]; then VIEW_ARGS+=("$VIEW_FLAG"); fi
    run_case --label "$LABEL" --fixture-dir "$DIRECTORY" --edges "$EDGES" \
        --cells "$CELLS" --genes "$GENES" --modules "$MODULES" \
        "${VIEW_ARGS[@]}" --check-only \
        > "$RUN_DIR/correctness_${LABEL}.jsonl"
done

for item in \
    "small|$SMALL|96|8|64|8|" \
    "large-ranked|$RANKED|262144|32|4096|128|--ranked-presence" \
    "large-events|$EVENTS|1280000|8|20000|256|"
do
    IFS='|' read -r LABEL DIRECTORY EDGES CELLS GENES MODULES VIEW_FLAG <<< "$item"
    VIEW_ARGS=()
    if [[ -n "$VIEW_FLAG" ]]; then VIEW_ARGS+=("$VIEW_FLAG"); fi
    "$COMPUTE_SANITIZER" --tool memcheck --error-exitcode 99 --target-processes all \
        "$BIN" --label "$LABEL" --fixture-dir "$DIRECTORY" --edges "$EDGES" \
        --cells "$CELLS" --genes "$GENES" --modules "$MODULES" \
        "${VIEW_ARGS[@]}" --check-only \
        > "$RUN_DIR/memcheck_${LABEL}.log" 2>&1
done

"$COMPUTE_SANITIZER" --tool synccheck --error-exitcode 99 --target-processes all \
    "$BIN" --label large-events --fixture-dir "$EVENTS" --edges 1280000 \
    --cells 8 --genes 20000 --modules 256 --check-only \
    > "$RUN_DIR/synccheck_large-events.log" 2>&1

for item in \
    "small|$SMALL|96|8|64|8|" \
    "large-ranked|$RANKED|262144|32|4096|128|--ranked-presence" \
    "large-events|$EVENTS|1280000|8|20000|256|"
do
    IFS='|' read -r LABEL DIRECTORY EDGES CELLS GENES MODULES VIEW_FLAG <<< "$item"
    VIEW_ARGS=()
    if [[ -n "$VIEW_FLAG" ]]; then VIEW_ARGS+=("$VIEW_FLAG"); fi
    run_case --label "$LABEL" --fixture-dir "$DIRECTORY" --edges "$EDGES" \
        --cells "$CELLS" --genes "$GENES" --modules "$MODULES" \
        "${VIEW_ARGS[@]}" --samples 21 --warmups 3 \
        > "$RUN_DIR/benchmark_${LABEL}.jsonl"
done

sha256sum "$BIN" | tee "$RUN_DIR/binary.sha256"
printf 'probe-complete\n' | tee "$RUN_DIR/status.txt"
