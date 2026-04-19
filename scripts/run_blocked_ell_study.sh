#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_blocked_ell_study.sh [options]

Options:
  --build-dir DIR        CMake build directory. Default: ./build
  --manifest PATH        Study manifest. Default: bench/real_data/generated/blocked_ell_optimization/manifest.tsv
  --dataset ID           Dataset id filter. Default: all
  --algorithm NAME       Algorithm filter. Default: all
  --block-size N         Fixed blocked-ELL block size. Default: 16
  --bucket-cap N         Exact-DP bucket cap. Default: 16
  --rhs-cols N           Dense RHS columns. Default: 128
  --profile MODE         none | nsys | ncu. Default: none
  --json-out PATH        Optional JSON summary output
  --skip-build           Do not rebuild the benchmark target
  --skip-spmm            Skip the CUDA SpMM runtime proxy
  -h, --help             Show this help

The wrapper checks live GPU memory/utilization first and exits immediately if
no suitably idle GPU is available. It records the pre-run GPU state before
running the benchmark or profiler.
EOF
}

BUILD_DIR="./build"
MANIFEST="bench/real_data/generated/blocked_ell_optimization/manifest.tsv"
DATASET="all"
ALGORITHM="all"
BLOCK_SIZE="16"
BUCKET_CAP="16"
RHS_COLS="128"
PROFILE_MODE="none"
JSON_OUT=""
SKIP_BUILD=0
SKIP_SPMM=0

while (($# > 0)); do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --algorithm)
      ALGORITHM="$2"
      shift 2
      ;;
    --block-size)
      BLOCK_SIZE="$2"
      shift 2
      ;;
    --bucket-cap)
      BUCKET_CAP="$2"
      shift 2
      ;;
    --rhs-cols)
      RHS_COLS="$2"
      shift 2
      ;;
    --profile)
      PROFILE_MODE="$2"
      shift 2
      ;;
    --json-out)
      JSON_OUT="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-spmm)
      SKIP_SPMM=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

choose_idle_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F',' '
        {
          gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
          if (($2 + 0) == 0 && ($3 + 0) <= 5) {
            print $1;
            exit 0;
          }
        }
      '
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi is required for the GPU availability check.\n' >&2
  exit 127
fi

GPU_ID="$(choose_idle_gpu || true)"
if [[ -z "${GPU_ID}" ]]; then
  printf 'No idle GPU is currently available; skipping instead of waiting on the mutex.\n' >&2
  exit 3
fi

RUN_ROOT="bench/real_data/generated/blocked_ell_optimization/runs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${RUN_ROOT}"
nvidia-smi > "${RUN_ROOT}/nvidia-smi.before.txt" || true
printf 'selected_gpu=%s\n' "${GPU_ID}" | tee "${RUN_ROOT}/gpu.txt"

if ((SKIP_BUILD == 0)); then
  cmake -S . -B "${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" -j 4 --target blockedEllStudyBench
fi

CMD=(
  "${BUILD_DIR%/}/blockedEllStudyBench"
  --manifest "${MANIFEST}"
  --dataset "${DATASET}"
  --algorithm "${ALGORITHM}"
  --device "${GPU_ID}"
  --block-size "${BLOCK_SIZE}"
  --bucket-cap "${BUCKET_CAP}"
  --rhs-cols "${RHS_COLS}"
)

if [[ -n "${JSON_OUT}" ]]; then
  mkdir -p "$(dirname "${JSON_OUT}")"
  CMD+=(--json-out "${JSON_OUT}")
fi
if ((SKIP_SPMM != 0)); then
  CMD+=(--skip-spmm)
fi

case "${PROFILE_MODE}" in
  none)
    "${CMD[@]}" | tee "${RUN_ROOT}/benchmark.txt"
    ;;
  nsys)
    /home/tumlinson/.agents/skills/cuda/scripts/profile_nsys.sh \
      --out-dir "${RUN_ROOT}/nsys" \
      --label "blocked-ell-study" \
      -- "${CMD[@]}"
    ;;
  ncu)
    /home/tumlinson/.agents/skills/cuda/scripts/profile_ncu.sh \
      --out-dir "${RUN_ROOT}/ncu" \
      --label "blocked-ell-study" \
      -- "${CMD[@]}"
    ;;
  *)
    printf 'Unsupported profile mode: %s\n' "${PROFILE_MODE}" >&2
    exit 2
    ;;
esac

nvidia-smi > "${RUN_ROOT}/nvidia-smi.after.txt" || true
printf 'artifacts: %s\n' "${RUN_ROOT}"
