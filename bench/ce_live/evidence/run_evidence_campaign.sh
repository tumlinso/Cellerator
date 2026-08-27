#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
mode=${1:-all}
repeats=${2:-3}

if [[ $mode != --run-only ]]; then
    bash "$repo_root/bench/ce_live/forward/run_quantitative_forward_test.sh" --build-only
    bash "$repo_root/bench/ce_live/training/run_training_program_test.sh" --build-only
    bash "$repo_root/bench/ce_live/replay/run_ce_live_program_replay_test.sh" --build-only
    bash "$repo_root/bench/ce_live/concurrency/run_concurrency_test.sh" --build-only
fi

if [[ $mode != --build-only ]]; then
    python "$repo_root/bench/ce_live/evidence/validate_ce_live_evidence.py"
    bash "$repo_root/bench/ce_live/forward/run_quantitative_forward_test.sh" \
        --run-only "$repeats"
    /tmp/cellerator-ce-live-33/training_program_test
    /tmp/cellerator-ce-live-34/ce_live_program_replay_test
    /tmp/cellerator-ce-live-35/ce_live_concurrency_test
fi
