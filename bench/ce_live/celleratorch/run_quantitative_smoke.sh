#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
fixture_dir=/tmp/cellerator-ce-live-44
fixture_bin="$fixture_dir/pbmc3k-r512-s7.bin"

mkdir -p "$fixture_dir"
python "$repo_root/bench/ce_live/forward/prepare_pbmc3k_fixture.py" \
    --fixture "$repo_root/bench/ce_live/fixture/local/pbmc3k-r512-s7.npz" \
    --manifest "$repo_root/data/manifests/ce_live/pbmc3k_quantitative_v1.json" \
    --output "$fixture_bin"

"$repo_root/build-celleratorch/celleraTorchQuantitativeSmokeTest" \
    "$fixture_bin"
