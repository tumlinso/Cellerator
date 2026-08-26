#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
build_root="$repo_root/build-dissolution-smoke"
sanitizer="/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer"

for test_binary in \
    celleratorProjectionActivationTest \
    celleratorCusparseCsrCandidateTest \
    celleratorPreparationFactoryTest \
    celleratorQuantitativeRelationTest \
    celleratorNativeTrainingSliceTest
do
    "$sanitizer" --tool memcheck --error-exitcode 99 \
        "$build_root/$test_binary"
done
