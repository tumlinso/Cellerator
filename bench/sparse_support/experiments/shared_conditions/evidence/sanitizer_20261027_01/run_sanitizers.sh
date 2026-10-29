#!/bin/sh
set -eu

sanitizer=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer
binary=/home/tumlinson/Cellerator/build-shared-condition-probe-12/shared_condition_probe

"$sanitizer" --tool memcheck --error-exitcode 99 "$binary" \
    --edges 2176 --cells 64 --sharing weak --check-only
"$sanitizer" --tool racecheck --error-exitcode 99 "$binary" \
    --edges 2176 --cells 64 --sharing weak --check-only
