#!/bin/sh
set -eu

/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer \
    --tool synccheck --error-exitcode 99 \
    /home/tumlinson/Cellerator/build-shared-condition-probe-12/shared_condition_probe \
    --edges 2176 --cells 64 --sharing weak --check-only
