#!/usr/bin/env bash
set -euo pipefail

probe=/tmp/cellerator_parallel_workset_build129/parallel_workset_probe
"$probe" --host-only --label e2176_b64_skew09 --edges 2176 --cells 64 --entities 256 --source-skew 0.9 --gate-density 0.5 --seed 20261027
"$probe" --host-only --label batch_b1_skew09 --edges 2176 --cells 1 --entities 256 --source-skew 0.9 --gate-density 0.5 --seed 20261027
"$probe" --host-only --label batch_b512_skew09 --edges 2176 --cells 512 --entities 256 --source-skew 0.9 --gate-density 0.5 --seed 20261027
"$probe" --host-only --label derivative_heavy --edges 2176 --cells 64 --entities 256 --source-skew 0.9 --gate-density 0.5 --seed 20261027 --derivative-heavy
