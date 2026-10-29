#!/usr/bin/env bash
set -euo pipefail

probe=/tmp/cellerator_parallel_workset_build129/parallel_workset_probe
expected_uuid=GPU-d745da9c-7649-8334-41f1-483afa6f3206
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "$expected_uuid" ]]; then
  printf 'expected scheduler-visible GPU %s; got %s\n' "$expected_uuid" "${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 20
fi
printf '{"kind":"identity","probe_sha256":"%s","cuda_visible_devices":"%s","resource_id":"gpu:testlsserver:%s"}\n' \
  "$(sha256sum "$probe" | cut -d' ' -f1)" "$CUDA_VISIBLE_DEVICES" "$CUDA_VISIBLE_DEVICES"

"$probe" --label e2176_b64_skew09 --edges 2176 --cells 64 --entities 256 --source-skew 0.9 --gate-density 0.5 --seed 20261027 --warmup 3 --samples 21 --iterations 3
"$probe" --label batch_b1_skew09 --edges 2176 --cells 1 --entities 256 --source-skew 0.9 --gate-density 0.5 --seed 20261027 --warmup 3 --samples 21 --iterations 3
"$probe" --label batch_b512_skew09 --edges 2176 --cells 512 --entities 256 --source-skew 0.9 --gate-density 0.5 --seed 20261027 --warmup 3 --samples 21 --iterations 3
"$probe" --label derivative_heavy --edges 2176 --cells 64 --entities 256 --source-skew 0.9 --gate-density 0.5 --seed 20261027 --derivative-heavy --warmup 3 --samples 21 --iterations 3
