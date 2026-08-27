# CE-LIVE CelleraTorch quantitative validation

CE-LIVE-44 validates CelleraTorch as a thin adapter over the already-live
Cellerator execution system. The checksum-pinned PBMC3K fixture is computational
evidence only; this task makes no donor, sample, chemistry, normalization, or
biological interpretation claim.

## Quantitative forward parity

`celleraTorchQuantitativeSmokeTest` embeds the existing CE-LIVE-31 quantitative
fixture and preparation path. For every run it executes the same prepared
program first through native Cellerator and then through
`run_program_forward`, using the same:

- feature/gene-source to cell/row-destination relation;
- immutable structure and projection;
- value generation and readiness record;
- input/output device storage;
- caller CUDA stream;
- selected candidate and explicit output order.

The test covers widths `1, 16, 17, 31, 32, 48, 64`, reuse horizons `1, 8,
1024`, two value generations, and three repeats. All 126 native-versus-adapter
comparisons were exact within `1e-6` relative tolerance, and the final adapter
output also passed the independent CSR referee used by CE-LIVE-31. Each program
remained prepared exactly once across relocated calls and both generations.

## Adapter cost

The foreground V100 run used one Tesla V100-SXM2-16GB, CUDA 12.9, driver
580.173.02, `sm_70`, the repository benchmark mutex, and an uncontaminated GPU
lease. Median values across the 126 paired samples were:

| measurement | median |
|---|---:|
| native GPU interval | 4003.84 us |
| CelleraTorch GPU interval | 3998.72 us |
| GPU interval delta | -5.12 us |
| native host enqueue | 5314 ns |
| CelleraTorch host dispatch | 5243 ns |
| host dispatch delta | -71 ns |
| zero-copy Torch view metadata construction | 3470 ns |

The negative GPU and host-dispatch deltas are timing noise, not speedup claims:
both routes launch the identical native candidate and their enqueue costs are
indistinguishable at this resolution. The clearly observable framework cost is
Torch tensor metadata construction. No tensor
payload is copied, no projection or CPE2 image is rebuilt, no planner is
reconstructed, and no native parameter is duplicated.

Controller evidence `b7cf7354-ddce-4afb-89ec-0f912a8724a1` passed correctness
and repeated timing. Machine-readable results are in
`bench/ce_live/celleratorch/quantitative_results_v1.json`.

## Autograd and ownership envelope

The combined CelleraTorch build also runs `celleraTorchAutogradOpsTest`. In the
currently supported N=16 training envelope it compares the Torch autograd path
with direct native Cellerator forward/backward/update execution on the same
topology, output gradient, relation values, bias, generation, and current Torch
stream. Dense-input gradients and native parameter updates match; Cellerator
remains the sole parameter/update owner and publishes the next readiness
generation. A Torch optimizer must not update those native parameter aliases a
second time.

The PBMC3K quantitative sweep is forward-only because the frozen native
training program is intentionally N=16 and is not a general-width training
engine. CE-LIVE does not misrepresent that support envelope.

## Sanitizer and reproduction

The CUDA controller's sanitizer debug wrapper substituted an unavailable CUDA
13.1 host shim even though the committed specification named CUDA 12.9; record
`bb4d9a9a-d95c-4b31-833d-bb0cbc47925d` is therefore inconclusive
infrastructure evidence. The exact installed CUDA 12.9 memcheck binary was
rerun under the same benchmark mutex and completed with `ERROR SUMMARY: 0
errors`.

Reproduce the foreground run with:

```bash
python /home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py run \
  --spec bench/ce_live/celleratorch/cuda_controller.json --json
```

Reproduce the conclusive memcheck with:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/tumlinson/.agents/skills/cuda/scripts/with_benchmark_mutex.sh \
  --label ce-live-44-memcheck -- \
  /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer \
  --tool memcheck --error-exitcode 99 \
  ./build-celleratorch/celleraTorchQuantitativeSmokeTest \
  /tmp/cellerator-ce-live-44/pbmc3k-r512-s7.bin
```

This establishes `CELLERATOR_CELLERATORCH_LIVE_V1` without transferring
planning, storage, runtime, parameter, readiness, or numerical ownership to
Torch.
