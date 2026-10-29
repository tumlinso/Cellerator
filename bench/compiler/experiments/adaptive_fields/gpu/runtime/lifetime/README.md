# E6 GPU executable lifetime prototype

This experiment-local prototype manages actual CUDA driver `CUmodule` and
`CUfunction` handles produced from NVRTC PTX. It keeps plan metadata separate
from device code and uses `std::shared_ptr` leases for active plans, launches,
and a forward tape. The one-worker compile service limits pending requests to
one, shares one future for duplicate keys, supports deterministic cancellation,
and returns PTX to the main thread. CUDA module loading and publication happen
on the thread with the current CUDA context.

The intended overlap case uses two streams and a host callback condition-variable
gate on stream B. Stream A queues a launch behind a CUDA event recorded after
the gate. Compilation and publication should complete while that old launch is
pending. The current admitted runs did not establish this condition: replacement
`cuModuleLoad` returned only after the 10-second callback watchdog released the
gate, so the replacement may have been published after the old launch was
released. See [E6 result correction](RESULT_CORRECTION.md). The harness now
derives its overlap result from the gate-release timestamp at publication.

The narrower tape case did retain the old module through the later backward
launch and its completion event. This synthetic result does not establish
production training-tape or prepared-projection ownership.
The callback has a 10-second self-release watchdog. A host RAII guard also
releases the callback and synchronizes its stream on both normal and exception
paths; if synchronization fails, it retains the callback storage through
process exit to avoid a use-after-free. Timestamped `E6_STAGE` records bracket
callback enqueue, each module load, and callback release/synchronization, so a
long module load can be distinguished from a callback that never entered. Each
module load also has a 45-second hard process watchdog; if the driver call does
not return, the launcher exits with status 124 and leaves device cleanup to the
parent scheduler rather than waiting indefinitely.

Other cases inject an NVRTC compilation error, cancel a request before NVRTC is
called, submit a duplicate key, reject a completed epoch-1 specialization after
the structure advances to epoch 2, select an epoch-generic fallback, and allow a
value-generation advance while preparing code for the unchanged structure.
All kernel numerical values are runtime arguments and are not part of the code
key.

The source is a lifetime test fixture with small forward and backward kernels;
it is not a Cellerator numerical operation or production runtime API.

## CPU-only build

```sh
python bench/compiler/experiments/adaptive_fields/gpu/runtime/lifetime/prepare.py \
  --output-dir bench/compiler/experiments/adaptive_fields/gpu/runtime/lifetime/artifacts
```

This compiles the launcher only. It does not load a module, initialize a
device, or launch a kernel. The manifest records the NVCC/NVRTC identities,
source hashes, build command, and executable hash.

## Root-owned admitted runtime

The finite GPU command is:

```sh
python bench/compiler/experiments/adaptive_fields/gpu/runtime/lifetime/run.py \
  --manifest bench/compiler/experiments/adaptive_fields/gpu/runtime/lifetime/artifacts/build_manifest.json \
  --output bench/compiler/experiments/adaptive_fields/gpu/runtime/lifetime/artifacts/lifetime_result.json
```

The parent scheduler launcher supplies `CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED=1`,
one UUID-routed `CUDA_VISIBLE_DEVICES` value, and the JobSpec-derived
`ADAPTIVE_FIELDS_RUN_KEY`. The wrapper records the run key only as correlation
metadata. The marker is an accidental-run guard, not authorization or isolation;
the external JobSpec, submission, admission, terminal, and cleanup receipts
remain authoritative. The wrapper checks source and executable hashes and adds
the CUDA 12.9 NVRTC library to `LD_LIBRARY_PATH`. The executable checks these
environment guards before calling CUDA APIs.

Run memcheck, racecheck, and synccheck as separate serial jobs using
`sanitizer_commands.json` under the same parent-owned scheduler admission
environment. The report is accepted only after all three complete successfully.

## Current evidence boundary

The original v3 build and its raw result JSONs remain unchanged. The admitted
normal run and three sanitizer runs completed with verified cleanup; all three
sanitizers reported zero errors/hazards. Their replacement load waited for the
callback watchdog release, so successful replacement while old work remains
pending is unresolved. The v3 output field claiming success was unconditional
and is not acceptance evidence. The updated CPU-built artifact reports the
predicate based on the actual callback-release and publication timestamps; it
has not been run on a GPU. The prototype's shared ownership and event contract
is local to this experiment; it does not imply a production executable-handle
ABI.
