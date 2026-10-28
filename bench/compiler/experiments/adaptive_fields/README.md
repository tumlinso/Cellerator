# Bounded adaptive execution-field experiments

These are isolated architectural probes. They do not add a production JIT,
change a numerical kernel, or establish new public APIs or dispatch defaults.
See [the initial contract](docs/draft.md) for equations, bounds, and acceptance.
The [consolidated decision report](REPORT.md) separates demonstrated behavior,
unresolved interfaces, performance evidence, and proposed production follow-up.

## Evidence map

- [CPU E1–E4 and E6](cpu/REPORT.md): source-driven field composition, selected
  semantic registration, dependency replay, actual CPU object loading and reuse,
  and one-worker executable lifetime.
- [Independent artifact/lifetime verification](verification/lifetime/REPORT.md):
  generated-artifact oracle, cross-process reuse, failure/retry counterexample,
  and host lease retirement.
- [Independent estimator verification](verification/estimator/README.md): explicit
  surrogate gradients, recurrent VJPs, and replay of the exported native trace.
- [GPU probes](gpu/README.md): frozen package/fixture checks and native training,
  indexer, runtime compiler, and CUDA lifetime evidence.
- `evidence/scheduler/`: actual JobSpecs, submission receipts, logs, and terminal
  cleanup records. Failed attempts are retained.
- `evidence/*manifest.json`: source identity snapshots; child probes also bind
  native artifacts, fixtures, and toolchains to their own results.

Large arrays, checkpoints, source snapshots, and binaries are retained in a
verified archive under `/mnt/block`, without remote upload. The exact archive
path, SHA-256, member inventory, and restore command are in
[LOCAL_ARTIFACTS.json](LOCAL_ARTIFACTS.json). Restore those files into this
directory before following links to raw payloads or reusing built executables.
The originals also remain locally; the experiment's `.gitignore` excludes them
from the compact source/evidence commit.

## Reproduction

CPU build/run commands are in the CPU and independent-verification reports.
They use canonical Cellerator sources with experiment-local adapters, rather
than installing or changing the library.

GPU executables and Torch probes must run through Device Scheduler admission.
`make_job.py` prepares an exclusive UUID-selected finite JobSpec; it does not
submit it. For example, from the repository root:

```sh
python bench/compiler/experiments/adaptive_fields/make_job.py \
  --stage runtime-sanitizers \
  --gpu-uuid GPU-d745da9c-7649-8334-41f1-483afa6f3206 \
  --package-root build-sparse-support/evidence/torch-v14-optional-policies/package \
  --timeout 3000 \
  --output bench/compiler/experiments/adaptive_fields/evidence/scheduler/new_spec.json \
  -- /home/tumlinson/Software/miniconda3/bin/python3 \
  /home/tumlinson/Cellerator/bench/compiler/experiments/adaptive_fields/runtime_sanitizers.py
```

Rediscover current resource availability before selecting a device. Submit the
prepared specification with the installed Cellerator Device Scheduler client;
retain its actual job/attempt and terminal cleanup receipt. Existing specs are
historical evidence, with idempotency keys identifying their original jobs.

Environment markers are guards against accidental direct invocation, not proof
of admission or process isolation. The installed scheduler supplies UUID routing
but does not inject job/attempt IDs. New specs include `ADAPTIVE_FIELDS_RUN_KEY`
for correlation with external scheduler receipts; no synthetic job IDs are used.

## Interpretation

Passes apply only to the stated operation, precision, identity, lifetime, and
fixture contracts. Local descriptors, rewrite selectors, and lifetime managers
are prototypes, not evidence that corresponding production integration exists.
Direct-versus-staged agreement is separate from independent mathematical
validation. Synthetic fixtures provide no biological validation.
