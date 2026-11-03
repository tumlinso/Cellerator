# Isolated native candidate runner

This directory contains a private build entry point for candidate dispatch
comparisons. It builds the native CUDA component and candidate runner from one
immutable Cellerator source snapshot. The source root is configured as the
top-level CMake project so its host-only compiler contract remains intact. A
generated, hashed CMake hook attaches the runner after the root has created the
`Cellerator::factorized_interaction_cuda` target. The executable links that
target from the same build and also compiles the minimal direct-layout owner
and unique-output candidate source.

The candidate runner retains its original fork identity in
`PROVENANCE.json` as `base_runner_sha256`. If the shared baseline runner later
receives protocol or input-shard updates, the same provenance file records the
new snapshot file digest as `snapshot_baseline_sha256`. The builder checks that
digest against the immutable source snapshot and records both values in the
build manifest. This preserves the original derivation while showing which
baseline content accompanied the candidate build. The source snapshot and
native build manifest also hash the candidate runner and all linked native
sources.
The candidate-specific CLI is `--candidate native_direct|packet|minimal_direct|direct_unique_output`; the normal protocol still supplies
`--route packet|direct`, input/output paths, gradient mode, UUID, and optional
timing arguments required by `validate_runner.py`.

Build into a fresh directory outside the source snapshot:

```sh
python3 "$SOURCE/bench/factorized_interactions/candidate_integration/build_candidate_runner.py" \
  --source "$SOURCE" \
  --build "$BUILD"
```

The resulting manifest is
`$BUILD/native_build/factorized_interactions_build_manifest.json`. Its runner
is `$BUILD/native_build/cmake/cellerator_factorized_interaction_candidate_runner`;
the path in the manifest is relative to the manifest, so it can be consumed directly by
`validate_runner.py --runner-binary` and `--build-manifest`. For example:

```sh
python3 "$SOURCE/bench/factorized_interactions/validate_runner.py" \
  --workload "$WORKLOAD" \
  --runner "$BUILD/native_build/cmake/cellerator_factorized_interaction_candidate_runner --candidate direct_unique_output" \
  --runner-binary "$BUILD/native_build/cmake/cellerator_factorized_interaction_candidate_runner" \
  --build-manifest "$BUILD/native_build/factorized_interactions_build_manifest.json" \
  --route direct \
  --expected-uuid "$GPU_UUID" \
  --artifact-dir "$ARTIFACTS"
```

The builder targets CUDA 12.9 and SM70 by default, uses GCC 12 as the host
compiler, and does not enable fast math. Configuration and compilation do not
launch a kernel. Runtime validation and timing remain subject to the existing
independent oracle and Scheduler admission path. The source and binary hashes
in the manifest identify the candidate build; they do not imply that the
candidate has passed runtime qualification or performance selection.
