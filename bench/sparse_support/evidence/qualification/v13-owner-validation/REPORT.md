# Scalable parameter-owner preparation qualification

The large workload campaign exposed a quadratic host-side check in
`mechanism_parameter_owner`: each scalar parameter identity was compared with
all earlier identities. A 1.3-million-parameter owner requires roughly 845
billion comparisons on that route, before numerical execution. Interrupted
v12 campaigns retain their partial results and cleanup records separately.

The replacement validates identity/value inputs and sorts a temporary identity
prefix for exact duplicate detection. It preserves canonical identity order,
parameter values, and the original error precedence. Numerical kernels, the
Torch adapter, optimizer ownership, and serialized identities are unchanged.
Cold preparation uses O(P) temporary memory and O(P log P) comparisons.

The focused host differential test passes 500 seeded cases plus malformed
identity/value and error-order cases. The standalone run measured 52.5 ms for
1.3 million identities; that is a single host-validation measurement, not an
end-to-end speedup. See
[host evidence](../../experiments/parameter_owner_identity_validation/validation.txt).
Root enabled assertions for the CMake test in Release builds, rebuilt it, and
reran its CTest successfully (1/1, 0.08 seconds).

The shared build rebuilt the native and Torch extensions and the CUDA workset
test. The immutable v13 snapshot is
`build-sparse-support/evidence/torch-v13-owner-validation/`. Its identity digest
is `4d3b6f0e1a63ea20785975597849bfad0b0f081221c47d9b858d6581b487afd4`;
the Torch extension digest is
`51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`.
[package_identity.json](package_identity.json) records source and snapshot hashes.

Admitted GPU3 job `f35242a8-f55b-4533-be47-c13e54d3fe8c` passed **63 tests**
in 9.26 seconds across sparse-support and the existing mechanism bindings. It
then passed the rebuilt native CUDA workset test, including the posting
cardinality guard. The job succeeded with exit code 0 and verified cleanup.
The retained [scheduler record](retry_job_get.json),
[test output](raw_stdout.txt), [specification](retry_job_spec.json), and
[runner](run_regressions.py) identify the exact execution.

The first qualification attempt passed the 33 sparse-support cases but failed
to import pytest-dependent legacy tests after the scheduler resolved the
Python symlink. It verified cleanup. The corrected runner uses the configured
pytest console entry point; the successful 63-test run above supersedes that
attempt. Its original record is [retained](first_job_get.json).

No sanitizer was rerun for this host-only identity-validation change. The
earlier source-bound numerical and posting sanitizer records remain separate.
The larger timing campaigns use fresh v13 tasks and workspaces, so no v12 and
v13 timings are silently combined.
