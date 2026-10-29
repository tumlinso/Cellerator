# CE-LIVE current-head baseline

This record closes the validation-only scope of CE-LIVE-10. It establishes the
Wave A correctness baseline used by the CE-LIVE-19 foundation fan-in; it does
not make a performance claim and it does not authorize production fixes.

## Source, toolchain, and device identity

- Validated source commit: `4df609998f380aa93e14f7050845c7a1a9ebd109`
- Source state before validation: clean except for generated todo projections
  belonging to the active CE-LIVE-10 claim.
- Configure: CMake 3.28.3, native mode with
  `CELLERATOR_ENABLE_TORCH_MODELS=OFF`.
- CUDA compiler:
  `/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc`, CUDA 12.9.86,
  targeting `sm_70`.
- Device: Tesla V100-SXM2-16GB, compute capability 7.0, driver 580.173.02;
  controller reservation used GPU UUID
  `GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a`.
- CUDA controller observed no foreign GPU activity and obtained the repository
  benchmark mutex for every device run.

## Reproducible host and build validation

```bash
cmake -S . -B build-dissolution-smoke -DCELLERATOR_ENABLE_TORCH_MODELS=OFF
cmake --build build-dissolution-smoke -j 4 --target \
  cellPackSemanticGeometryAdapterTest \
  cpMathRefereeFoundationTest \
  celleratorOperationCoreTest \
  celleratorExecutionSessionTest \
  cellPackExecutionImageV2DeviceTest \
  celleratorRowMaskedN1CandidateTest \
  celleratorCsrFallbackCandidateTest \
  celleratorFeatureMajorSmallNCandidateTest \
  celleratorTransposeBackwardCandidateTest \
  celleratorValueGenerationReuseTest
./build-dissolution-smoke/cellPackSemanticGeometryAdapterTest
./build-dissolution-smoke/cpMathRefereeFoundationTest
./build-dissolution-smoke/celleratorOperationCoreTest
python -m unittest tests/live/fixture/test_ce_live_fixture.py
python bench/ce_live/fixture/prepare_pbmc3k_fixture.py verify \
  --input data/pbmc3k/3k_pbmc_v3_nextgem_filtered_feature_bc_matrix.h5ad \
  --manifest bench/ce_live/fixture/pbmc3k_fixture_v1.json
jq empty bench/ce_live/candidate_inventory/candidate_inventory_v1.json
jq empty bench/ce_live/tensor_core/contract/v100_dense_fragment_candidate_v1.json
```

All configured targets built and all listed host, fixture, and contract checks
passed. Fixture verification emitted
`CELLERATOR_QUANTITATIVE_FIXTURE_V1_READY`. The PBMC3K fixture is computational
evidence only; this validation adds no biological claims. The native cache
confirmed that Torch models remain disabled by default.

The CE-LIVE-16 source audit used its completing commit, `b8b208b`, and confirmed
that the task added only the Tensor Core contract JSON and its documentation.
Legacy experimental WMMA sources elsewhere in the repository remain historical
implementation evidence; CE-LIVE-16 introduced no kernel, registration, build
wiring, or universal dense-fragment requirement.

## CUDA correctness and sanitizer evidence

The versioned controller specifications are under `bench/ce_live/baseline/`.

- Wave A CUDA suite:
  `ce_live_10_wave_a_cuda_suite.json`; evidence
  `a55c75b2-0f42-42f8-9e37-ce9f8eb93a1a`; passed seven focused device tests.
- Value-readiness correctness:
  `ce_live_10_value_readiness_correctness.json`; evidence
  `0a8e4ec3-df08-4ea1-8333-01abd1d19287`; compiled the readiness implementation
  and passed its device test.
- Value-readiness Compute Sanitizer:
  `ce_live_10_value_readiness_sanitizer.json`; evidence
  `964f0224-0eec-43ef-9607-40eefaf39015`; the debug sanitizer wrapper completed
  with return code zero.

The Wave A suite covers the execution session, CPE2 device relocation, current
row-masked and CSR candidates, feature-major small-N execution, explicit
transpose/backward projection, and value-generation reuse. Correctness was
established before sanitizer execution. No timing campaign was launched.

## Closure checks

The task closes only after these commands pass against the final evidence
commit:

```bash
cmake --build build-dissolution-smoke --target \
  celleratorValueGenerationReuseTest -j 4
./build-dissolution-smoke/celleratorValueGenerationReuseTest
git diff --check
git diff --name-status HEAD
```

The scope audit must contain no production implementation. The only authored
paths are this document and `bench/ce_live/baseline/`.
