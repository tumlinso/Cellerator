# Device and precision milestone: implementation contract

Implement the approved coordinated milestone directly in velocity.cppm and
anndata.cppm. Cellerator and its coding-style authority remain unchanged.

The reference methods are first/second neighborhood moments, deterministic
steady-state fitting, and the existing stochastic joint fit. Storage, compute
precision, and device are independent. FP16 storage uses FP32 arithmetic;
FP32 and FP64 CUDA paths compare with the matching scalar C++ reference on
identical stored inputs. No TF32, implicit fast math, CPU solver fallback, or
automatic precision promotion is part of CUDA execution.

The uniform FP16 path always bypasses screening and promotion. Optional
screening is advisory: two initialization fits perturb active finite trimming
bounds by 0.5*2^-11*max(1,abs(bounds)); report status/selection changes, R2
changes above .05, or prediction changes above .10 of residual L2 (with the
approved 32*epsilon signal floor). Only an explicit gene mask requests original
S/U values, and missing originals retain FP16. Resolve global selection after
promoted initialization; graph precision is unchanged.

Numeric scratch defaults to 64 MiB, including conversion, sorting, and batch
temporaries. Count input/prepared storage, O(genes) diagnostics, and retained
outputs separately. Reuse buffers; do not retain whole-input FP32 shadows.

Initial toolchain is Clang 18 modules plus ordinary NVCC 12.9 translation units,
with SM70 qualification. CPU consumption remains CUDA-independent. AVX2/F16C
selection is build-time and the scalar reference remains explicitly callable.
Private implementation changes must preserve public BMIs/consumer objects;
warm CPU imports must remain within 25 percent of the frozen baseline.

Validation uses existing M1/M2 analytic and source-derived fixtures, independent
binary16/file-format checks, scalar/AVX2/CUDA parity, custom/external storage,
and installed module consumers. Preserve the accepted M2 initialization
exception and strict raw reports. Per-gene scaled moment atol/rtol are
1e-12/1e-11 (FP64), 2e-6/2e-5 (FP32); fit/output gates are 1e-10/1e-9 and
2e-5/2e-4. R2 gates are 1e-10/2e-4; objective gates 1e-9/5e-4.

Performance workers are finite (30 seconds maximum), using the canonical
4096-cell, 2048-gene, approximately 32-neighbor, one-percent-expression workload
and bounded geometry controls. Include allocation, packing, transfers, host
and device memory, and resident/end-to-end timings. CUDA execution requires a
live UUID-specific exclusive Device Scheduler grant; correctness/sanitizer
jobs have a five-minute maximum. Do not impose CPU virtual-address-space
limits on CUDA. No GPU execution is established by compilation alone.

Use the CUDA skill and KDA agent entrypoint/starter flow for this implementation.
Keep source identities and actual command receipts with local evidence; native
Todo acceptance is separate from local validation. The user's execution-handle
waiver remains in force; do not edit semantic authority/projections.
