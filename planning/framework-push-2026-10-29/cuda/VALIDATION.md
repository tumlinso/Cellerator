# CUDA framework validation, 2026-10-29

## Aggregate result before the 18:44 candidate additions

Focused CUDA C++ library, optional Python extension and eleven smoke executables
(runtime, dispatch, quantization, memory, frontier, attention, grouped_linear,
packed_embedding, reactions, regulatory_gates, parameter_transfer) compiled and linked with
CUDA 12.9.86, GCC 12.4, C++17, SM70, Debug. Six host-only Python binding checks
passed. The runtime executable's `--host-only` checks passed. An installed C++
consumer compiled, linked, and exercised a zero-byte allocation without CUDA
initialization; the installed extension imported and passed the same six
host-only checks. Python compilation and scoped whitespace checks passed.

The final cellular-kernel pass adds packed embedding bags, supplied reaction
application and regulatory gates to the base library, plus a separate
revision-pinned parameter-transfer target linked to the conditional host owner.
The installed consumer loads all three exports and exercises transfer-memory
accounting and regulatory host preparation without initializing CUDA.
Packed-embedding host reference, regulatory `--host-only`, and parameter-transfer
`--host-only` fixtures passed. The reaction GPU fixture was compiled, not run.
All smoke targets use `-UNDEBUG` so assertions remain active in Release builds.

No GPU execution, sanitizer, profiling or timing occurred. Device portions of GPU smoke bodies
remain unexecuted. Attention and grouped-linear host-reference executables
also ran successfully without invoking CUDA. This evidence does not establish numerical device
correctness or performance. NVCC emitted its expected notice about future
removal of offline compilation for pre-SM75 architectures.

## Reproduction

Harness: `/tmp/cellerator-framework-cuda-harness/CMakeLists.txt`, containing
`project(CelleratorFrameworkCUDA LANGUAGES CXX)` and includes of the repository's `cmake/CelleratorConditional.cmake` and
`cmake/CelleratorFrameworkCUDA.cmake`. It sets `PROJECT_SOURCE_DIR` to the
repository because the existing conditional module resolves source paths there,
and disables unrelated host test registration.

Build: `/tmp/cellerator-framework-cuda-build`; installation:
`/tmp/cellerator-framework-cuda-install`; installed C++ consumer:
`/tmp/cellerator-framework-cuda-consumer`.

Configure with `CMAKE_CXX_COMPILER=/usr/bin/g++-12`,
`CMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc`,
`CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12`, `CMAKE_CUDA_ARCHITECTURES=70`,
`CMAKE_BUILD_TYPE=Debug`, `CELLERATOR_ENABLE_FRAMEWORK_CUDA=ON`,
`CELLERATOR_ENABLE_FRAMEWORK_CUDA_PYTHON=ON`,
`CELLERATOR_BUILD_FRAMEWORK_CUDA_SMOKE=ON`,
`CELLERATOR_ENABLE_FRAMEWORK_PARAMETER_TRANSFER=ON`, and
`pybind11_DIR=$(python -m pybind11 --cmakedir)`; then `cmake --build ... -j4`.

Host fixture: `cellerator_framework_cuda_runtime_smoke --host-only`.
Python fixture: append the built or installed extension directory to
`cellerator.__path__` and discover `tests/framework/cuda/test_runtime_binding.py`
with unittest. The optional module is not copied into the source package.

## Review corrections

The root's independent reviewer identified pinned Python transfer ownership
and typed pointer alignment. Returned pinned-transfer tickets now retain all
owners and synchronize before destruction; typed plan launches check alignment.
The reviewer rechecked these fixes. FP64 host route weights explicitly narrow
to FP32; finite rounding/underflow are accepted, overflow/nonfinite rejected.
Integration inspection found and corrected NaN encoding's unsafe codebook
lookup and the no-override decoder-view mismatch. These are source/host checks,
not a substitute for a future admitted GPU run.

The ML extension pass adds sparse CSR attention and cuBLAS segmented linear
forward/backward, plus the Python CudaGroupedLinear wrapper. Integration review
corrected empty-expert overwrite gradients, pointer alignment, parameter/output
alias checks, invalid gradient modes and noexcept destruction. Attention's
matching-forward probability requirement and byte-overflow guards are explicit.
The eleven base CUDA translation units, separate transfer TU and extension rebuilt after these fixes;
the installed consumer links and validates grouped-linear host metadata without
creating a cuBLAS handle. The exported target carries CUDA::cublas.

## Guidance identity

KDA commit: `277c79b8cdd1d13c34a0dfd132b8bb504bcb5afc`.
CUDA skill checkout commit: `4691951cab5ba89216c3b4ee49c20b69261c9d39`.
Consumed cuda SKILL.md/native guide and KDA agent entry/basic flow/agent flow.

## Source fingerprints (18:43 cellular-kernel build, before candidate additions)

- `include/Cellerator/modeling/cuda/attention.hh`: `0c2d07fb3e8608b698d5a511ca92a36f24f1ed7ce75aa3e9d2ddb3a67c0dd283`
- `include/Cellerator/modeling/cuda/dispatch.hh`: `7edefe924d4b577955a6a105813c79846257b598b346f9531500552b132f24e2`
- `include/Cellerator/modeling/cuda/frontier.hh`: `a38341b7ad7909bd7493f4048a3b30a2a867cc2e54a622b1ee6e688fd1363531`
- `include/Cellerator/modeling/cuda/grouped_linear.hh`: `0c24d60f57fb745701a0ff925bf0b4815445ca0b72c0d32b0995e96249a8bc09`
- `include/Cellerator/modeling/cuda/memory.hh`: `51ed6a64fee6caa504308e17a724971a4188ffada40eeef36a478501d2ecd8a8`
- `include/Cellerator/modeling/cuda/packed_embedding.hh`: `9e2b8906f68f61efc0c56b60f565c60473d915d9b4b3d452fa96a40f25139a7d`
- `include/Cellerator/modeling/cuda/parameter_transfer.hh`: `0aea2d86a178b17a1845a68386b69f2f5436a8c8d0a56d7700b90280528c2d4e`
- `include/Cellerator/modeling/cuda/plans.hh`: `0e7a852d126cf2818fb5f29009e2824e206d63a0cfd24671413bf8f4478f7d31`
- `include/Cellerator/modeling/cuda/quantization.hh`: `341be48630dd51552bd0bbae39d3db8f8f0e5bc375f9fec93a320e42285f9667`
- `include/Cellerator/modeling/cuda/reactions.hh`: `6a47130432bd7a6522e1c6e19d216fe170d626929fb8840f2d526a45d839721e`
- `include/Cellerator/modeling/cuda/regulatory_gates.hh`: `fd4b1b2231e92c6800cd61b7ab434920760940fdcaf161c9cff4a95e42ad61f1`
- `include/Cellerator/modeling/cuda/runtime.hh`: `e89d9bb08367a47ffdfcf99e1659f98f0246d1912905bd0b5b5d9ae08eaa374a`
- `src/modeling/cuda/attention.cu`: `2987f43961d5150d6991d2ffd11ffd54a706c2fc2d875911df1c5f06c5cbc8ea`
- `src/modeling/cuda/dispatch.cu`: `b1b502222dc5f9d28f3a78271b643308a0cc8efadaf605ba62834022951665ca`
- `src/modeling/cuda/frontier.cu`: `3dbd8f04fc03dee34cf4ce6adae3345aa487ad6e16c6bff65e516cc9d8e5dede`
- `src/modeling/cuda/grouped_linear.cu`: `3725f4e212531442b13df900f351b5592258acacc4b42241a722212c41505469`
- `src/modeling/cuda/memory.cu`: `1759526002f05ab63c8b5037a3ceda3fbb81e9e0d85c0a2ddaf346843cc0d0a9`
- `src/modeling/cuda/packed_embedding.cu`: `ca686517c86eead656a79a09b10b0c75052355a697b1c7d169e18f3785236b8b`
- `src/modeling/cuda/parameter_transfer.cu`: `98bcdfcda290bb1599c85019243ae0038401cbadfcb2b5cc32fffd6eb1f883e1`
- `src/modeling/cuda/plans.cu`: `2ba6afce4437f349b7920e1030df588deb2ae7d4a165f9a2096098faaaa71a89`
- `src/modeling/cuda/quantization.cu`: `d8aedbe838d89192220ac1e229001dd242a165a06bf87309512696264fcc7e1b`
- `src/modeling/cuda/reactions.cu`: `d5906f424dd0079a0fc83572944799a0d0132a465baef4e16ff10857811d2120`
- `src/modeling/cuda/regulatory_gates.cu`: `33901830de8d43a7171106e4670d24006b531466640e8431c44327c400f8d8f9`
- `src/modeling/cuda/runtime.cu`: `8dec5ff0a107758bac58f0a586fa60920f70b5105e9c548a0af03eab9273065b`
- `bindings/modeling_cuda/module.cc`: `7d66395d138a0e274b31237121b934ebd749bce4268b83e51addf95edd4a633b`
- `python/cellerator/adapters/cuda.py`: `ca5537099dac420dab76505ea0a3990571ae735d40b1cf3b5511152f68b7be93`
- `cmake/CelleratorFrameworkCUDA.cmake`: `67a921c7bcaeea011b1f499a4cffd6e74896caee011c436260c9b78f2558f862`

## Experimental candidates added at 18:44 UTC

Two opt-in alternatives were added to existing translation units and smoke
fixtures. Existing default entrypoints remain unchanged.

- `sparse_attention_fused_small_fp16` and `_fp32` fuse score, softmax and value
  aggregation for neighborhoods with at most 32 edges. Larger neighborhoods
  use the staged implementation. The fused path avoids logits materialization
  for small neighborhoods but retains probabilities required by backward.
- `evaluate_regulatory_bitplanes` is a separate feature-major representation
  with 32 independent cells per uint32 word, explicit validity/tail masks,
  sparse rule-feature indexes and packed eligibility output. It does not
  automatically convert or replace the occurrence-based regulatory API.

The owners reported CUDA 12.9 SM70 compile/link success and host-oracle success
for the updated fixtures, including regulatory bitplane truth tables and tail
handling. These are compile and host-reference results, not device numerical
validation or measured performance. No GPU execution, timing, sanitizer or
promotion claim is associated with either candidate. The aggregate/install
results earlier in this record describe the pre-candidate fingerprints; the
root owns the final aggregate rebuild and staging after these two additions.
No additional build, install or test was run by the runtime integration owner
for this documentation update.

### Candidate source fingerprints recorded at 18:45 UTC

- `include/Cellerator/modeling/cuda/attention.hh`: `0c2d07fb3e8608b698d5a511ca92a36f24f1ed7ce75aa3e9d2ddb3a67c0dd283`
- `src/modeling/cuda/attention.cu`: `2ac78d8c29abe73c003e4f834e5ec4d9282293e7ce9769cc396cd1172a294d3a`
- `tests/framework/cuda/attention_smoke.cu`: `bc364ed0bc3873bc478f7d859d382f63cb2ffb09fdaf2113d9a86c3556831b00`
- `include/Cellerator/modeling/cuda/regulatory_gates.hh`: `1455e471d5e04f7204735177d00dc3f1c8d6d1d9be71515b0f847e20098fabce`
- `src/modeling/cuda/regulatory_gates.cu`: `a9bb2054723ca048d01bd65ff97de58c1f4d111feab6ae9ea9835ec3f055d808`
- `tests/framework/cuda/regulatory_gates_smoke.cu`: `2e344a6e4087c84456675de176a3c5e4ee43a21ec3b46d7ae6f8e9fb4034e54c`
