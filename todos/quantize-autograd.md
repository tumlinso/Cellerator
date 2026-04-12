# Current Objective

## Summary
Move the quantizer reconstruction and range gradients off the dense libtorch path and into fused sparse CUDA kernels under `src/compute/autograd`, while keeping `src/models/quantize/` as the thin model-facing wrapper.

## Planning Notes
- The quantizer math is feature-affine and scalar-heavy, so the hot win is fusion and sparse correction over a zero baseline rather than chaining generic sparse primitives.
- CSR is the first model-facing sparse path; Blocked-ELL support should land in the low-level runtime at the same time so execution-layout selection stays explicit.
- This workstream must not undo or reshape the existing SpMM and benchmark edits in the tree.

## Assumptions
- Sparse quantizer training batches arrive as cell-major CSR (`rows = cells`, `cols = genes`) with implicit zeros.
- Quantizer inputs are non-negative count-like features, so the offset anchor term can treat the dense floor as zero in the sparse CUDA path.

## Suggested Skills
- `todo-orchestrator` - Keep the shared work ledger current while this workstream is active.
- `cuda-v100` - Own the fused CUDA kernel structure and Volta-specific sparse layout decisions.

## Useful Reference Files
- `src/models/quantize/quantize.hh` - Current dense training and evaluation path.
- `src/compute/autograd/autograd.hh` - Pointer-first sparse runtime surface and execution context helpers.
- `src/compute/autograd/kernels/base_sparse.cu` - Existing sparse reference kernels where the new fused quantizer path belongs.
- `custom_torch_ops.md` - Registry for model-facing custom CUDA/autograd boundaries.

## Plan
- Add fused quantizer reconstruction/range forward-backward entrypoints for CSR and Blocked-ELL to `src/compute/autograd`.
- Route sparse CUDA quantizer training through those kernels while leaving the model wrapper and future-neighbor supervision surface intact.
- Add direct runtime coverage for the new low-level kernels and CUDA-side model coverage for sparse CSR training.

## Tasks
- [x] Add fused CSR and Blocked-ELL quantizer loss/gradient kernels to `src/compute/autograd`.
- [x] Hook sparse CUDA quantizer training in `src/models/quantize/`.
- [x] Extend runtime and model tests for the new sparse quantizer path.
- [x] Record the model-op boundary in `custom_torch_ops.md`.

## Blockers
_None recorded yet._

## Progress Notes
- Chose a zero-baseline plus sparse-correction formulation so the kernels can evaluate the full dense objective without materializing the dense input target.
- Added fused CSR and Blocked-ELL reconstruction/range kernels to `src/compute/autograd/kernels/base_sparse.cu`.
- Moved sparse CUDA quantizer reconstruction/range training into `src/models/quantize/quantize_cuda.cu`, keeping `quantize.hh` as the thin model-facing surface.
- Verified the low-level math with `./build/computeAutogradRuntimeTest` and the model integration with `./build/quantizeModelTest` on visible V100 GPUs.

## Next Actions
- Use the new low-level entrypoints as the baseline if pair-local quantizer execution or quantizer-specific benchmarks are added later.

## Done Criteria
- Sparse CUDA quantizer training no longer depends on dense reconstruction/range gradients for CSR batches.
- Low-level quantizer kernels exist for both CSR and Blocked-ELL layouts in `src/compute/autograd`.
- Tests cover the new autograd math directly and the model wrapper consumes it successfully.
