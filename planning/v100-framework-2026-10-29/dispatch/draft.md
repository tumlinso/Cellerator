# V100 Dispatch CUDA Validation and Tuning Draft

## Task contract

- Task: validate the CUDA origin-to-expert dispatch primitives and seek one bounded, measured improvement on Tesla V100 SM70.
- Baseline: `dispatch.cu` gather, weighted combine, combine reverse, and gather reverse; plan upload/validation is the host setup path in `plans.cu`.
- Shapes: variable origin/selection counts, empty origins/routes, duplicate and overlapping selections, signed and zero weights, feature tails and contiguous feature rows. Exercise FP16 and FP32. Stable origin selection order defines forward reduction order.
- Independent oracle: CPU double or FP32 loops reconstructed solely from host `selection_origin`, per-origin dispatch permutation, and weights. Test forward outputs and reverse-mode inner products / analytic gradients. Use dtype-specific tolerances, preserve FP32 accumulation.
- Hardware/build: admitted Tesla V100 16GB (`sm_70`), CUDA 12.9, GCC 12, narrow single-architecture binary.
- Validation: existing dispatch smoke currently verifies compile/link only. Add device cases and run only under Device Scheduler admission; use Compute Sanitizer memcheck, racecheck, synccheck if available/admitted.
- Performance baseline: resident gather/combine/backward on immutable uploaded plans and buffers; separately report plan construction/upload plus any transfer required by caller. Compare current kernels and one warp-per-dispatch weight-reduction candidate under identical inputs and stream/timing protocol. Do not compare kernel-only result to an end-to-end baseline.
- Promotion: exact matching for FP32 where operation order agrees; bounded FP16 error within the test reference tolerance; no sanitizer findings; improvement over baseline beyond observed run variance in one representative case without material regression in the family matrix. Otherwise keep original path and record the negative result.
- Scope/authority: own dispatch CUDA source, plan CUDA source, their headers, dispatch smoke, and this evidence directory. Root owns scheduler admission, shared CUDA builds and final aggregate evidence. No GPU launch before root supplies the admitted route.

## Risks and alternatives

The existing one-CTA-per-dispatch weight-gradient reduction uses 128 lanes plus shared-memory barriers even when the feature extent is small. A warp-per-dispatch design can process four dispatches per CTA and use warp shuffle reductions, reducing wasted lanes and block barriers for small feature counts. It may regress large features or alter FP32 summation order; benchmark against the existing implementation across both regimes and retain only a measured winner. Gather/combine flattening includes integer row division, but address-generation work may be hidden by memory behavior; do not rewrite without evidence.
