# Bounded WMMA and decomposition design

## Existing implementation to repair

The live rectangular contraction really invokes m16n16k16 half-input/f32-accumulator WMMA, followed by a scalar K tail. Its request uses logical width as ldm and does not establish all pointer/stride/tile/capacity requirements. The fix is on that actual path, not a declaration-only alternate. Official CUDA requirements include 32-byte matrix-pointer alignment, half ldm in multiples of eight elements and collective uniform warp participation [R1].

A scalar tail cannot fix an illegal WMMA prefix. Validate both the base and the derived pointer for every tile; aligned allocations alone do not prove aligned submatrix starts. Decouple physical strides from logical K. The required new execution is N16. Tests must still reject unsafe legacy direct requests such as K17 with ldm17. A legality fix may preserve existing legal widths/tails but is not a mandate to implement a general width portfolio.

## Preparation and exact support

Prepare a bounded cover from existing relation/support information and explicit hints. The initial demo contains a 16x16 dense module and six residual edges; full sparse remains a valid complete candidate. Prefer occupied-group bookkeeping and linear edge marking over an all-source-groups by all-destination-groups census.

For each rectangle, gather source/destination entity IDs, choose aligned half operand panels and retain a map from a valid tile score to the authoritative physical edge slot. Extra computed scores in padded holes are never new biological edges. Validate that each original supported edge is assigned exactly once to a rectangle or sparse residual, and every output write is in range. Reject duplicate/overlapping ownership. Validation is cold and may use host support; hot execution does not read device descriptors back to the CPU.

## Actual stages

```
canonical scalar VJP + explicit operand policy
  -> prepared exact rectangle/residual cover and physical edge map
  -> half-rounded aligned operand packing (refresh on changed versions)
  -> rectangular WMMA score tiles
  -> supported-score extraction into persistent f32 edge-gradient slots
  -> sparse contraction of the exact remaining edges into disjoint slots
  -> optional update of authoritative f16 physical weights
  -> publication event
```

The sparse-only half profile must consume the same quantized operands. The strict-f32 profile is a separate legal arithmetic policy and stays sparse here. Numerical failures cannot be disguised as fallback success. Preserve NaN/inf handling for supported scores and ensure padded writes are neither observed nor accumulated into biological outputs.

The gradient values do not depend on current weights, so changing only a weight generation does not intrinsically invalidate the gradient cover or X/dY packing. Input/cotangent changes do invalidate their packed values. The association with a forward snapshot remains explicit in the gradient binding/effect sequence; accidental reuse of an old cotangent must not be called a fresh learning step.

## Selection and performance

A local prepared selector exposes `automatic`, `force_sparse` and `force_hybrid`. Force changes profitability decisions only. Full-f32 plus force_hybrid is an explicit unsupported request. Half-rounded plus dense/mixed support must have a real runnable hybrid option. Automatic stays conservatively sparse until a justified crossover rule is measured; no runtime autotuning on live mutable state.

Account for preparation, quantization/packing, tile padding, extraction, residual, update, publication, transfers and required observations. Compare resident lifetime as well as cold latency. The tiny fixture proves integration, not a speedup. Use larger controlled dense-heavy, sparse-negative and mixed fixtures in I06. Numerical policies must match between candidates. Timings of a half-rounded WMMA path against a full-f32 sparse path are not an apples-to-apples promotion comparison.

FlashSparse [R5] supports treating redundant tile work and data movement as first-class performance questions, but its reported GPU results are not evidence for Cellerator on V100. The plan deliberately does not import newer-generation tile mechanisms or its speedups.

## Required near-term follow-up

Record `DEFER-WMMA-01` prominently: general contraction and hybrid portfolio, sharing reusable panel representations across apply/transpose/VJP, best tile granularity, wider widths, integrated candidate costing, and alternative authoritative orders. Revisit before these provider/packing interfaces are stabilized. Do not let this bounded N16 implementation become a permanent rule that every biological operation must use 16x16 rectangles.
