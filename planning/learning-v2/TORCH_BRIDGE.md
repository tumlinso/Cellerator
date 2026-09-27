# Composable Torch integration: the consequential implementation seam

## The current route is not the desired route

`components/CelleraTorch/docs/autograd_ops.md` explicitly describes a combined N16 native training step. Its backward recomputes the native forward, performs value/bias updates and returns only the dense-input gradient. The code calls `run_training_program` and returns no native-parameter gradients [S2]. Preserve this explicit compatibility route; do not silently change its behavioral contract.

The new consumer needs a distinct composable path:

    forward(native relation, values, input, activities)
        → Torch-visible output
    backward(output cotangent)
        → requested parameter/input/activity gradients, NO learned-value update
    optimizer step
        → one authorized update, native readiness/generation publication

This is more than exposing a data pointer. Save the actual values used by forward or retain an owner/version that prevents stale replay. Do not assume a native parameter's generation changes merely because a Torch view was modified.

## A small safe design, not another framework

Expose actual differentiable values as Tensor inputs/registered parameters, rather than burying all trainable data in an opaque binding which cannot receive gradients. Non-differentiable topology, identity and prepared resources can remain native handles. Reuse standard Torch module and optimizer semantics where supported.

The existing documented native parameter owner remains CE. CT can hold lifetime-bound registered tensor views, but needs a sanctioned optimizer/update boundary: validate write admission, ensure prior readers have completed, run the selected optimizer arithmetic exactly once, publish updated generations/readiness and refresh derived value projections if needed. A small explicit update/commit wrapper is preferable to pretending an unguarded stock `step()` is safe. Ordinary Torch-only parameters need no forced CE ownership.

Choose the minimum safe realization against actual CE contracts. Do not promise every view supports every optimizer, storage mutation or optimizer option. One normal training path is enough initially; document unsupported mutations or options. A new canonical CE mutation/admission capability belongs in CE; the Torch hook/wrapper belongs in CT; GH merely selects the inference schedule.

## Required behaviors

Return correctly accumulated gradients for shared parameters used at several times/sites. Permit the selected gradient-accumulation workflow before a single update. Treat unused gradients consistently. Do not apply update, weight decay or native loss scaling twice. Keep gradient ordering aligned with the exposed parameter coordinate order. Saved-forward identity includes structure, values, activities and observation versions that actually affect the result.

Use current framework streams and correct lifetime/readiness relationships. Test non-default streams and tail batches; do not hardwire biological dimensions to N16. An explicit bounded width/dtype contract is acceptable; hidden host copies, stale projections and silent unsupported derivatives are not.

Preserve logical checkpoint values/identities while rebuilding native runtime handles on restore. A round trip should not move gradients to another logical edge just because physical packing changed.

## Registration and scope

Current official PyTorch guidance favors registered custom operators for external C++/CUDA computation, with separate autograd and optional transform registrations [W1–W3]. `opcheck` verifies registration contracts, not gradient mathematics. Add mathematical gradient tests separately. Do not create a custom op for every composition already expressible in built-in Torch operations.

Use the host's qualified Torch/libtorch/CUDA ABI, not the newest tutorial wheel by default. Do not make a toolchain upgrade, `torch.compile`, `vmap`, general forward AD, double backward, all optimizers or zero-copy everywhere prerequisites. Where those actions are needed by the selected scientific interrogation, implement/test them or provide an explicit correct selected-response route. Never infer them from first-order backward support.

The acceptance target is an actual imported client: ordinary Torch layers before/after a CE-backed trained operator; loss-driven parameter and input gradients; a safe optimizer step; repeated evaluation; save/reload; and one measured fitting lifecycle. This is the substrate the experiment needs.

A safe initial bridge may explicitly commit/repack updated logical values rather than maintain a zero-copy writable physical projection. Prefer such a measured, correct bounded path to a large allocator/lease redesign. Preserve one authoritative value representation and account for its commit cost. Supporting one optimizer configuration does not require supporting every optimizer option.
