# Python product mechanisms

The package exposes prepared Cellerator mechanisms as ordinary `torch.nn`
modules while keeping their biological axes explicit. Build CelleraTorch
against the same PyTorch distribution used by the Python process, then point
`CELLERATORCH_NATIVE_LIBRARY` at its mechanism operator shared library before
importing `celleratorch`. Do not load a second libtorch distribution into the
same process.

An `Axis` records stable identities for domain, order, geometry, and partition
separately from its extent. `MechanismSpec` describes ordered argument slots,
logical coefficient identities, and additive output contributions. Repeated
argument indices remain repeated; shared coefficient identities bind to one
canonical native value. A same-shaped axis with a different identity is not a
valid input.

`MechanismModule` requires a contiguous CUDA FP32 initial coefficient tensor.
The native owner stores the canonical logical-order values and prepares the
requested FP32 or mixed FP16 execution plane. Its single registered
`nn.Parameter` is shared when a native owner is reused. Inputs have shape
`[batch, input_axis.extent]`; the prepared batch and outstanding-forward
capacities are explicit.

Use standard Torch layers and loss functions around the module. For a
native-backed optimizer update, use `guarded_step(modules, optimizer)`. The
initial supported optimizer is stock `torch.optim.Adam`, without weight decay,
closures, sparse gradients, foreach/fused/capturable/differentiable modes, or
AMSGrad. The call checks the native owner, runs Adam once, and publishes updated
native readiness. PyTorch's automatic `foreach=None` and `fused=None` choices
are pinned to `False` on the first guarded call. With `GradScaler`, it unscales once; a non-finite step is
skipped without publishing a new generation.

The mixed execution values are stored half values promoted to FP32 for
multiplication and reduction. Gradients use the identity straight-through
convention through half rounding, so master coefficients receive FP32
gradients; this is a declared surrogate, not the derivative of discrete
rounding. Native tapes support one first-order backward each.

Save a logical checkpoint with `save_checkpoint(path, mechanism, optimizer)`
and restore into a freshly prepared model with `load_checkpoint`. The
checkpoint stores ordinary model tensors, logical declarations/IDs, canonical
coefficients, precision, and Adam state keyed by model parameter names; native
pointers, events, and prepared caches are rebuilt. Restore can explicitly
remap a changed coefficient order by logical ID when the biological
domain/geometry/partition and mechanism declarations match. The changed order
must carry its own order identity. Normal forward calls still reject a
same-shaped input with the wrong axis identities. Do not move or cast
native-backed modules with `.to()`/`.half()` or replace their parameter
storage. Direct pointer or `.data` writes bypass the guard and are outside its
detectable mutation guarantee.
