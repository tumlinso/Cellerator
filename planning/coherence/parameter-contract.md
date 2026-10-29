# Transition parameters: implementation decision

Source review at baseline `2f851ddb9719a85064f7140864c614245424f4cc`.
This is an implementation decision for CE-COHERE-05, not a completed capability.

Keep the existing sparse module's flat parameter owner, IDs, offsets, stacked
shapes and optimizer state. Treat that storage layout as a versioned legacy
contract. Native v1 uses block zero for transition dynamics, while detached
routing currently reads every transition-B block. Changing the stored shape or
broadcasting block zero into routing would change existing checkpoints.

Separate mathematical sharing from storage. A new mode-aware numerical entry
point should accept compact shared A[W,W], B[W,I], bias[W], or the corresponding
mechanism stacks. Existing shared modules pass block-zero views without a second
parameter owner; per-mechanism modules pass stacks. Preserve native v1 symbols
and behavior. Expose explicit transition parameter accessors so storage blocks
are not mistaken for independent dynamics.

The constructor mode is immutable and explicit. Existing entry points default
to shared. Save mode, storage-layout and routing-policy identities in checkpoint
metadata. An untagged legacy checkpoint means existing shared dynamics and
existing per-mechanism detached routing over the old stacked owner. Validate an
explicit destination-mode mismatch before changing values or optimizer state.
Never infer the mode merely from tensor rank.

Shared derivatives accumulate across selected cells and mechanisms into block
zero; per-mechanism derivatives accumulate into each mechanism block. Unaffected
state retains its identity derivative. A/bias tails remain an explicit legacy
storage limitation; B tails also serve the current routing policy. An independent
future layout may separate compact transition storage from routing keys, with an
intentional checkpoint/optimizer migration.

Decisive source at the baseline:

- `include/Cellerator/compute/operation/sparse_support/training.hh`: shared v1 contract.
- `src/compute/operation/sparse_support/training.{cc,cu}`: shared block-zero forward/VJP.
- `python/cellerator/torch/sparse_support.py`: stacked allocation and detached routing.
- `bindings/sparse_support/module.cc`: native owner block shapes and restore checks.
- `python/cellerator/torch/mechanism.py`: prepared restore and flat-leaf optimizer state.
- `tests/sparse_support/test_cuda_training.py`: current transition test selects only mechanism zero.

Required additions include distinct parameters for multiple selected mechanisms
and cells, both sharing modes, untouched states, full input/state/parameter
VJPs, CPU references, declared precision checks, invalid modes/shapes, and
legacy/current checkpoint continuation with optimizer state. GPU execution and
sanitizers require the existing admitted execution controls. No code or GPU
validation was performed during this decision review.
