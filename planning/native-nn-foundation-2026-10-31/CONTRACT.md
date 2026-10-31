# Native NN foundation integration contract

The user's approved native plan governs this implementation. Cellerator is a
biological ML framework, not a simulator or a Torch-only library. New native
models are ordinary C++ types returning arbitrary typed results. There is no
module registration, member traversal, universal model base, or tensor facade.
Parameters are explicitly owned by `nn::Parameters`; components allocate through
lightweight groups. Public structural views expose types, extents and ownership.

## Shared vocabulary

- `cellerator::nn::Run { bool training = false; }` controls dropout/evaluation,
  independently of autograd.
- `Parameters(at::Device, at::ScalarType)`, `.group()`, `.torch_parameters()`.
- `ParameterGroup` has `.options()`, `.add(at::Tensor)`, `.empty(shape)`,
  `.zeros(shape)`, `.normal(shape, stddev)`, `.set_trainable(bool)`. One owner,
  stable numeric slots; groups and tied aliases share actual leaf tensors.
- `Linear(group, input, output, bias=true)`, `Embedding(group, count, width)`,
  `Identity`, and `LinearParameters` support concrete callable composition.
- `FeatureInput` contains `ids`, optional `values`, `observed`, `attributes`.
  `FeatureEncoderOptions` has vocabulary_size, width, optional value_size=1 and
  attribute_size. Functional execution accepts borrowed parameter tensors.
- Relations have one identity-bound prepared contract and explicit device
  realization. Do not add another attention cache/preparation system.
- Attention defaults to differentiable ATen; existing staged CUDA CSR kernels
  are explicit opt-in, first-order only. Empty support is zero, duplicates stay.
- Queries/readouts use concrete decoder and selector types, no required names
  or tag declarations. Multiple compatible queries share one base evaluation.
- `nn::Prediction` is optional boundary metadata, not a mandatory tensor wrapper.

Native CPU references use FP32; CUDA adapters support matching FP16/FP32 storage
and existing FP32 accumulation. Casts are explicit. No numerical kernel rewrites.
Use ordinary Torch optimizer/autograd; old guarded native-owner paths remain
specialized and are not a universal NN policy.

## Work ownership

All owners share the checkout. Preserve others' edits and communicate interface
changes directly; never revert concurrent work. No commits/pushes requested.

- Core owner: nn types, parameters, conventional layers, feature encoder.
- Biology owner: native feature bindings, alignment, prepared relation/order
  structures and device preparation; coordinate attention signatures with root.
- Query owner: typed readouts/query/conditioning, feature readout, supervision.
- Dynamics owner: generic native Cell and explicit Dynamical* migration.
- Programs owner: typed overlapping programs and gate-logit conditioner.
- Derivative principal: Product2 + grouped-linear + staged CSR-attention autograd
  adapters, GPU qualification workspace and receipts. No numerical kernel edits.
- Build integrator: CMake NN component/export, optional Python functional binding.
- IO owner: biological manifests and numeric-slot parameter checkpoints.
- Modules owner: optional host import surface and real consumer qualification.
- Root: attention functional implementation, cross-workstream API integration,
  Python integration after parallel owner settles, examples, docs and acceptance.

## Preservation and evidence

The baseline is preservation.json. Existing Python biological files/examples
are actively owned by a separate user agent; avoid broad rewrites. GPU execution
requires Device Scheduler admission. Build artifacts belong in /tmp; durable
qualification and scripts belong in this planning tree or scoped test sources.
The existing Project Control front door currently returns
workflow_task_already_active without an actionable handle. Do not repair or
hand-edit workflow authority; no workflow closure is claimed without a handle.
