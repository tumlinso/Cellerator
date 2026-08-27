# CelleraTorch native views

CE-LIVE-40 exposes Cellerator-owned CUDA storage as lifetime-bound Torch tensor
views. It does not transfer canonical ownership to Torch, allocate tensor
storage, copy values, change devices, synchronize, or prepare an execution
program.

`make_dense_tensor_view` consumes the frozen Cellerator `dense_tensor_view`.
`make_parameter_tensor_view` consumes a `native_parameter_descriptor`, keeping
its structure epoch, value generation, biological axes, mutability, and role
observable in adapter metadata. The factories validate rank, positive shape,
non-overlapping positive element strides, supported dtype, declared CUDA
device, actual allocation device, biological identity, and requested access.
Callers enumerate `native_parameter_view` themselves and create one alias per
descriptor; the adapter does not allocate or own a second parameter collection.

Every view requires a `native_storage_lease`: a weak alias of the native
owner's shared control block. Creation fails if that lease is empty or expired.
The Torch storage deleter captures a strong copy of the lease, releases only
that lease when the final Tensor alias dies, and never frees native storage
itself. The native owner must therefore construct the aliasing shared pointer
so its control block actually governs the backing allocation.

Torch has no read-only tensor type. A read-only view is consequently a semantic
contract recorded in `native_view_metadata`; callers must not mutate it.
Requesting read/write access to a parameter whose native descriptor is not
writable fails. Cellerator remains the sole parameter and value-generation
authority, and later autograd integration must not apply both a native update
and a Torch optimizer update.

The existing copied CPU CSR exporter in `bindings.hh` remains an explicit
debugging and interoperability path. These native views do not call it and do
not reconstruct CSR, COO, dense storage, or canonical order.

Current bounds are deliberate: CUDA device allocations only, Cellerator's
supported scalar types only, rank one through four, and non-overlapping
positive-stride layouts. Managed/peer storage, broadcast/overlapping layouts,
and unsupported numeric codes are rejected rather than copied silently.
