# Prepared host execution

`Cellerator::prepared_host` provides `prepared_host::scaled_tanh<float/double>`:
prepare once, then call a fixed multiply→tanh sweep repeatedly with borrowed buffers.
The two stages are actual native differential forward callbacks, executed by
`prepared_program_v2`; the façade does not evaluate numerical expressions itself.
Its compiled blocks borrow fixed internal contract storage, so the program is
noncopyable and nonmovable. Inspection exposes the native stage/dependency graph.
Direct `native_numeric::local_forward` calls remain available.

The caller supplies state, parameters, intermediate and output, all in the declared
physical axis order. `required_elements()` is the exact extent for each buffer.
The program allocates no numerical buffers and performs no host/device copies.
State and parameters may alias each other (ordered repeated reads); writable buffers
must be disjoint from reads and each other. Host execution is synchronous and admits
only a null stream. Domain/order/geometry/partition and definition/structure/epoch/
projection are fixed. Values and parameter generations remain owned by the existing
`nf1::instance_binding`, supplied by pointer on each call. No parameter registry,
optimizer, allocator, scheduler or second generic program engine is introduced.

Admission checks the whole composition before either native launch, including late
output capacity and alias failures. Unexpected provider failure is not transactional.
The serial forward-attempt counter invalidates prior saved primals when an accepted
attempt may overwrite workspace. Saved records additionally require identical live
borrowed buffers, instance owner, preparation, state/parameter generations and tie
group. They are validity records, not an owned tape or storage lease. All buffers,
instance owners and the program must outlive the record's use; callers must publish
value generations after every mutation. Concurrent calls and unreported writes are
unsupported. No new reverse sweep is implemented.

## Existing ownership retained

- `include/Cellerator/compute/operation/indexed_mechanism/training.hh` and its
  `src/compute/operation/indexed_mechanism/training.cu` remain the device parameter
  owner, prepared mechanism program, capacity reservation, stream reader/writer
  events and single-use `mechanism_tape` owner. This façade does not replace them.
- `include/Cellerator/execution/program/program_v2.h` and
  `src/execution/program/program_v2.cc` remain the native fixed stage-graph runner.
- `native_numeric/local_arithmetic` and `differential/local_arithmetic` remain the
  arithmetic and derivative providers. Only their host forward callbacks are used.
- The optional `cellerator.torch` adapter continues to use the native mechanism
  owner; this additive host façade does not change tensor/device ownership.

FP32/FP64 use the providers' nearest-even and nonfinite propagation policy. This
focused gate does not qualify CUDA streams, asynchronous leases, capture, adaptive
queues, training, higher derivatives, scientific inference or throughput. The existing
differential header requires CUDA SDK headers for host compilation; no CUDA compiler,
runtime library or GPU launch is required by this fixture.

## Validation and integration

Run from the CE worktree:

```
python3 -B tests/substrate/execution/check.py --build-dir /tmp/ce-is1-exec-host
```

The test uses unchanged native translation units, repeated state/parameter updates,
fixed graph identity, saved-primal invalidation, buffer rebinding, capacities, aliases,
wrong axis metadata, topology changes, invalid generations, stream/policy rejection,
empty extent and direct native calls. It checks physical per-coordinate output.

MERGE-A/BUILD should add `src/execution/prepared_host` after the existing
`Cellerator::local_differential` target, export `Cellerator::prepared_host`, install
`include/Cellerator/execution/prepared_host_sweep.hh`, and add this test directory to
its focused gate. The interface target inherits existing native provider requirements.
All shared build edits are intentionally returned to the integration owner.
