# CE-LIVE executable program v1

`executable_program` is the host-side bridge from validated biological
identity and activated projections to one planner-selected prepared operation.
It composes the existing operation core, built-in catalog, complete-cost
planner, preparation factory, execution session, launch bindings, and value
readiness contract. It is not a runtime or a second planner.

## Compile and prepare

The compile request carries both hot axis handles and their persistent domain,
order, geometry, and partition identities. Structure identity and epoch are
checked independently of pointers. The request enumerates non-owning typed
projection references and complete phase-cost descriptions. The program
filters the immutable built-in catalog by operation, numeric tuple, width,
projection kind/schema/variant, and the available complete-cost record before
calling `plan_end_to_end`.

The planner remains authoritative. Analytical, measured, and cached selection
sources remain observable, as do all considered candidates, their projection
keys, diagnostics, expected complete cost, and the selected conventional/native
status. The selected catalog entry is prepared exactly once through the typed
preparation factory. Candidate state is caller/session-owner storage registered
in the sole execution session plan cache; the program allocates nothing and
owns no projection or CPE2 bytes.

## Run

`run_executable_program` accepts changing launch bindings, an explicit expected
structure epoch and value generation, the value-readiness record, and the
caller stream. Same-stream readiness is a no-op; cross-stream visibility uses
the existing event wait. The run path performs no allocation, device selection,
descriptor construction, structural hashing, host wait, device-wide
synchronization, transfer, conversion, or canonicalization.

Changing dense, output, value, workspace, or stream pointers does not alter the
prepared program. The result reports the actual candidate, physical projection,
planner selection source, expected complete cost, structure epoch, consumed
generation, caller completion stream, and the prepared output-order contracts.
Noncanonical execution order therefore remains visible; canonical recovery is
an explicit, separately costed consumer request.

## Boundaries

- Forward relations retain feature/gene source to row/cell destination.
- Transpose/backward remains a distinct typed projection over shared logical
  edge identity and explicit value-position maps.
- CPE2 stays pointer-free; activated device views are non-owning runtime state.
- The execution session remains the only CUDA resource owner.
- Missing optional empirical infrastructure does not prevent the supported
  analytical route; empirical-required candidates still obey planner policy.
- Tensor Core activation, quantitative evidence, training integration, and
  CelleraTorch remain later CE-LIVE tasks.
