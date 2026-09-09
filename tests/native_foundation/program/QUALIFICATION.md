# Prepared program qualification

Status: partial P01 checkpoint. Native acceptance is not complete. The typed
host path needs the pending canonical host-stream correction described below.

P01 extends `execution/program/program_v2.h` by borrowing the canonical
`prepared_binding_contract` and `launch_bindings` owners. Callbacks retain their
signature and see typed input/output/value arrays through `binding.typed`.
All stages undergo structural and dynamic validation before any callback runs.
The runner never owns an allocator, stream, device context, scheduler, or operand
storage. This is host execution qualification; no CUDA performance is claimed.

Both new pointer members are appended and default null, preserving source use of
legacy aggregate initializers and callbacks. Their addition changes C++ object
layout: rebuild producers and consumers together. These pointer-bearing structs
are transient process objects, not serialized records. Program version 2 is
retained for source compatibility, not binary compatibility with older builds.

Typed mode requires both pointers. It rejects mixed legacy payload pointers,
divergent workspace descriptors, and a stream different from the caller's.
Existing axis, structure epoch, value generation, output effects, and alias
validation remain owned by `execution/launch_bindings.hh`. This initial owner
requires at least one real bound relation structure even for dense callbacks.
Prepared effect declarations express callback obligations: validation does not
implement numerical effects on behalf of the callback or prove its behavior.
Dense views do not carry allocation bounds; no capacity claim is made here.

`ce_nf1_p01` executes three dense inputs and one value plane into two distinct
outputs, with exact integer-valued float arithmetic and overwrite/accumulate
effects. It rejects stale generations, axis/count mismatch, forbidden alias,
invalid effect, ambiguous legacy/typed payloads, missing paired contracts and
stream mismatch. An invalid final stage leaves callback counters and earlier
outputs unchanged. A callback failure after dispatch cannot roll back prior
callbacks; callers must keep all descriptors immutable during execution.

`ce_nf1_program_legacy` compiles the unchanged existing
`tests/jbc/fragment/canonical_relation_apply_smoke.cc` and its real canonical
owners, exercising fragment compilation, external value binding, numerical
execution (23, 18), and output affordance publication.

Standalone qualification:

```sh
cmake -S tests/native_foundation/program -B /tmp/ce-nf1-program -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/ce-nf1-program -j 4
ctest --test-dir /tmp/ce-nf1-program --output-on-failure --no-tests=error
```

Four build jobs bound this small host test while other native lanes compile
concurrently on the shared controller host. No GPU reservation is needed.

## Pending native scope correction

The existing `launch_bindings.hh` validator rejects host stream ordinal -1.
A narrow scope amendment prepared from CE authority revision 7153 lives at
`/tmp/nf1-p01-scope-delta/ce-p01.todo-plan.json`; its simulation changes only
P01 ownership scopes. The adjacent `host-launch.patch` admits host ordinal -1
with a null stream and host workspace, preserving the matching-device case.
It has **not** been applied to repository source. The coordinator must apply
the reviewed scope amendment and refresh the task brief before this owner edit.

Prospective testing used that header only in an external include overlay.
The real typed calculation, legacy adapter, all negative controls, and existing
execution-order contract test passed with that prospective correction. This is
review evidence, not native acceptance or qualification of this source commit.
The unchanged fragment numerical test already passes against repository source.
The source-only `ce_nf1_p01` will reject host mode until the correction lands.
No task completion, integrated capability, or GPU qualification is claimed.
