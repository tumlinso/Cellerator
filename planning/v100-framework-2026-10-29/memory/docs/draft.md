# Event memory V100 validation and performance probe

## Contract and baseline

The CUDA `EventMemoryView<Storage>` stores channel values at per-entity anchor
times. `apply_events` lazily decays each channel between its anchor and the
sorted sparse impulses, adds impulses only to matching channels, then advances
the updated entity anchor. `read_traces` computes a later observation without
changing stored anchors. FP16 values accumulate in FP32 then round back to FP16;
FP32 stays FP32. Each apply row must name one unique valid physical entity and
its events must be time sorted; data pointers are device borrowed through stream
completion. The independent oracle is the host `TraceBank<float>` / `TraceBank<
half>` recurrence, comparing the GPU with a float64 reference and explicit
storage-rounding policy.

## Risks and unknowns

- The checked-in CUDA smoke contains host-only coverage and launches no device
  function. The main acceptance question is device correctness and race safety.
- One CTA owns each updated entity and each channel scans its event row. Small
  update counts, narrow channel counts and long event rows may underfill the
  device; benchmark representative combinations before changing ownership.
- Event order and unique update-slot preconditions are caller contracts;
  validating all device payload contents inside the kernel would add overhead.
- Device scheduler admission, V100 UUID mapping, and available Compute
  Sanitizer support are resolved by the root coordinator.

## Planned cases and commands

Use isolated SM70 builds and independent CPU recurrence inputs: empty read and
update; 1/33/257 channels; multiple entities with untouched rows; duplicate
channel impulses; multiple time-ordered impulses; rows with no events; duplicate
read slots; FP16/FP32. Compare values and anchors after apply, then compare
read output and verify reads leave state unchanged. Exercise invalid scalar
shapes/capacity through immediate API errors. Run the smoke under Device
Scheduler admission, then Compute Sanitizer memcheck and racecheck if supported.
Performance is a separate bounded sweep over entities x channels x events per
updated entity and compares resident apply/read latency against a prior source
fingerprint and the same implementation on identical inputs. Do not promote an
optimization without correctness and an admitted, uncontended V100 measurement.
