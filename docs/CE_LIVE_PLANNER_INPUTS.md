# CE-LIVE planner inputs

CE-LIVE-26 derives planner inputs from the live quantitative relation without
creating a second planner. `bench/ce_live/planner/live_planner_inputs.*` feeds
the existing end-to-end planner contracts.

The structural statistics are source count, destination count, logical edge
count, destination-degree range and mean, and density. The quantitative
statistics record the observed value generation, nonzero count, range, and L1
norm. They describe a validated input; they do not become biological claims.

## Persistent keys and reuse horizons

Persistent planning identity contains the mathematical problem, exact source
and destination domains and orders, geometry, partition, structure identity
and epoch, device performance class, runtime/build identity, and policy.
Pointer identity is never used.

Reuse remains factored into three independent horizons:

- structure reuse amortizes semantic packing;
- projection reuse amortizes projection construction and backend preparation;
- value reuse amortizes static value packing.

The current value generation and allocation address stay outside the persistent
key. They remain explicit quantitative/runtime evidence. A new structure epoch,
semantic identity, build, device class, or reuse policy invalidates the relevant
key; rebinding the same generation at another address does not.

## Complete candidate cost

`candidate_phase_input` carries the existing planner's full phase record:
host preparation, semantic packing, projection construction, backend prepare,
static value packing, transfer, dynamic packing, kernel, epilogue, output-order
transform, synchronization, communication, and return transfer, plus persistent
and transient bytes. `account_candidate_phases` applies the three reuse
horizons through the existing complete-cost function.

Analytical phase estimates are shortlist inputs only. They are not promotion
evidence. `authoritative_for_promotion` is true only for a phase record produced
by a correct measured candidate run, preserving empirical selection as the
authority for uncertain candidates.

Focused validation:

```bash
g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -Iinclude -I. \
  tests/planner/ce_live_planner_features_test.cc \
  bench/ce_live/planner/live_planner_inputs.cc \
  src/planner/end_to_end_planner.cc \
  -o /tmp/ce-live-26-planner-features-test
/tmp/ce-live-26-planner-features-test
```
