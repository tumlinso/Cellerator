# T03 asynchronous lifetime and failure conformance

The executable `ce_nf1_t03` exercises the retained production relation pair and
`relation_value_readiness` through public headers and linked library targets.
It requires real CUDA even though the original task catalog specified a host
minimum. The controller strengthened the native gate accordingly. No fake
provider, reference readiness machine, private `.cu` inclusion, or replacement
runtime is used.

The test checks three independent boundaries:

- Preparation consumes CSR host arrays before returning. The caller overwrites
  their original addresses with a different valid topology. Device publication
  input is overwritten only after its documented borrow completes. The retained
  pair must still compute the original independent formulas `[17, 21]`; the
  alternative topology computes `[10, 26]` and is explicitly distinguishable.
- After a valid launch, six adversarial bindings cover insufficient output/input
  capacities, wrong output order, stale structure epoch, stale value generation,
  and input/output overlap. Rejection must preserve both sentinel outputs and
  accepted-launch/publication counters. A later valid launch must still execute;
  checked close must complete its output before releasing the pair.
- A real CUDA stream host callback holds a reader pending. Six independently
  forged ticket fields and a wrong stream cannot return its borrow. An accepted
  ticket return and accepted producer overwrite must still leave a completion
  event pending. Releasing the callback and observing completion must yield the
  old reader value `37` and new owner value `91`, proving the actual done edge.

An official cold event API seam injects one event-record failure after a real
kernel has changed the value from `5` to `19`. The owner must report CUDA failure,
retain generation `1` only as diagnostic metadata, reject subsequent reads and
publication as poisoned, preserve an output sentinel and caller ticket, and
permit checked cleanup. This is a deterministic publication failure, not an
intentional device-memory fault or a claim of exhaustive asynchronous fault
coverage. Arbitrary external writes through borrowed raw pointers remain caller
contract violations; the API cannot intercept such writes.

`ce_nf1_t03_memcheck` runs the same actual executable through Compute Sanitizer
with a nonzero error exit code. Launch acceptance alone qualifies neither test.
External native receipts provide final run results; this document does not
predeclare a pass.

Before M20 integration, controller-authorized production owner delivery is the
clean B workspace at `8375dea1e72ebd77bda591fe497b1ee9fd01b9aa`.
The scoped CMake project accepts `NF1_OWNER_ROOT` and links its existing target;
it defaults to the integrated repository root. `prepare.py` verifies the exact
producer commit, material source hashes, test source commit, compile database,
executable and CMake home, then repeats source checks after a clean-first build.
Baseplane tracked material is similarly pinned; unrelated untracked files are
preserved and excluded from the dependency. M20 must rebuild against the final
integrated owner and repeat both tests.

Eight build jobs bound aggregate host pressure while the V and P lanes compile.
No device is used for compilation. Device qualification uses the actual native
lease and `/tmp/nf1-20260908-v1-device-evidence.lock`. The cpp-context scan was
explicitly degraded text routing; scoped lint passed without source rewrites.
