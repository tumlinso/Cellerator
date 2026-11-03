# Bounded grouping probe: qualification stop

The two original packet issues were fixed and independently reviewed: the
collector now uses the frozen AB/BA order and the paired estimator consistently
for its point estimate, bootstrap interval and practical-improvement decision.

Qualification job `e2aa5223-5c67-412b-ab1f-ac8c8e596238` ran on the selected V100.
The focused two-arm supplied-product qualifier passed; memcheck, racecheck,
initcheck and synccheck each reported zero errors. The first full-chain masked
R1 run reported numerical accuracy accepted and completed its native descriptor
check, but the packet rejected its descriptor-validation label. The native
emitter reports `exact-work-membership-and-raw-group-identity`; the adapter
expects `exact-gate-planes-no-materialized-workset` for the masked arm.

Independent review confirms this is a metadata contract mismatch. It is not a
reported numerical failure or a measured performance result. The remaining
full-chain cases did not run, so complete qualification is not established.

| Reuse horizon | Paired resident timing | Preparation cost and uncertainty |
| --- | --- | --- |
| 1 evaluation | Not run | Not measured |
| 16 evaluations | Not run | Not measured |

The agreed stop rule was applied: no retry and no timing submissions. This
continuation used **25.589826107 of 600 GPU-seconds**, with **one of three GPU
job slots consumed**. The overall study has used 2,357.710014582 of 14,400
GPU-seconds. Cleanup is verified, all probe lanes are inactive and paused,
the parent workspace remains paused, and no GPU reservation is open. The
allowance has not been reset.

Before any further execution, reconcile the adapter's expected descriptor
contract with the captured native emitter, preserving the original failed
packet and its partial evidence. No grouping-payoff conclusion follows from
this checkpoint.

Evidence: [stop receipt](../../../planning/chromatin-geometry-study-2026-11-02/grouping-probe-qualification-stop.json),
[independent review](../../../planning/chromatin-geometry-study-2026-11-02/reviews/grouping-probe-review.md),
and [retained packet](../runs/packets/e2aa5223-5c67-412b-ab1f-ac8c8e596238).
