# Indexed shared edge values

This isolated probe asks whether the current shared endpoint scorer should
evaluate only edge ordinals consumed by the sparse relation worksets. For each
step it forms `U = unique(P union D)`, computes the same shared-embedding value
formula for U, and differentiably scatters those values into an E-sized vector
with zeros elsewhere. The native relation API and shared trainable parameter
owner are unchanged. Both primal interactions P and derivative candidates D
participate, so derivative-only gate work retains its value gradients.

It compares complete native SIMT training steps against the full-E reference:
gate scoring, P/D union formation, value calculation, relation forward, loss,
backward, and guarded Adam publication. A matched full-vector Torch/native
VJP check runs before timing. The isolated value-refresh timings and estimated
gather/product storage are diagnostics; they do not replace the complete-step
comparison. All measurements use the frozen Torch v11 package and scaled v3
fixtures. The small fixture is a control, the event case has E=1.28M, and the
ranked-512 case is a denser counterexample.

KDA candidate registration and build preparation are local CPU steps. The
finite GPU evaluation uses `gpu3_job.json`; scheduler admission and quiescence
remain required. `run_kda.sh` supplies the frozen v11 package, invokes the KDA
experiment engine, and retains its raw stdout/stderr under `evidence/`.

The primary implementation intentionally rebuilds U on each call, to include
the union construction cost. Candidate ordinals are fixed within the prepared
workset here, but values are recomputed on every parameter generation. No
computed value graph is reused across an optimizer publication. This probe
does not yet cover packed CE-GEO execution or a multi-event accumulation window.
