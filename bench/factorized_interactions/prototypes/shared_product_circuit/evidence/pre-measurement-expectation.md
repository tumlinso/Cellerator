# Root's expectations before measurement

This is a hypothesis record, written before any GPU attempt in this prototype.
There is no claimed speedup or numeric performance target.

Factoring can save repeated common-factor multiplications and accumulate the
shared inputs' adjoints once. The matched unfactored control is allowed normal
compiler optimization; repeated source expressions alone do not prove repeated
machine work. Both arms have the same ownership, launch phases, gate/slope
computation and compensated bank reductions. Those shared phases may dominate
the complete operation, making a useful local saving immaterial overall.

The first inference will use the paired full resident forward/VJP host-block
times. Separate event diagnostics can locate a possible effect but cannot
replace that metric. A circuit win over the matched control establishes only
the benefit of this factorization in this fixed motif and mapping. A faster
existing native reference would limit its practical value even if the causal
comparison favours factoring. A loss or an inconclusive result is a valid
experimental outcome; neither justifies a performance-driven revision before
the user discussion.

Check storage-quantized oracle errors, actual candidate/backend, scratch and
launch consistency, block order, native reference rotation, timing scope,
resource interference evidence and cleanup before interpreting a ratio.
Report the seven raw pairs and their variability. Any confidence interval is
exploratory for this one measured case, not production qualification or broad
evidence about packet layouts, biological modeling, or optimizer steps.
