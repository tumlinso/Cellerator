# Planning and expert-control helpers v1

`planning.cell` provides transparent POD builders for preferences, constraints,
candidate offers, complete cost records, decompositions, and forced plans.
Unsafe selection is explicit. Helpers produce the same Planning IR fields a
user can write directly and never restrict direct CEIR access.
