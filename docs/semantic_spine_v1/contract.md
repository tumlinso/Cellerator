# Relation mathematical contract

The value-owned descriptor reuses persistent biological identities and 64-bit
logical extents. Forward consumes source and produces destination; transpose
consumes destination and produces source, preserving the same logical edges.
For edges (s,d), forward sums W[e]*X[s,n] into Y[d,n]; transpose sums
W[e]*U[d,n] into Z[s,n]. Distinct edges with equal endpoints remain additive.
Empty support yields zero under overwrite and leaves output under accumulate.

No source provenance, stream, physical candidate, pointer or value generation
participates in mathematical equivalence. Copies own every mathematical field.
Comparison is fieldwise including all 128 identity bits and axis record metadata.
Preparation keys must separately bind the complete descriptor; runtime publication
is not a topology change. No serialized byte-copy/hash of native padding is valid.

Arithmetic is round-to-nearest with separately declared FMA/reassociation
permissions, never a test tolerance. Valid wider/different-precision requests do
not imply a matching provider. Affine update lacks coefficients and is rejected.
Alias permission describes semantics; actual overlap must be checked at binding.
All callers validate before using directional axis helpers.

C01 validation: g++ -std=c++17 -Wall -Wextra -Werror -pedantic -Iinclude
/tmp/ce-ss1-c01.cc -o /tmp/ce-ss1-c01; /tmp/ce-ss1-c01, exit 0.
The compiled assertion fixture uses a 4-source/5-destination descriptor, tests
both directional axes and a copied descriptor, and statically checks trivial
copyability and standard layout. This is host contract evidence only.
