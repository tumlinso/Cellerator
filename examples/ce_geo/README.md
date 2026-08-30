# CE-GEO public biological examples

This host-only example constructs and validates six biological scenarios using
Cellerator's public typed-relation and execution-lifetime contracts. It does
not execute a kernel, parse or store a dataset, or depend on a framework
adapter.

- Sparse state embedding: gene-feature to latent-state relation apply.
- Regulatory propagation: directed regulator to target-gene propagation.
- Transition/transport: identity-preserving state transition.
- Hierarchy incidence: forward incidence pooling and transpose broadcast.
- Multimodal relations: distinct RNA and ATAC sources with one destination.
- Perturbation delta propagation: immutable response structure with a
  generation-checked mutable delta value plane.

Compile and run the example directly with a strict C++17 compiler:

```bash
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -Iinclude \
    examples/ce_geo/biological_relations.cc \
    -o build/ceGeoBiologicalRelations
build/ceGeoBiologicalRelations
```
