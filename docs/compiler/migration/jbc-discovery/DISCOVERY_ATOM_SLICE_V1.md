# Cellerator discovery and atom compiler v1

`CE-CCP1-I20-DISCOVERY-ATOM` freezes the Cellerator-owned public contracts at
`include/Cellerator/compiler/discovery/discovery_v1.hh` and
`include/Cellerator/compiler/discovery/atom_v1.hh`. The discovery contract
collects the seven migrated provider families, exact certification, and the
profile-environment admission boundary. The atom contract collects persistent
identity, typed ports, structure/value plane separation, requirement matching,
certification indexes, and the temporary deprecated CellShard adapter.

The source-linked receipt in
`freeze_the_migrated_discovery_and_atom_compiler_slice_v1.hh` pins consumed
interfaces I02, I18, and I19, the CellShard source commit, migrated source and
fixture counts, and the compatibility retirement gate. The adapter is retained
for source compatibility even though its audited retirement preconditions are
now satisfied; removal is a separate integration decision.

Exact coverage produces certified atom Planning IR but never directly
authorizes execution. Planning and lowering remain separate consumers. Invalid
profiles, incomplete or duplicate coverage, stale generations, invalid atom
ports, and unsupported requirements continue to fail explicitly. Conventional
execution remains available downstream because discovery is proposal-only.

The `ce_ccp1_e02_018` test exercises profile admission, support-signature
discovery, exact rescan, certified atom construction, rejection of an incomplete
certificate, all cold migration receipts, and a 2,000-iteration vertical
microbenchmark. The benchmark reports wall-clock nanoseconds for regression
visibility only. It makes no promotion or hardware-performance claim; it has no
GPU work, transfer, synchronization, or runtime execution in scope.

Validation command:

```sh
ctest --test-dir build --output-on-failure -R '^ce_ccp1_e02_018$'
```
