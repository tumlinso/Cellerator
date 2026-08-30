# CE-GEO public-contract examples

`relation_algebra_examples.cc` expresses six biological mappings using only
public Cellerator contracts: sparse state embedding, regulatory propagation,
transition/transport, hierarchy incidence, multimodal relation bundles, and
perturbation-delta propagation.

The example validates biological axis identity, relation structure identity,
numeric semantics, and operation semantics. It deliberately does not select a
kernel, load data, allocate device memory, or claim that schema-v2 relation
operations already execute. Storage and biological preprocessing remain owned
by CellShard and BioPrep respectively; framework adapters remain consumers of
the same native contracts.
