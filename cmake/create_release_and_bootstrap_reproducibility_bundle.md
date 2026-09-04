# Release reproducibility bundle receipt

The bundle contains a source baseline, fixtures, CEIR material, toolchain/build
metadata, tests, benchmark contracts, SDK consumers, and migration provenance.
`sha256sum -c docs/compiler/PART_ONE_REPRODUCIBILITY.sha256` verifies every
immutable input before a clean host or NVIDIA rebuild.
