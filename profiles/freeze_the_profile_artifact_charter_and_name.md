# Cellerator Profile Artifact v1

The compiler-owned profile container is named **Cellerator Profile Artifact
v1**. Its human-facing suffix is `.ceprofile`, its eight-byte magic is
`CELLPRF1`, and its initial schema version is `1`.

The format stores representative, data-derived semantic states and their
evidence. It does not store compiler policy, concrete runtime pointers, dataset
paths as language semantics, or claims that statistical evidence establishes
mathematical correctness. The fixed header is standard-layout, trivially
copyable, pointer-free, endian-marked, and explicitly versioned. Artifact and
semantic-environment identities are independent from evidence revision.

## Collision audit

The chosen magic and `.ceprofile` suffix were checked against the repository's
reserved persistent identities: `CPI1`, `CPS1`, `SCR1`, `CELLPK01`, `CELLCSG1`,
`CELLEX02`, `CEORCL1\0`, `CCECHNK1`, `CSH5`, `CSPACK`, and `CPEXEC01`.
`CELLPRF1` is distinct from each. CEIR has no frozen binary magic or mandatory
suffix at this milestone, so this charter does not reserve one for it.

Later revisions must use an adjacent versioned identity rather than changing
the meaning of `CELLPRF1`. Unknown extension sections may be skipped, but the
header identity and correctness boundary remain stable.
