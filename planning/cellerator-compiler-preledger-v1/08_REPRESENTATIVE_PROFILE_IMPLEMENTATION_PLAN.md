# Representative-profile implementation plan

## Profile role

Profiles are data-derived compiler evidence. They are not handwritten scheduling files.

A compilation environment may contain several named states such as baseline, activated, perturbed, or branch-conditioned variants. One semantic program references those states without cloning itself.

## Artifact model

The plan calls for a separate sectioned, memory-mappable, checksummed profile artifact with:

- domain, axis, relation, support, occupancy, degree, hierarchy, and stratum evidence;
- value ranges, distributions, zeros, nonfinite rates, moments, and update magnitude;
- structure/value/support/order mutability;
- recurrence, reuse horizon, and mutation half-life;
- confidence, sampling method, source identity, producer, revision, and validity;
- named states, alternatives, priors, joins, and unknown/widened dimensions;
- skip-unknown extension sections.

Large evidence may remain externally referenced. Small summaries may be embedded in Semantic IR.

## Ingestion boundary

Core Cellerator exposes pointer-plus-count/streaming profile builders. General HDF5, AnnData, workflow, and dataset loading remains outside the compiler.

## Data-state analysis

Known operations and native effect contracts provide transfer functions. Branch alternatives remain bounded; the analysis widens uncertain dimensions after an explicit complexity limit rather than manufacturing exponential specialized code.

## Generic reference profiles

Part One may ship explicitly selected low-performance testing profiles for a small audited set of species. They exist to make examples and CI usable, not to claim high-performance biological representativeness.

## Workstream task catalog

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-D03-001` | Freeze the profile artifact charter and name | Define a collision-free, versioned, pointer-free profile container distinct from CSG1/CPE2/CEIR. |
| `CE-CCP1-D03-002` | Implement sectioned binary profile storage | Use aligned directories, checksums, stable identities, optional compression, memory mapping, and skip-unknown sections for fast loading and extension. |
| `CE-CCP1-D03-003` | Define named profile environments and alternatives | Represent one compilation environment containing multiple named semantic states, aliases, priors/weights, branch conditions, and explicit default selection without duplicating program IR. |
| `CE-CCP1-D03-004` | Represent domain, axis, relation, and support evidence | Store extents, support counts, degree/occupancy distributions, strata, co-support summaries, ordering stability, hierarchy summaries, and confidence with exact source identities. |
| `CE-CCP1-D03-005` | Represent value and numerical evidence | Store type-relevant ranges, sparsity/zero/nonfinite rates, moments/quantiles, update magnitudes, dynamic range, and approximation risk without treating distributions as guarantees. |
| `CE-CCP1-D03-006` | Represent mutability, recurrence, and reuse evidence | Store observed structure/value/support/order change rates, mutation half-lives, reuse horizons, recurrence, field frequency, loop counts, and confidence intervals. |
| `CE-CCP1-D03-007` | Represent evidence provenance and revision | Track dataset/source identity, sampling method, time window, transformation stage, producer/tool version, confidence, revision, and validity predicates independently of semantic identity. |
| `CE-CCP1-D03-008` | Build pointer-plus-count profile ingestion APIs | Accept caller-provided relation/support/value/trace observations without owning workflow file formats. |
| `CE-CCP1-D03-009` | Implement streaming profile builders | Use count/scan/fill, sketches, bounded top-L summaries, histograms, and exact small-instance modes to derive profiles with explicit memory budgets. |
| `CE-CCP1-D03-010` | Implement profile selection and binding | Resolve command-line/build-provided profile artifacts to source symbols and fields, validate biological identities, and reject complete absence for activated semantic compilation. |
| `CE-CCP1-D03-011` | Implement semantic transfer functions | Describe how known operations and native effect contracts transform profile state: values only, support, structure, order, generation, or unknown. |
| `CE-CCP1-D03-012` | Implement bounded branch alternatives and joins | Carry per-branch states up to configured complexity, merge compatible evidence, widen uncertain dimensions, retain confidence, and avoid exponential code cloning. |
| `CE-CCP1-D03-013` | Implement profile-state attachment to Semantic IR | Attach environments and state values by stable references, not path strings; allow embedded small evidence and external large evidence. |
| `CE-CCP1-D03-014` | Expose profile inspection and diff tooling | Add cellerator subcommands/APIs to dump summaries, compare states, explain confidence, show expected mutations, and identify missing evidence. |
| `CE-CCP1-D03-015` | Deliver the first profile-aware compile benchmark | Compile the same relation field against at least two profile states and prove candidate/search inputs differ while source semantics remain identical. |
