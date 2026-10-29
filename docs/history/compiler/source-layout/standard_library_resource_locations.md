# Standard-library and resource location freeze

Status: frozen for the Part One source-layout interface candidate.

Task: `CE-CCP1-A04-004`

## Source locations

Checked-in Cellerator language sources and reference data use these roots:

```text
stdlib/
    manifest.json
    cellerator/
        core.cell
        domain/
        relation/
        operation/
        profile/
        planning/
        reflection/
        ir/

profiles/
    reference/
        manifest.json
        schemas/
```

All standard-library implementation units use the `.cell` suffix. The
top-level manifests provide stable logical resource names, schema/version
identity, content hashes, and relative paths; they contain no process-specific
absolute path.

`profiles/reference/` contains deliberately portable, low-performance
reference profiles and their validation schemas. A reference profile is a
correctness and bootstrapping fallback, not measured evidence for promotion on
a particular accelerator.

## Installed resources

Installations place the corresponding resources relative to the package data
directory:

```text
${CMAKE_INSTALL_DATADIR}/cellerator/stdlib/
${CMAKE_INSTALL_DATADIR}/cellerator/profiles/reference/
${CMAKE_INSTALL_DATADIR}/cellerator/schemas/
```

The spelling is compatible with the conventional `share/cellerator/` prefix,
but no contract assumes that `share` is absolute or adjacent to an executable.
Build-tree tests point the same logical resource locator at the source roots.
Installed package discovery resolves paths from package configuration and the
install prefix. Explicit caller overrides are accepted through the shared
resource-locator API and remain testable.

## Path is not semantics

Language and CEIR objects record stable logical resource identity, version, and
content hash. They do not record an installation path as the meaning of a
standard-library unit, profile, schema, or target. Moving an installation does
not change source semantics or invalidate an artifact whose identified content
is unchanged.

The parser, Sema, planner, and backend never open a hard-coded repository or
installation path. Resource discovery occurs before semantic use and produces
an explicit resolved resource plus provenance. Missing, ambiguous, stale, or
hash-mismatched resources produce diagnostics; the compiler does not silently
substitute a different profile.

Data-derived representative profiles remain distinct from checked-in reference
profiles. Reference resources cannot be treated as measured hardware or
biological evidence merely because they were found in the default data path.

## Ownership and packaging

The standard library is Cellerator-owned source and ships with libCellerator.
Schemas and reference profiles are installed resources, not C++ headers.
CellShard may transport or consume compiler artifacts but does not own or
reinterpret these resources. Central build/package tasks own installation and
generated resource manifests; leaf tasks do not edit package aggregators.

## Compatibility and deferred work

This location freeze creates no resources and changes no current runtime or JBC
behavior. It introduces no fixed host path, executable-relative lookup,
Part Two JIT, or deep CellShard runtime dependency.

## Validation evidence

`tests/compiler/a04/freeze_standard_library_and_resource_locations_test.cc`
checks the source and install layouts, relocatable tokens, logical identity and
hash rule, diagnostics, reference-profile status, ownership, and absence of
home- or repository-specific absolute paths.
