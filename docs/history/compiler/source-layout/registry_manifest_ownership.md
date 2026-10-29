# Central registry and generated-manifest ownership

Status: frozen for the Part One source-layout interface candidate.

Task: `CE-CCP1-A04-006`

## Integration-owned surfaces

The following singleton surfaces are edited only by their declared integration
task or lane. Leaf providers consume them or publish isolated input fragments;
they never append directly to a singleton.

| Central surface | Sole mutation owner |
| --- | --- |
| `CMakeLists.txt` and subsystem aggregators | active milestone integration lane |
| `cmake/compiler/CelleratorCompilerTargets.cmake` | compiler build-graph integration task |
| `include/Cellerator/compiler.hh` and `include/Cellerator/Cellerator.hh` | public API integration task |
| canonical grammar/token registry | frontend integration task |
| canonical CEIR dialect/operation manifest | CEIR integration task |
| canonical backend and pass registries | backend/pass integration tasks |
| `cmake/package/CelleratorConfig.cmake.in` and version template | package integration task |
| `stdlib/manifest.json` | standard-library/package integration task |
| generated compiler/resource manifests | owning integration task, output in build tree |
| `.gitmodules` and the `components/CellShard` gitlink | repository integration lane |
| `tests/CMakeLists.txt`, `bench/CMakeLists.txt`, `tools/CMakeLists.txt` | active milestone integration lane |

The run's foundation integrator is `CE-CCP1-L-INTEGRATE-FOUNDATION`; later
milestones use their declared integration lanes and tasks. Owning a central
surface does not authorize an integrator to reinterpret a frozen provider
contract.

## Isolated provider fragments

Providers publish source-linked fragments inside their own exclusive subtree:

| Provider family | Isolated fragment pattern | Central consumer |
| --- | --- | --- |
| grammar/token | `src/compiler/frontend/parser/fragments/<stable-id>.json` | grammar/token registry generator |
| CEIR dialect/operation | `src/compiler/ir/<level>/fragments/<stable-id>.json` | CEIR manifest generator |
| backend | `src/compiler/backend/<provider>/fragments/<stable-id>.json` | backend registry generator |
| pass/extension | `src/compiler/pass/fragments/<stable-id>.json` | pass registry generator |
| standard library | `stdlib/cellerator/<area>/<stable-id>.cell` | `stdlib/manifest.json` generator |
| build component | owning `src/`, `tools/`, `tests/`, or `bench/` subtree | compiler target aggregator |

Every fragment declares a stable identifier, schema version, owning task,
source path, public contract version, dependencies, and content hash. A provider
commit can therefore add or revise one fragment without editing a central file.

At integration, generators collect only declared fragments, reject duplicate
stable identifiers, missing dependencies, invalid schemas, and out-of-scope
paths, then sort by stable identifier. Generated results record all source
hashes and the generator version. The same inputs produce byte-identical
outputs. Generated parser tables, dialect manifests, backend manifests, version
headers, and embedded resource tables live in the build tree.

## Integration protocol

1. The provider validates its fragment and public contract in its isolated lane.
2. It commits and pushes that coherent result and requests integration through
   Project Control with exact source identities and gate evidence.
3. The integrator merges queued provider commits serially, resolves only within
   declared integration authority, and runs fragment uniqueness/schema gates.
4. The integrator regenerates singleton outputs, validates build/package
   consumers, commits the integrated state, and publishes any owned interface.

No provider guesses registry order, edits an umbrella, changes a package export,
or updates the CellShard gitlink merely to make its leaf task build. A rejected
fragment returns an explicit diagnostic and leaves the previous generated
manifest usable.

## Compatibility and deferred work

Existing runtime/provider registries remain authoritative until their versioned
compiler replacements integrate. This ownership contract changes no runtime or
JBC behavior and introduces no Part Two JIT or deep CellShard runtime work.

## Validation evidence

`tests/compiler/a04/define_central_registry_and_generated_manifest_ownership_test.cc`
checks every reserved singleton, each provider-fragment route, deterministic
validation rules, Project Control integration protocol, and the compatibility
boundary.
