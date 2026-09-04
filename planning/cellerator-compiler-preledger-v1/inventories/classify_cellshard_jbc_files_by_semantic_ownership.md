# CellShard JBC Semantic-Ownership Classification

This receipt completes `CE-CCP1-A02-003`. It classifies every unique path in
the 24-branch inventory from `CE-CCP1-A02-002` under exactly one primary
semantic disposition. Classification is evidence for later rehoming; it does
not move, delete, or rename any CellShard file.

## Frozen source set

- CellShard repository: `/home/tumlinson/Cellerator/components/CellShard`
- Common JBC ancestry base:
  `7762a5925fe18b2ca45ab8a436f3461804ed2ad9`
- Branch universe: all 24 local `refs/heads/jbc/*` tips recorded by A02-002
- Unique changed paths: 979
- SHA-256 of the sorted, newline-terminated path list:
  `af783b7c35be048289a8da5798e8b11c7895846f0d42d938dc6a235e73a5aee9`

The authoritative file-level enumeration is reproducible from immutable Git
objects:

```sh
for branch in $(git for-each-ref --format='%(refname:short)' 'refs/heads/jbc/*'); do
    git diff --name-only 7762a5925fe18b2ca45ab8a436f3461804ed2ad9.."$branch"
done | sort -u
```

## Primary dispositions

| Primary disposition | Paths | Sorted-list SHA-256 | Ownership reading |
|---|---:|---|---|
| compiler discovery | 82 | `7e6b20d0259b1a08e56f25202e7ade2ea550054188ee0112ce0308e8c563f8ef` | Candidate discovery mechanisms and their biological evidence inputs; future compiler ownership. |
| exact certification | 16 | `29f4780feea4ff8f859f26807579b811ccb475e359d4408e2995002d0eadd07e` | Exact identity, coverage, compatibility, and independent-verification contracts; future compiler ownership. |
| atom semantics | 38 | `7c89fd400d69df4f0b2c89c6658eaf1dc7123526f02eb43ce76953eb44b32bf2` | Logical atom and partial-result meaning, not persistence; future compiler/CEIR ownership. |
| grammar/composition | 51 | `d0581be9fc8ea5b455ae918c27eff7d815c660d3dcb65d244f228d0b2527b17d` | Grammar, derivation, composition, and superatom semantics; future compiler ownership. |
| basis | 17 | `cfa4d3cd38ca9475a446acc133c6d50eb4e20fa37f56f8df25c948fe5462b331` | Basis selection and promotion policy; future compiler ownership. |
| global program/schedule | 19 | `27ae6b5d3fdd437aa7d9e946d366734534e3bcadea6098bcc91231960ca4bbab` | Global graph, program, and schedule descriptions; future CEIR/planner ownership. |
| concrete storage | 43 | `af34cffe53b13fa9c440b8bbd3d71bb17283d014ebaf2d8fd8add6abca5a7b6f` | Atom-store bytes, publication, recovery, codecs, and storage specification; remains CellShard-owned. |
| concrete materialization | 13 | `171c04d0db32782b12e7c3c79180381a24be0319cbd9ba5dea0c4046d22bb51d` | Concrete read sources, read plans, action IR, recovery, and worker execution; remains a CellShard runtime consumer. |
| transport/residency | 24 | `08a7133ac6995396366d19609ccd7d875aa1e1e02cd4f5fe0a44b423277cb627` | Node, topology, residency, staging, and transport mechanisms; remains CellShard-owned. |
| bridge | 3 | `11565f6a69c42d7d733d7c161f2fc0596812cfa0c738cc881bb535a13c3372f7` | Umbrella/build/interop seams that must be replaced or narrowed during integration. |
| test | 328 | `76c5964efb797cf6ecee3dc26763f77d6ce4d352240a74bbbf4968a5e6401126` | JBC tests; preserved and rehomed with the contract they validate. |
| evidence | 345 | `16c07f8fe4edf7844e34281a439d7f98433f40f093f600c57381a04c1b14e4a1` | Benchmarks, evidence documents, compiler evidence records, and historical Todo projections. |

The counts sum to 979. No path is unclassified and no path has two primary
dispositions.

## Deterministic classification rules

Rules are mutually exclusive and evaluated against the full repository-relative
path. The accompanying test rebuilds the source set from Git and requires every
path to match exactly one rule.

| Path rule | Primary disposition |
|---|---|
| `include/CellShard/compiler/discovery/**` | compiler discovery |
| `include/CellShard/compiler/certification/**` | exact certification |
| `include/CellShard/compiler/atom/**`, `src/compiler/atom/**`, `include/CellShard/compiler/partial/**` | atom semantics |
| `include/CellShard/compiler/grammar/**`, `src/compiler/grammar/**`, `include/CellShard/compiler/composition/**` | grammar/composition |
| `include/CellShard/compiler/basis/**` | basis |
| `include/CellShard/compiler/graph/**`, `src/compiler/graph/**`, `include/CellShard/compiler/schedule/**`, `src/compiler/schedule/**` | global program/schedule |
| `include/CellShard/artifact/atom_store/**`, `src/artifact/atom_store/**`, `docs/SPEC_ATOM_STORE_V1.md` | concrete storage |
| runtime basenames containing `async_file_atom_source`, `atom_source.`, `command_ir`, `exact_read_baseline`, `read_plan`, `runtime_recovery`, or `worker_cuda_graph` | concrete materialization |
| all other `include/CellShard/runtime/v2/**` and `src/runtime/v2/**` | transport/residency |
| `CMakeLists.txt`, `include/CellShard/CellShard.hh`, `include/CellShard/interop/**` | bridge |
| `tests/**` | test |
| `bench/**`, `docs/JBC/evidence/**`, `include/CellShard/compiler/evidence/**`, `src/compiler/evidence/**`, `todos/**`, `.todo-orchestrator/state.snapshot.json`, `todos.md`, `todo-status.md` | evidence |

## Rehoming consequence

The classification separates 223 compiler-semantic source paths (discovery,
certification, atom semantics, grammar/composition, basis, and global
program/schedule) from 80 concrete CellShard implementation paths (storage,
materialization, and transport/residency). Tests and evidence follow their
subject during later migration. Bridge files are integration seams, not a new
authority. This preserves useful JBC work while making the Part One ownership
boundary explicit and leaves deferred Part Two runtime work in CellShard.
