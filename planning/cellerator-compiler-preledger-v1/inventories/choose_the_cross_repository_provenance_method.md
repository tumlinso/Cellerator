# Cross-repository JBC provenance method

Part One uses a **source export manifest plus target commit trailers**. A raw
copy, an unqualified “ported from CellShard” comment, or a matching filename is
not provenance. The manifest identifies immutable source objects before any
adaptation; trailers bind the resulting Cellerator commit and implementing Todo
to that manifest. Optional `git format-patch` bundles preserve a replayable
source commit series when one exists, but patch ancestry is never fabricated.

## Source export procedure

For each coherent migration unit, the applying Todo must:

1. Require a clean source worktree and resolve the source remote URL, branch,
   full 40-hex commit, source paths, file modes, and Git blob IDs. Resolve test
   and benchmark paths at the same commit and record their blob IDs too.
2. Create a source export manifest before editing the destination. Each row has
   `migration_id`, `source_repository`, `source_branch`, `source_commit`,
   `source_path`, `source_blob`, `implementing_todo`, `source_test`,
   `source_test_blob`, intended destination, and disposition.
3. Export content from the recorded commit with `git show
   <commit>:<path>`—never from an uncommitted working file. When a coherent
   source commit range should be retained, produce a path-filtered bundle with
   `git format-patch --full-index --binary --no-signature --stdout
   <base>..<tip> -- <paths>` and record its SHA-256 in the manifest.
4. Dry-run with `git apply --check` or an isolated throwaway worktree before an
   applying Todo changes its owned paths. A metadata-only planning Todo does not
   generate or apply a patch.
5. Adapt namespaces, includes, ownership, and tests only in the applying Todo's
   declared scope. Preserve original copyright/license notices. Do not imply
   that an adapted commit has Git ancestry from the source repository.
6. Commit and push the applying Todo with the mandatory trailers below. The
   target receipt records the resulting Cellerator commit and test/gate output.

## Mandatory target commit trailers

Every commit that moves or adapts implementation carries one block per source
export manifest (repeat path/test trailers as needed):

```text
JBC-Migration-ID: JBC-Dxx
Source-Repository: git@github.com:tumlinso/CellShard.git
Source-Branch: main
Source-Commit: <40-hex commit>
Source-Path: <repository-relative path>@<Git blob ID>
Source-Todo: <CellShard implementing Todo or UNKNOWN-WITH-REASON>
Implementing-Todo: <current Cellerator Project Control Todo>
Source-Test: <repository-relative evidence path>@<Git blob ID>
Provenance-Receipt: <Cellerator repository-relative receipt path>
Migration-Disposition: rehome|adapt|retain-distinct
Patch-SHA256: <64-hex digest or NOT-APPLICABLE>
```

`Source-Todo` may be unknown only when the source commit history and frozen
ledger cannot establish it; the receipt must state the search and reason. An
applying Todo may not use `DRY-RUN-NOT-APPLIED`. Trailers are part of the target
commit message and must survive integration; Git notes, local reflogs, branch
names alone, and external issue links are supplementary, not substitutes.

## Patch versus export choice

| Source shape | Method | Rule |
|---|---|---|
| One coherent source commit or range, same path/ownership semantics | Path-filtered `git format-patch` | Verify bundle digest and `git apply --check`; apply in the authorized isolated lane, then add target trailers. |
| Source must change namespace, ownership, ABI, or path | Blob-pinned source export plus adaptation | Reconstruct from `git show`, preserve tests, and review the semantic diff; do not claim a cherry-pick. |
| Multiple commits contain valuable evolution | Ordered patch bundle plus blob-pinned final manifest | Preserve commit order/authors in the bundle even if target adaptation becomes one coherent Todo commit. |
| CellShard storage/runtime code remains owned there | No code move; interface adapter only | Record `retain-distinct` or `adapt`, source identity, and boundary tests. |

## Metadata-only dry run

The dry run below covers every `JBC-D01` through `JBC-D16` migration row from
the duplicate-mechanism inventory. No patch was generated or applied. All rows
reference one immutable source envelope:

- Export envelope: `PX-CS-b9749ad3`
- Source repository: `git@github.com:tumlinso/CellShard.git`
- Source branch: `main`
- Source commit: `b9749ad3e5146a04f847533d8c6f1a54146aed20`
- Implementing Todo: `CE-CCP1-A02-009/DRY-RUN-NOT-APPLIED`
- Intended destination: assigned by the future applying Project Control Todo
- Patch SHA-256: `NOT-APPLICABLE`

<!-- PROVENANCE-ROWS-BEGIN -->
| Migration | Envelope | Source path | Source blob | Source test evidence | Test blob | Disposition |
|---|---|---|---|---|---|---|
| JBC-D01 | PX-CS-b9749ad3 | `include/CellShard/compiler/atom/persistent_identity_v1.hh` | `4d72445b24f1c70caf4dd8dadc1f9471ad4c79e2` | `tests/jbc/atom/atom_persistent_identity_v1_test.cc` | `1bb0162074a0f4e9f27e98ef09301e3c20ec3d9e` | rehome |
| JBC-D02 | PX-CS-b9749ad3 | `include/CellShard/compiler/atom/persistent_identity_v1.hh` | `4d72445b24f1c70caf4dd8dadc1f9471ad4c79e2` | `tests/jbc/atom/atom_persistent_identity_v1_test.cc` | `1bb0162074a0f4e9f27e98ef09301e3c20ec3d9e` | adapt |
| JBC-D03 | PX-CS-b9749ad3 | `include/CellShard/compiler/atom/logical_coverage_v1.hh` | `5023047fbebab24911d56aaa2d091ef14ddf9007` | `tests/jbc/atom/atom_logical_coverage_v1_test.cc` | `57def8cd1297aa5054fdfb12600d6e7a62955e13` | adapt |
| JBC-D04 | PX-CS-b9749ad3 | `include/CellShard/compiler/atom/common_atom_v1.hh` | `42690414ceefab6eaacf52e195f70ce1749336f8` | `tests/jbc/atom/common_atom_v1_test.cc` | `054ee237ba19c59b473a9264a9c158e3a94aea29` | rehome |
| JBC-D05 | PX-CS-b9749ad3 | `include/CellShard/compiler/evidence/atom_evidence_record_v1.hh` | `3fbe76843cd4dba1c7bac0c03cdf7faec9d9999c` | `tests/jbc/evidence/atom_evidence_record_v1_test.cc` | `57b84e2d1c158a1426bd858ae7d354151f6c1efb` | rehome |
| JBC-D06 | PX-CS-b9749ad3 | `include/CellShard/compiler/evidence/evidence_atlas_v1.hh` | `69a305ec3c563b5b58dec145452fe9a8dbbbd3c9` | `tests/jbc/evidence/evidence_atlas_builder_v1_test.cc` | `0ccec4cb51898cb03d90bf82c2db31710ef09345` | rehome |
| JBC-D07 | PX-CS-b9749ad3 | `include/CellShard/compiler/composition/superatom/cost.hpp` | `4e6071e99b5ddce4cbc81a4c786a85bb82269ece` | `tests/jbc/superatom/cost_test.cc` | `1b63fa0fe0985b66b05d096537b762cba737a3d9` | rehome |
| JBC-D08 | PX-CS-b9749ad3 | `include/CellShard/compiler/graph/operation_node.hh` | `4d4ded345a74ad34e7912ebc63ed4ef7dc6d2570` | `tests/jbc/global_ir/operation_node_test.cc` | `96c1b6de0886374de4e7b47b43a03e8a4fb4ef0f` | rehome |
| JBC-D09 | PX-CS-b9749ad3 | `include/CellShard/compiler/graph/graph_recipe.hh` | `18e2bacd44f035a654c3ee17bdd07f0a57370aeb` | `tests/jbc/global_ir/graph_recipe_test.cc` | `2e2f55ad6a714f18614cae8d4deb3e2d2319896d` | rehome |
| JBC-D10 | PX-CS-b9749ad3 | `include/CellShard/compiler/composition/derivation_dag_v1.hh` | `8e8649a237dab2ac9734eb98645523d30f814b52` | `tests/jbc/composition/derivation_dag_v1_test.cc` | `3f5e7746317fda761c8ae8c56a4c02805a1811eb` | rehome |
| JBC-D11 | PX-CS-b9749ad3 | `include/CellShard/compiler/partial/partial_atom_v1.hh` | `4e7f3c88176d947aeec781ae6cecba98f91b270f` | `tests/jbc/partial/partial_atom_v1_test.cc` | `7a928d5d5da0c3a822caccfbf18c54a3ae1a227d` | adapt |
| JBC-D12 | PX-CS-b9749ad3 | `include/CellShard/compiler/graph/physical_realization.hh` | `14898d643d379a2ccfbac6c88ec251174b50a09e` | `tests/jbc/global_ir/physical_realization_test.cc` | `4fcc8774be674c87f0eddf57bd55873918412d6e` | adapt |
| JBC-D13 | PX-CS-b9749ad3 | `include/CellShard/compiler/schedule/portable_artifact.hh` | `28c18ff92d345d1586933b31ee840eef47bca6f5` | `tests/jbc/global_ir/portable_artifact_test.cc` | `a0dd90b1d62254911c2ab769e674cf5386890216` | retain-distinct |
| JBC-D14 | PX-CS-b9749ad3 | `include/CellShard/compiler/schedule/distributed_certificate.hh` | `03244b2faa0d1ef26f1b88d1e783dd70e33f8595` | `tests/jbc/global_ir/distributed_certificate_test.cc` | `8031a467e346bce87eebe3d118b22d29ea348828` | rehome |
| JBC-D15 | PX-CS-b9749ad3 | `include/CellShard/compiler/certification/partial_result_compatibility_v1.hh` | `f682c506212e05e9a33f0fa4069c154a5655b72d` | `tests/jbc/certification/partial_result_compatibility_v1_test.cc` | `bb62818f76b4e7094f5270c3541c197c2cd2c41b` | rehome |
| JBC-D16 | PX-CS-b9749ad3 | `include/CellShard/compiler/partial/dependency_freshness_v1.hh` | `6def450d09ea76feecefa4f819a0a2f61ccd4b4c` | `tests/jbc/partial/dependency_freshness_v1_test.cc` | `74bf164a348ef5ca3e2829c23169000c5434755b` | adapt |
<!-- PROVENANCE-ROWS-END -->

The dry-run Todo is intentionally not an applying Todo and its metadata cannot
authorize a future move. A future task must regenerate the envelope from its
then-current authoritative source commit, name its own implementing Todo and
destination paths, run the source tests plus target gates, and emit the required
trailers in the pushed target commit.
