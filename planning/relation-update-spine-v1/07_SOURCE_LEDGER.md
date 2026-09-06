# Source ledger and provenance

## Authority order and limits

The supplied comprehensive architectural review remains the design basis. It is preserved byte-for-byte in `basis/architecture-review.html`. Its b3340736 snapshot predates Semantic Spine; it is not used to deny the completed c2830e34 semantic unification. The user's subsequent decisions refine implementation scope and authorize package/demo construction only. Current source evidence explains reusable mechanics and gaps; source organization remains revisable.

Live Cellerator reads used clean `main`, commit `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`, worktree `wt-d3fea47b1244da16`. Observed source/workflow identity did not change across the recorded review. Todo revision and workflow revision were both 6877. Final read at 2026-09-06 15:14:48 UTC still showed clean source and the same authority. This package was created in a separate artifact filesystem, not in the observed repository. No source, Todo, run, worktree or pending patch was mutated.

The source ledger is a curated record of the paths, ranges, file identities and facts returned by Project Control, not a fabricated raw tool transcript. A path not read is not claimed to have been audited. No V100 build, sanitizer run or kernel benchmark was performed here. Existing SS1 acceptance is baseline evidence to revalidate in I01, not new RU1 evidence.

## Current implementation evidence

### S01 · `include/Cellerator/compute/operation/relation_semantics.hh`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-66.

Canonical relation contract; f16 values and f32 dense arithmetic. Runtime identity is not mathematical equivalence.

Source file identity: `17a8608cf7d63f86ae1f83b0f1c04d8076bc31f0f5a8723ef84c67a03782e044`.

### S02 · `include/Cellerator/compute/operation/prepared_relation.hh`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-78.

Existing pair, 32-bit CSR host indices, f32 views, enqueue-generation reporting, single-stream ownership. Read again at final observation.

Source file identity: `1db1b18b805abf69bcf7d653b415aa27da66f2828b6ca6e587d3e8f8b6afb194`.

### S03 · `src/compute/operation/prepared_relation.cu`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-179.

Current N1 adapter restrictions, independent projections and physical values, rejection and cleanup behavior.

Source file identity: `c66189c4f111ce369ef6a4b633f69b9f9d5f8b626df8e7ad9063987bc35bce98`.

### S04 · `src/compute/training/native_training_slice.cu`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-180.

Useful N16 forward/transpose code; additional epilogue/training mechanics are not automatically promoted.

Source file identity: `987fe7a3ce4fcd32fe3a526741695a1678cdd398c1d4d731b56244e21c1eed96`.

### S05 · `src/compute/candidate/feature_major_small_n_candidate.cu`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-140.

FMP1 small-N candidate already supports N16 and f32 state inputs.

Source file identity: `e7051174262420f6f282d44e56c97d6fe27d199643ca3aa16129527ef61a2ae2`.

### S06 · `include/Cellerator/compute/architecture/providers/nvidia/sm70/contract/contract_portfolio_v1.cuh`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-109.

Sparse/rectangular contraction operands are half; physical/logical edge output descriptions.

Source file identity: `8349114967deba38871fddde6b0d0a3e274f7952f337742bfa5b7441255e4d68`.

### S07 · `src/compute/architecture/providers/nvidia/sm70/contract/rectangular_mma_contract_v1.cu`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-94.

Actual WMMA prefix plus scalar tail; K reused as ldm and missing sufficient stride/alignment/capacity validation.

Source file identity: `618a053c6e6ef2b886525bedee5ebd32e97f25f9ac03eab348411495cf6fdc7e`.

### S08 · `src/runtime/value_readiness.cu`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-121.

Reusable event, monotonic readiness and generation/device matching; does not itself retire external readers.

Source file identity: `68bb6d814128a7d1f2c66fea0c925ed86280980c94431a422f4fcd3ab62e63b6`.

### S09 · `include/Cellerator/compiler/sema/relation_spine_bridge.hh`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-49.

Actual bounded source-origin bridge, not full source-to-device driver.

Source file identity: `3d86d80bf3c56efd4ace7d64d75b271a3c99c91dbb5a1f72e1a9595351708fec`.

### S10 · `include/Cellerator/compiler/ir/semantic/implement_gradient_and_publication_operations_v1.hh`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-71.

Older calculus/publication vocabulary still includes a training_program_v2 dependency.

Source file identity: `e28f823ad017bfd9445ac6e55cb353b170e7ddedeb629034ad9c7d819cb24052`.

### S11 · `src/compiler/ir/semantic/implement_gradient_and_publication_operations.cc`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-87.

Older mappings require reconciliation; flags are not sufficient proof of ordered mutation/publication effects.

Source file identity: `0740541dea09a4b568dd0c3a75b668df6595b463c9a21f8d0f5365df1cf0765f`.

### S12 · `include/Cellerator/compiler/ir/semantic/implement_relation_apply_and_transpose_operations_v1.hh`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-83.

Native canonical descriptor is already retained by compiler-originated relation lowering.

Source file identity: `83520e94e8aa0cce8cb56a5d892cfca59f88e5e3005733b338eb285dd5f28e19`.

### S13 · `examples/semantic_spine_v1/regulatory_reuse.cc`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 60-179.

Existing N1 demonstration and counters; source was read, no GPU rerun in this package turn.

Source file identity: `33b5de0658995c5d942fc8228a1c195ce347215512bbac47d7055ab6fc6d31e9`.

### S14 · `include/Cellerator/execution/identity.hh`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-130.

Persistent biological IDs, axis records and generation representation.

Source file identity: `4d71b98bb9a641053a5a8bcc72c369fd53cf27984160d5de5254b9cd4101f8ae`.

### S15 · `AGENTS.md`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-153.

Canonical workflow, source authority, repository implementation conventions.

Source file identity: `a27cc5f1c6eefdf5fab576be57b285a6982ab4ba83ddec81d7fb16c3c8ffbc90`.

### P01 · `planning/cellerator-compiler-preledger-v1/21_MANUAL_TODO_BOOTSTRAP.md`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-170.

CCP bootstrap separation, additive ingestion and later activation; do not use generic compile to discard native workflow.

Source file identity: `1d6d271dc60faf966843b49de82f8c2260ce2535825de0001e381b839593fc03`.

### P02 · `planning/jbc-preledger-v1/03_CELLERATOR_ATOMIZED_IMPLEMENTATION_PLAN.md`

Repository `cellerator` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`; inspected lines 1-110.

JBC task/evidence granularity and Git/Todo cursor distinctions, not its historical task count as a quota.

Source file identity: `a1c0bd3d14688df43b43e0b77ba5d5066ab114c81204a82fac28e23b0eef0cde`.

### PC01 · `src/project_control/cli.py`

Repository `project-control` at `5866689e425056b3171f7fc13b2b9d20a9494a1e`; inspected lines 1-120;181-224.

CLI native validate/apply; apply builds a fresh snapshot rather than a saved approval cursor.

Source file identity: `82e577afee96158f156cdbe41cfa00d758feccdfd2d6f43678377291c77ed70a`.

### PC02 · `src/project_control/mutation.py`

Repository `project-control` at `5866689e425056b3171f7fc13b2b9d20a9494a1e`; inspected lines 1-346.

Verified validate_native_plan and reviewed-precondition apply_proposal; transaction checks expected revision.

Source file identity: `477745bd7b6a8bfe9ac05b044b5149fb53b8135a7492e0af22cbe25196ab9c2d`.

### PC03 · `src/project_control/services/planning.py`

Repository `project-control` at `5866689e425056b3171f7fc13b2b9d20a9494a1e`; inspected lines 145-225.

Read-only full native plan validate/diff and mutation guard. Observer hashes pretty JSON without newline.

Source file identity: `462492200fc78545f9f76ed77fee9c407672261e5a340c7c7f8d64b3b0ea6aef`.

### PC04 · `src/project_control/models.py`

Repository `project-control` at `5866689e425056b3171f7fc13b2b9d20a9494a1e`; inspected lines 40-165.

ObservationPreconditions and ProposalEnvelope.create signatures, inert authority flag and deterministic envelope validation.

Source file identity: `147a56b895e4668bc54f4963b756b6389b2cce915609d52edb5b44c2c2ee4a98`.

## Bootstrap precedents actually used

The nearest native-schema precedent is `planning/semantic-spine-v1/`: its README, compile_plan.py, validate_package.py, todo_bootstrap.py, native schema-3 plan portions, lanes, barriers and tooling contract were reviewed. Its rich task catalog, flattened CSV, dependency/checkpoint/interface/lane/workstream projections, acceptance/delivery metadata and single native apply input are retained here. Its task-count cap is not copied. This package is sized by 34 meaningful leaves, not by the predecessor's count.

CCP's manual-bootstrap guide and plan-key compatibility record distinguish format inspection from actual native validation. JBC's atomized plan provides task/evidence and ownership precedent. Not every historical machine file was re-audited. This package does not import historical active states, existing run IDs, completion marks, old cross-repository dependencies or campaign-specific kernel claims.

The current Project Control mutation path was also inspected. The guarded helper deliberately uses the reviewed observation preconditions at apply time, rather than silently approving a new snapshot through a second CLI call. A live native read-only validation of the final 35-record plan passed on 6 September 2026; see `evidence/native_plan_validation.json`. That selected-field receipt cannot serve as the later installed-package approval receipt.

## Fact, choice and hypothesis

The descriptor/provider type mismatch, existing event component, N16 code and native bootstrap mechanics above are source facts. The half-rounded VJP policy, bounded reader lease, exact gathered rectangular cover and lane ownership are design choices. Their profitability and scalability are performance hypotheses. Their actual implementation and sm70 correctness remain required future acceptance, not package-delivery claims.

### Final ownership recheck

`src/compute/training/native_training_slice.cu:1-26` was re-read at the final snapshot, including its actual header include. `include/Cellerator/compute/training/native_training_slice.hh:1-12` confirms the on-path ownership target and readiness/projection dependencies. Header source file identity: `7e4ad3fbfcca030f4a40a4f49549f6e77f7d8359c4698afe730964e24e5bba5a`. The historical tiny training timings in the source comment were not rerun or promoted as current evidence.
