# Compiler design and source-layout history

These documents preserve completed migration decisions, compiler ownership contracts, benchmark records, and source-layout freeze notes from the baseline `c65df803`. The files were moved into this archive without rewriting their captured contents. Their old path strings and test assertions are historical provenance; they do not describe current path ownership. The related source-layout freeze tests are preserved under `source_layout_freeze_tests/` and intentionally retain their original path literals.

The current compiler surface and owner are described by the [contributor routing guide](../../development/routing.md) and [developer reference](../../developer_reference.qmd). Verify any operation or API claim against current source and `docs/current_implementation.qmd`.

| Baseline path | Preserved record |
|---|---|
| `docs/compiler/source-layout/SOURCE_LAYOUT_V1.md` | [source_layout_contract_v1.md](source-layout/source_layout_contract_v1.md) |
| `docs/compiler/source-layout/split_umbrella_headers_and_public_component_imports.md` | [component_import_policy.md](source-layout/component_import_policy.md) |
| `docs/compiler/migration/define_cellerator_ownership_of_atom_semantics.md` | [atom_semantics_ownership.md](migration/atom_semantics_ownership.md) |
| `docs/compiler/migration/define_cellerator_ownership_of_basis_selection.md` | [basis_selection_ownership.md](migration/basis_selection_ownership.md) |
| `docs/compiler/migration/define_cellerator_ownership_of_evidence_and_proposal_dis.md` | [evidence_proposal_discovery_ownership.md](migration/evidence_proposal_discovery_ownership.md) |
| `docs/compiler/migration/define_cellerator_ownership_of_exact_certification.md` | [exact_certification_ownership.md](migration/exact_certification_ownership.md) |
| `docs/compiler/migration/define_cellerator_ownership_of_global_operation_program.md` | [global_operation_program_ir_ownership.md](migration/global_operation_program_ir_ownership.md) |
| `docs/compiler/migration/define_cellerator_ownership_of_portable_schedule_compila.md` | [portable_schedule_compilation_ownership.md](migration/portable_schedule_compilation_ownership.md) |
| `docs/compiler/migration/define_cellerator_ownership_of_superatom_promotion.md` | [superatom_promotion_ownership.md](migration/superatom_promotion_ownership.md) |
| `docs/compiler/migration/define_cellerator_ownership_of_typed_composition_and_gra.md` | [typed_composition_grammar_ownership.md](migration/typed_composition_grammar_ownership.md) |
| `docs/compiler/migration/define_cellerator_ruleset_export_consumed_by_cellshard.md` | [cellshard_ruleset_export_ownership.md](migration/cellshard_ruleset_export_ownership.md) |
| `docs/compiler/migration/define_temporary_compiler_to_cellshard_migration_adapter.md` | [compiler_cellshard_migration_adapter.md](migration/compiler_cellshard_migration_adapter.md) |
| `docs/compiler/migration/define_the_retained_cellshard_concrete_application_bound.md` | [cellshard_application_boundary.md](migration/cellshard_application_boundary.md) |
| `docs/compiler/migration/freeze_the_compiler_ownership_rehoming_contract.md` | [compiler_ownership_rehoming_contract.md](migration/compiler_ownership_rehoming_contract.md) |
| `docs/compiler/performance/freeze_compiler_benchmark_methodology.md` | [benchmark_methodology.md](performance/benchmark_methodology.md) |
| `docs/compiler/performance/freeze_the_part_one_performance_and_milestone_receipt.md` | [part_one_performance_milestone_receipt.md](performance/part_one_performance_milestone_receipt.md) |
| `docs/compiler/source-layout/define_agents_and_ownership_inheritance.md` | [agent_ownership.md](source-layout/agent_ownership.md) |
| `docs/compiler/source-layout/define_central_registry_and_generated_manifest_ownership.md` | [registry_manifest_ownership.md](source-layout/registry_manifest_ownership.md) |
| `docs/compiler/source-layout/define_generated_source_and_build_tree_boundaries.md` | [generated_source_build_boundary.md](source-layout/generated_source_build_boundary.md) |
| `docs/compiler/source-layout/freeze_compiler_executable_locations.md` | [compiler_executable_locations.md](source-layout/compiler_executable_locations.md) |
| `docs/compiler/source-layout/freeze_compiler_test_and_benchmark_locations.md` | [test_benchmark_locations.md](source-layout/test_benchmark_locations.md) |
| `docs/compiler/source-layout/freeze_standard_library_and_resource_locations.md` | [standard_library_resource_locations.md](source-layout/standard_library_resource_locations.md) |
| `docs/compiler/source-layout/freeze_the_compiler_implementation_tree.md` | [compiler_implementation_tree.md](source-layout/compiler_implementation_tree.md) |
| `docs/compiler/source-layout/freeze_the_public_compiler_header_tree.md` | [public_compiler_header_tree.md](source-layout/public_compiler_header_tree.md) |
| `docs/compiler/source-layout/freeze_the_source_layout_and_ownership_map.md` | [source_layout_receipt.md](source-layout/source_layout_receipt.md) |
