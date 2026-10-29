cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_discovery_tests LANGUAGES CXX)

enable_testing()

get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(cellerator_common_jbc_atom_identity_adapters_test
    common_jbc_atom_identity_adapters_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/common_jbc_atom_identity_adapters.cc")
target_include_directories(cellerator_common_jbc_atom_identity_adapters_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_common_jbc_atom_identity_adapters_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_common_jbc_atom_identity_adapters_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_common_jbc_atom_identity_adapters_test COMMAND cellerator_common_jbc_atom_identity_adapters_test)

add_executable(cellerator_overlapping_evidence_atlas_core_test
    overlapping_evidence_atlas_core_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/overlapping_evidence_atlas_core.cc")
target_include_directories(cellerator_overlapping_evidence_atlas_core_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_overlapping_evidence_atlas_core_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_overlapping_evidence_atlas_core_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_overlapping_evidence_atlas_core_test COMMAND cellerator_overlapping_evidence_atlas_core_test)

add_executable(cellerator_support_signature_discovery_test
    support_signature_discovery_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/support_signature_discovery.cc")
target_include_directories(cellerator_support_signature_discovery_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_support_signature_discovery_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_support_signature_discovery_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_support_signature_discovery_test COMMAND cellerator_support_signature_discovery_test)

add_executable(cellerator_co_support_and_overlap_discovery_test
    co_support_and_overlap_discovery_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/co_support_and_overlap_discovery.cc")
target_include_directories(cellerator_co_support_and_overlap_discovery_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_co_support_and_overlap_discovery_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_co_support_and_overlap_discovery_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_co_support_and_overlap_discovery_test COMMAND cellerator_co_support_and_overlap_discovery_test)
add_executable(cellerator_relation_motif_and_operation_trace_discovery_test relation_motif_and_operation_trace_discovery_test.cc "${CELLERATOR_ROOT}/src/compiler/discovery/relation_motif_and_operation_trace_discovery.cc")
target_include_directories(cellerator_relation_motif_and_operation_trace_discovery_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_relation_motif_and_operation_trace_discovery_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_relation_motif_and_operation_trace_discovery_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_relation_motif_and_operation_trace_discovery_test COMMAND cellerator_relation_motif_and_operation_trace_discovery_test)

add_executable(cellerator_trajectory_and_lineage_pattern_discovery_test
    trajectory_and_lineage_pattern_discovery_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/trajectory_and_lineage_pattern_discovery.cc")
target_include_directories(cellerator_trajectory_and_lineage_pattern_discovery_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_trajectory_and_lineage_pattern_discovery_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_trajectory_and_lineage_pattern_discovery_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_trajectory_and_lineage_pattern_discovery_test COMMAND cellerator_trajectory_and_lineage_pattern_discovery_test)

add_executable(cellerator_multimodal_and_identity_spine_discovery_test
    multimodal_and_identity_spine_discovery_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/multimodal_and_identity_spine_discovery.cc")
target_include_directories(cellerator_multimodal_and_identity_spine_discovery_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_multimodal_and_identity_spine_discovery_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_multimodal_and_identity_spine_discovery_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_multimodal_and_identity_spine_discovery_test COMMAND cellerator_multimodal_and_identity_spine_discovery_test)

add_executable(cellerator_factor_bicluster_and_signature_proposal_strategy_test
    factor_bicluster_and_signature_proposal_strategy_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/factor_bicluster_and_signature_proposal_strategy.cc")
target_include_directories(cellerator_factor_bicluster_and_signature_proposal_strategy_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_factor_bicluster_and_signature_proposal_strategy_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_factor_bicluster_and_signature_proposal_strategy_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_factor_bicluster_and_signature_proposal_strategy_test COMMAND cellerator_factor_bicluster_and_signature_proposal_strategy_test)

add_executable(cellerator_exact_rescan_and_proposal_certification_test
    exact_rescan_and_proposal_certification_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/exact_rescan_and_proposal_certification.cc")
target_include_directories(cellerator_exact_rescan_and_proposal_certification_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_exact_rescan_and_proposal_certification_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_exact_rescan_and_proposal_certification_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_exact_rescan_and_proposal_certification_test COMMAND cellerator_exact_rescan_and_proposal_certification_test)

add_executable(cellerator_atom_envelope_and_typed_ports_test
    atom_envelope_and_typed_ports_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/atom_envelope_and_typed_ports.cc"
    "${CELLERATOR_ROOT}/src/compiler/discovery/common_jbc_atom_identity_adapters.cc")
target_include_directories(cellerator_atom_envelope_and_typed_ports_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_atom_envelope_and_typed_ports_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_atom_envelope_and_typed_ports_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_atom_envelope_and_typed_ports_test COMMAND cellerator_atom_envelope_and_typed_ports_test)

add_executable(cellerator_atom_plane_separation_test
    atom_plane_separation_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/atom_plane_separation.cc")
target_include_directories(cellerator_atom_plane_separation_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_atom_plane_separation_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_atom_plane_separation_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_atom_plane_separation_test COMMAND cellerator_atom_plane_separation_test)

add_executable(cellerator_atom_requirement_affordance_matching_test
    atom_requirement_affordance_matching_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/atom_requirement_affordance_matching.cc")
target_include_directories(cellerator_atom_requirement_affordance_matching_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_atom_requirement_affordance_matching_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_atom_requirement_affordance_matching_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_atom_requirement_affordance_matching_test COMMAND cellerator_atom_requirement_affordance_matching_test)

add_executable(cellerator_scalable_certification_indexes_test
    scalable_certification_indexes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/scalable_certification_indexes.cc")
target_include_directories(cellerator_scalable_certification_indexes_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_scalable_certification_indexes_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_scalable_certification_indexes_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_scalable_certification_indexes_test COMMAND cellerator_scalable_certification_indexes_test)

add_executable(cellerator_preserve_migration_provenance_in_source_and_artifacts_test
    preserve_migration_provenance_in_source_and_artifacts_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/preserve_migration_provenance_in_source_and_artifacts.cc")
target_include_directories(cellerator_preserve_migration_provenance_in_source_and_artifacts_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_preserve_migration_provenance_in_source_and_artifacts_test PRIVATE cxx_std_17)
target_compile_definitions(cellerator_preserve_migration_provenance_in_source_and_artifacts_test PRIVATE
    CELLERATOR_SOURCE_ROOT="${CELLERATOR_ROOT}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_preserve_migration_provenance_in_source_and_artifacts_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_preserve_migration_provenance_in_source_and_artifacts_test COMMAND cellerator_preserve_migration_provenance_in_source_and_artifacts_test)

add_executable(cellerator_temporary_cellshard_compiler_adapter_test
    temporary_cellshard_compiler_adapter_test.cc
    temporary_cellshard_compiler_adapter_old_consumer.cc
    temporary_cellshard_compiler_adapter_new_consumer.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/temporary_cellshard_compiler_adapter.cc"
    "${CELLERATOR_ROOT}/src/compiler/discovery/atom_envelope_and_typed_ports.cc"
    "${CELLERATOR_ROOT}/src/compiler/discovery/common_jbc_atom_identity_adapters.cc")
target_include_directories(cellerator_temporary_cellshard_compiler_adapter_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_temporary_cellshard_compiler_adapter_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_temporary_cellshard_compiler_adapter_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_temporary_cellshard_compiler_adapter_test COMMAND cellerator_temporary_cellshard_compiler_adapter_test)

add_executable(cellerator_port_discovery_tests_and_evidence_fixtures_test
    port_discovery_tests_and_evidence_fixtures_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/port_discovery_tests_and_evidence_fixtures.cc")
target_include_directories(cellerator_port_discovery_tests_and_evidence_fixtures_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_port_discovery_tests_and_evidence_fixtures_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_port_discovery_tests_and_evidence_fixtures_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_port_discovery_tests_and_evidence_fixtures_test COMMAND cellerator_port_discovery_tests_and_evidence_fixtures_test)

add_executable(cellerator_validate_no_compiler_discovery_remains_authoritative_in_test
    validate_no_compiler_discovery_remains_authoritative_in_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/validate_no_compiler_discovery_remains_authoritative_in.cc")
target_include_directories(cellerator_validate_no_compiler_discovery_remains_authoritative_in_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_validate_no_compiler_discovery_remains_authoritative_in_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_validate_no_compiler_discovery_remains_authoritative_in_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_validate_no_compiler_discovery_remains_authoritative_in_test COMMAND cellerator_validate_no_compiler_discovery_remains_authoritative_in_test)

add_executable(cellerator_migrated_discovery_and_atom_compiler_slice_test
    migrated_discovery_and_atom_compiler_slice_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/migrated_discovery_and_atom_compiler_slice.cc"
    "${CELLERATOR_ROOT}/src/compiler/discovery/atom_envelope_and_typed_ports.cc"
    "${CELLERATOR_ROOT}/src/compiler/discovery/exact_rescan_and_proposal_certification.cc"
    "${CELLERATOR_ROOT}/src/compiler/discovery/support_signature_discovery.cc"
    "${CELLERATOR_ROOT}/src/compiler/discovery/common_jbc_atom_identity_adapters.cc"
    "${CELLERATOR_ROOT}/src/compiler/discovery/temporary_cellshard_compiler_adapter.cc"
    "${CELLERATOR_ROOT}/src/compiler/discovery/port_discovery_tests_and_evidence_fixtures.cc"
    "${CELLERATOR_ROOT}/src/compiler/discovery/preserve_migration_provenance_in_source_and_artifacts.cc"
    "${CELLERATOR_ROOT}/src/compiler/discovery/validate_no_compiler_discovery_remains_authoritative_in.cc")
target_include_directories(cellerator_migrated_discovery_and_atom_compiler_slice_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_migrated_discovery_and_atom_compiler_slice_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_migrated_discovery_and_atom_compiler_slice_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_migrated_discovery_and_atom_compiler_slice_test COMMAND cellerator_migrated_discovery_and_atom_compiler_slice_test)
