cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_discovery_tests LANGUAGES CXX)

enable_testing()

get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(ce_ccp1_e02_001
    import_the_common_jbc_atom_identity_adapters_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_the_common_jbc_atom_identity_adapters.cc")
target_include_directories(ce_ccp1_e02_001 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_001 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_001 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_001 COMMAND ce_ccp1_e02_001)

add_executable(ce_ccp1_e02_002
    import_the_overlapping_evidence_atlas_core_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_the_overlapping_evidence_atlas_core.cc")
target_include_directories(ce_ccp1_e02_002 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_002 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_002 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_002 COMMAND ce_ccp1_e02_002)

add_executable(ce_ccp1_e02_003
    import_support_signature_discovery_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_support_signature_discovery.cc")
target_include_directories(ce_ccp1_e02_003 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_003 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_003 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_003 COMMAND ce_ccp1_e02_003)

add_executable(ce_ccp1_e02_004
    import_co_support_and_overlap_discovery_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_co_support_and_overlap_discovery.cc")
target_include_directories(ce_ccp1_e02_004 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_004 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_004 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_004 COMMAND ce_ccp1_e02_004)
add_executable(ce_ccp1_e02_005 import_relation_motif_and_operation_trace_discovery_test.cc "${CELLERATOR_ROOT}/src/compiler/discovery/import_relation_motif_and_operation_trace_discovery.cc")
target_include_directories(ce_ccp1_e02_005 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_005 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_005 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_005 COMMAND ce_ccp1_e02_005)

add_executable(ce_ccp1_e02_006
    import_trajectory_and_lineage_pattern_discovery_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_trajectory_and_lineage_pattern_discovery.cc")
target_include_directories(ce_ccp1_e02_006 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_006 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_006 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_006 COMMAND ce_ccp1_e02_006)

add_executable(ce_ccp1_e02_007
    import_multimodal_and_identity_spine_discovery_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_multimodal_and_identity_spine_discovery.cc")
target_include_directories(ce_ccp1_e02_007 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_007 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_007 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_007 COMMAND ce_ccp1_e02_007)

add_executable(ce_ccp1_e02_008
    import_factor_bicluster_and_signature_proposal_strategie_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_factor_bicluster_and_signature_proposal_strategie.cc")
target_include_directories(ce_ccp1_e02_008 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_008 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_008 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_008 COMMAND ce_ccp1_e02_008)

add_executable(ce_ccp1_e02_009
    import_exact_rescan_and_proposal_certification_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_exact_rescan_and_proposal_certification.cc")
target_include_directories(ce_ccp1_e02_009 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_009 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_009 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_009 COMMAND ce_ccp1_e02_009)

add_executable(ce_ccp1_e02_010
    import_atom_envelope_and_typed_ports_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_atom_envelope_and_typed_ports.cc"
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_the_common_jbc_atom_identity_adapters.cc")
target_include_directories(ce_ccp1_e02_010 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_010 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_010 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_010 COMMAND ce_ccp1_e02_010)

add_executable(ce_ccp1_e02_011
    import_atom_plane_separation_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_atom_plane_separation.cc")
target_include_directories(ce_ccp1_e02_011 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_011 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_011 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_011 COMMAND ce_ccp1_e02_011)

add_executable(ce_ccp1_e02_012
    import_atom_requirement_affordance_matching_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_atom_requirement_affordance_matching.cc")
target_include_directories(ce_ccp1_e02_012 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_012 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_012 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_012 COMMAND ce_ccp1_e02_012)

add_executable(ce_ccp1_e02_013
    import_scalable_certification_indexes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_scalable_certification_indexes.cc")
target_include_directories(ce_ccp1_e02_013 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_013 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_013 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_013 COMMAND ce_ccp1_e02_013)

add_executable(ce_ccp1_e02_014
    preserve_migration_provenance_in_source_and_artifacts_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/preserve_migration_provenance_in_source_and_artifacts.cc")
target_include_directories(ce_ccp1_e02_014 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_014 PRIVATE cxx_std_17)
target_compile_definitions(ce_ccp1_e02_014 PRIVATE
    CELLERATOR_SOURCE_ROOT="${CELLERATOR_ROOT}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_014 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_014 COMMAND ce_ccp1_e02_014)

add_executable(ce_ccp1_e02_015
    create_temporary_cellshard_compiler_compatibility_adapte_test.cc
    create_temporary_cellshard_compiler_compatibility_adapte_old_consumer.cc
    create_temporary_cellshard_compiler_compatibility_adapte_new_consumer.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/create_temporary_cellshard_compiler_compatibility_adapte.cc"
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_atom_envelope_and_typed_ports.cc"
    "${CELLERATOR_ROOT}/src/compiler/discovery/import_the_common_jbc_atom_identity_adapters.cc")
target_include_directories(ce_ccp1_e02_015 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_015 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_015 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_015 COMMAND ce_ccp1_e02_015)

add_executable(ce_ccp1_e02_016
    port_discovery_tests_and_evidence_fixtures_test.cc
    "${CELLERATOR_ROOT}/src/compiler/discovery/port_discovery_tests_and_evidence_fixtures.cc")
target_include_directories(ce_ccp1_e02_016 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e02_016 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e02_016 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e02_016 COMMAND ce_ccp1_e02_016)
