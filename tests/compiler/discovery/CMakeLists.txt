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
