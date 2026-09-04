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
