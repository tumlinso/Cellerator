cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_profile_tests LANGUAGES C CXX)

enable_testing()

get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(ce_ccp1_d03_001
    ../d03/freeze_the_profile_artifact_charter_and_name_test.cc
    "${CELLERATOR_ROOT}/src/compiler/profile/freeze_the_profile_artifact_charter_and_name.cc")
target_include_directories(ce_ccp1_d03_001 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_001 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d03_001 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d03_001 COMMAND ce_ccp1_d03_001)

add_executable(ce_ccp1_d03_002
    ../d03/implement_sectioned_binary_profile_storage_test.cc
    "${CELLERATOR_ROOT}/src/compiler/profile/freeze_the_profile_artifact_charter_and_name.cc"
    "${CELLERATOR_ROOT}/src/compiler/profile/implement_sectioned_binary_profile_storage.cc")
target_include_directories(ce_ccp1_d03_002 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_002 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d03_002 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d03_002 COMMAND ce_ccp1_d03_002)

add_executable(ce_ccp1_d03_003
    ../d03/define_named_profile_environments_and_alternatives_test.cc
    "${CELLERATOR_ROOT}/src/compiler/profile/define_named_profile_environments_and_alternatives.cc")
target_include_directories(ce_ccp1_d03_003 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_003 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d03_003 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d03_003 COMMAND ce_ccp1_d03_003)

add_executable(ce_ccp1_d03_004
    ../d03/represent_domain_axis_relation_and_support_evidence_test.cc
    "${CELLERATOR_ROOT}/src/compiler/profile/represent_domain_axis_relation_and_support_evidence.cc")
target_include_directories(ce_ccp1_d03_004 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_004 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d03_004 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d03_004 COMMAND ce_ccp1_d03_004)

add_executable(ce_ccp1_d03_005
    ../d03/represent_value_and_numerical_evidence_test.cc
    "${CELLERATOR_ROOT}/src/compiler/profile/represent_value_and_numerical_evidence.cc")
target_include_directories(ce_ccp1_d03_005 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_005 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d03_005 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d03_005 COMMAND ce_ccp1_d03_005)

add_executable(ce_ccp1_d03_006 ../d03/represent_mutability_recurrence_and_reuse_evidence_test.cc
    "${CELLERATOR_ROOT}/src/compiler/profile/represent_mutability_recurrence_and_reuse_evidence.cc")
target_include_directories(ce_ccp1_d03_006 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_006 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_d03_006 COMMAND ce_ccp1_d03_006)
add_executable(ce_ccp1_d03_007 ../d03/represent_evidence_provenance_and_revision_test.cc
 "${CELLERATOR_ROOT}/src/compiler/profile/represent_evidence_provenance_and_revision.cc")
target_include_directories(ce_ccp1_d03_007 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_007 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_d03_007 COMMAND ce_ccp1_d03_007)
add_executable(ce_ccp1_d03_008 ../d03/build_pointer_plus_count_profile_ingestion_apis_test.cc ingestion_c_producer.c
 "${CELLERATOR_ROOT}/src/compiler/profile/build_pointer_plus_count_profile_ingestion_apis.cc")
target_include_directories(ce_ccp1_d03_008 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_008 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_d03_008 COMMAND ce_ccp1_d03_008)
add_executable(ce_ccp1_d03_009 ../d03/implement_streaming_profile_builders_test.cc "${CELLERATOR_ROOT}/src/compiler/profile/implement_streaming_profile_builders.cc")
target_include_directories(ce_ccp1_d03_009 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_009 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_d03_009 COMMAND ce_ccp1_d03_009)
add_executable(ce_ccp1_d03_010 ../d03/implement_profile_selection_and_binding_test.cc "${CELLERATOR_ROOT}/src/compiler/profile/implement_profile_selection_and_binding.cc" "${CELLERATOR_ROOT}/src/compiler/profile/define_named_profile_environments_and_alternatives.cc")
target_include_directories(ce_ccp1_d03_010 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_010 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_d03_010 COMMAND ce_ccp1_d03_010)
add_executable(ce_ccp1_d03_011 ../d03/implement_semantic_transfer_functions_test.cc "${CELLERATOR_ROOT}/src/compiler/profile/implement_semantic_transfer_functions.cc")
target_include_directories(ce_ccp1_d03_011 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_011 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_d03_011 COMMAND ce_ccp1_d03_011)
add_executable(ce_ccp1_d03_012 ../d03/implement_bounded_branch_alternatives_and_joins_test.cc "${CELLERATOR_ROOT}/src/compiler/profile/implement_bounded_branch_alternatives_and_joins.cc")
target_include_directories(ce_ccp1_d03_012 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_012 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_d03_012 COMMAND ce_ccp1_d03_012)
add_executable(ce_ccp1_d03_013 ../d03/implement_profile_state_attachment_to_semantic_ir_test.cc "${CELLERATOR_ROOT}/src/compiler/profile/implement_profile_state_attachment_to_semantic_ir.cc")
target_include_directories(ce_ccp1_d03_013 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_013 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_d03_013 COMMAND ce_ccp1_d03_013)
add_executable(ce_ccp1_d03_014 ../d03/expose_profile_inspection_and_diff_tooling_test.cc "${CELLERATOR_ROOT}/src/compiler/profile/expose_profile_inspection_and_diff_tooling.cc")
target_include_directories(ce_ccp1_d03_014 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_014 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_d03_014 COMMAND ce_ccp1_d03_014)

add_executable(ce_ccp1_d03_015
    ../d03/deliver_the_first_profile_aware_compile_benchmark_test.cc
    "${CELLERATOR_ROOT}/src/compiler/profile/deliver_the_first_profile_aware_compile_benchmark.cc")
target_include_directories(ce_ccp1_d03_015 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_015 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d03_015 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d03_015 COMMAND ce_ccp1_d03_015)
