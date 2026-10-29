cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_profile_tests LANGUAGES C CXX)

enable_testing()

get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(cellerator_profile_artifact_charter_and_name_test
    ../d03/profile_artifact_charter_and_name_test.cc
    "${CELLERATOR_ROOT}/src/compiler/profile/profile_artifact_charter_and_name.cc")
target_include_directories(cellerator_profile_artifact_charter_and_name_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_profile_artifact_charter_and_name_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_profile_artifact_charter_and_name_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d03_001 ALIAS cellerator_profile_artifact_charter_and_name_test)
add_test(NAME ce_ccp1_d03_001 COMMAND cellerator_profile_artifact_charter_and_name_test)

add_executable(cellerator_sectioned_binary_profile_storage_test
    ../d03/sectioned_binary_profile_storage_test.cc
    "${CELLERATOR_ROOT}/src/compiler/profile/profile_artifact_charter_and_name.cc"
    "${CELLERATOR_ROOT}/src/compiler/profile/sectioned_binary_profile_storage.cc")
target_include_directories(cellerator_sectioned_binary_profile_storage_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_sectioned_binary_profile_storage_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_sectioned_binary_profile_storage_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d03_002 ALIAS cellerator_sectioned_binary_profile_storage_test)
add_test(NAME ce_ccp1_d03_002 COMMAND cellerator_sectioned_binary_profile_storage_test)

add_executable(cellerator_named_profile_environments_and_alternatives_test
    ../d03/named_profile_environments_and_alternatives_test.cc
    "${CELLERATOR_ROOT}/src/compiler/profile/named_profile_environments_and_alternatives.cc")
target_include_directories(cellerator_named_profile_environments_and_alternatives_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_named_profile_environments_and_alternatives_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_named_profile_environments_and_alternatives_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d03_003 ALIAS cellerator_named_profile_environments_and_alternatives_test)
add_test(NAME ce_ccp1_d03_003 COMMAND cellerator_named_profile_environments_and_alternatives_test)

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
add_executable(cellerator_streaming_profile_builders_test ../d03/streaming_profile_builders_test.cc "${CELLERATOR_ROOT}/src/compiler/profile/streaming_profile_builders.cc")
target_include_directories(cellerator_streaming_profile_builders_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_streaming_profile_builders_test PRIVATE cxx_std_17)
add_executable(ce_ccp1_d03_009 ALIAS cellerator_streaming_profile_builders_test)
add_test(NAME ce_ccp1_d03_009 COMMAND cellerator_streaming_profile_builders_test)
add_executable(cellerator_profile_selection_and_binding_test ../d03/profile_selection_and_binding_test.cc "${CELLERATOR_ROOT}/src/compiler/profile/profile_selection_and_binding.cc" "${CELLERATOR_ROOT}/src/compiler/profile/named_profile_environments_and_alternatives.cc")
target_include_directories(cellerator_profile_selection_and_binding_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_profile_selection_and_binding_test PRIVATE cxx_std_17)
add_executable(ce_ccp1_d03_010 ALIAS cellerator_profile_selection_and_binding_test)
add_test(NAME ce_ccp1_d03_010 COMMAND cellerator_profile_selection_and_binding_test)
add_executable(cellerator_semantic_transfer_functions_test ../d03/semantic_transfer_functions_test.cc "${CELLERATOR_ROOT}/src/compiler/profile/semantic_transfer_functions.cc")
target_include_directories(cellerator_semantic_transfer_functions_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_semantic_transfer_functions_test PRIVATE cxx_std_17)
add_executable(ce_ccp1_d03_011 ALIAS cellerator_semantic_transfer_functions_test)
add_test(NAME ce_ccp1_d03_011 COMMAND cellerator_semantic_transfer_functions_test)
add_executable(cellerator_bounded_branch_alternatives_and_joins_test ../d03/bounded_branch_alternatives_and_joins_test.cc "${CELLERATOR_ROOT}/src/compiler/profile/bounded_branch_alternatives_and_joins.cc")
target_include_directories(cellerator_bounded_branch_alternatives_and_joins_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_bounded_branch_alternatives_and_joins_test PRIVATE cxx_std_17)
add_executable(ce_ccp1_d03_012 ALIAS cellerator_bounded_branch_alternatives_and_joins_test)
add_test(NAME ce_ccp1_d03_012 COMMAND cellerator_bounded_branch_alternatives_and_joins_test)
add_executable(cellerator_profile_state_attachment_to_semantic_ir_test ../d03/profile_state_attachment_to_semantic_ir_test.cc "${CELLERATOR_ROOT}/src/compiler/profile/profile_state_attachment_to_semantic_ir.cc")
target_include_directories(cellerator_profile_state_attachment_to_semantic_ir_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_profile_state_attachment_to_semantic_ir_test PRIVATE cxx_std_17)
add_executable(ce_ccp1_d03_013 ALIAS cellerator_profile_state_attachment_to_semantic_ir_test)
add_test(NAME ce_ccp1_d03_013 COMMAND cellerator_profile_state_attachment_to_semantic_ir_test)
add_executable(ce_ccp1_d03_014 ../d03/expose_profile_inspection_and_diff_tooling_test.cc "${CELLERATOR_ROOT}/src/compiler/profile/expose_profile_inspection_and_diff_tooling.cc")
target_include_directories(ce_ccp1_d03_014 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d03_014 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_d03_014 COMMAND ce_ccp1_d03_014)

add_executable(cellerator_first_profile_aware_compile_benchmark_test
    ../d03/first_profile_aware_compile_benchmark_test.cc
    "${CELLERATOR_ROOT}/src/compiler/profile/first_profile_aware_compile_benchmark.cc")
target_include_directories(cellerator_first_profile_aware_compile_benchmark_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_first_profile_aware_compile_benchmark_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_first_profile_aware_compile_benchmark_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d03_015 ALIAS cellerator_first_profile_aware_compile_benchmark_test)
add_test(NAME ce_ccp1_d03_015 COMMAND cellerator_first_profile_aware_compile_benchmark_test)
