cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_profile_tests LANGUAGES CXX)

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
