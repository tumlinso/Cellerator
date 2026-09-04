cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_cpu_backend_tests LANGUAGES CXX)

enable_testing()
get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

add_executable(ce_ccp1_f02_001
    freeze_the_backend_provider_abi_test.cc
    mock_backend_provider_v1.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/freeze_the_backend_provider_abi.cc")
target_include_directories(ce_ccp1_f02_001 PRIVATE
    "${CELLERATOR_ROOT}/include"
    "${CMAKE_CURRENT_LIST_DIR}")
target_compile_features(ce_ccp1_f02_001 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f02_001 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f02_001 COMMAND ce_ccp1_f02_001)

add_executable(ce_ccp1_f02_002
    implement_backend_registry_and_selection_test.cc
    mock_backend_provider_v1.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/freeze_the_backend_provider_abi.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/implement_backend_registry_and_selection.cc")
target_include_directories(ce_ccp1_f02_002 PRIVATE
    "${CELLERATOR_ROOT}/include"
    "${CMAKE_CURRENT_LIST_DIR}")
target_compile_features(ce_ccp1_f02_002 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f02_002 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f02_002 COMMAND ce_ccp1_f02_002)

add_executable(ce_ccp1_f02_003
    implement_backend_code_generation_plans_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/implement_backend_code_generation_plans.cc")
target_include_directories(ce_ccp1_f02_003 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f02_003 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f02_003 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f02_003 COMMAND ce_ccp1_f02_003)

add_executable(ce_ccp1_f02_004
    implement_generated_c_representation_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/implement_generated_c_representation.cc")
target_include_directories(ce_ccp1_f02_004 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f02_004 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f02_004 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f02_004 COMMAND ce_ccp1_f02_004)

add_executable(ce_ccp1_f02_005
  implement_generic_cpu_relation_apply_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/implement_generic_cpu_relation_apply.cc)
target_include_directories(ce_ccp1_f02_005 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_f02_005 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_f02_005 PRIVATE -Wall -Wextra -Werror)
add_test(NAME ce_ccp1_f02_005 COMMAND ce_ccp1_f02_005)
