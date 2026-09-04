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

add_executable(ce_ccp1_f02_006
  implement_cpu_transpose_and_contraction_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/implement_cpu_transpose_and_contraction.cc)
target_include_directories(ce_ccp1_f02_006 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_f02_006 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_f02_006 PRIVATE -Wall -Wextra -Werror)
add_test(NAME ce_ccp1_f02_006 COMMAND ce_ccp1_f02_006)

add_executable(ce_ccp1_f02_007
  implement_cpu_segment_gate_update_bundle_and_chain_paths_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/implement_cpu_segment_gate_update_bundle_and_chain_paths.cc)
target_include_directories(ce_ccp1_f02_007 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_f02_007 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_f02_007 PRIVATE -Wall -Wextra -Werror)
add_test(NAME ce_ccp1_f02_007 COMMAND ce_ccp1_f02_007)

add_executable(ce_ccp1_f02_008
  implement_cpu_projection_packing_and_order_transforms_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/implement_cpu_projection_packing_and_order_transforms.cc)
target_include_directories(ce_ccp1_f02_008 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_f02_008 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_f02_008 PRIVATE -Wall -Wextra -Werror)
add_test(NAME ce_ccp1_f02_008 COMMAND ce_ccp1_f02_008)

add_executable(ce_ccp1_f02_009
  compile_generated_c_into_ordinary_objects_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/compile_generated_c_into_ordinary_objects.cc)
target_include_directories(ce_ccp1_f02_009 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_f02_009 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_f02_009 PRIVATE -Wall -Wextra -Werror)
add_test(NAME ce_ccp1_f02_009 COMMAND ce_ccp1_f02_009)

add_executable(ce_ccp1_f02_010
  implement_host_runtime_binding_abi_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/implement_host_runtime_binding_abi.cc)
target_include_directories(ce_ccp1_f02_010 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_f02_010 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_f02_010 PRIVATE -Wall -Wextra -Werror)
target_compile_definitions(ce_ccp1_f02_010 PRIVATE
  CELLERATOR_TEST_INCLUDE_DIR="${CELLERATOR_ROOT}/include"
  CELLERATOR_TEST_BINDING_OBJECT="$<TARGET_OBJECTS:ce_ccp1_f02_010_object>")
add_library(ce_ccp1_f02_010_object OBJECT
  ${CELLERATOR_ROOT}/src/compiler/backend/implement_host_runtime_binding_abi.cc)
target_include_directories(ce_ccp1_f02_010_object PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_f02_010_object PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_f02_010 COMMAND ce_ccp1_f02_010)

add_executable(ce_ccp1_f02_011
  map_backend_diagnostics_to_source_and_ceir_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/map_backend_diagnostics_to_source_and_ceir.cc)
target_include_directories(ce_ccp1_f02_011 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_f02_011 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_f02_011 PRIVATE -Wall -Wextra -Werror)
add_test(NAME ce_ccp1_f02_011 COMMAND ce_ccp1_f02_011)

add_executable(ce_ccp1_f02_012
  benchmark_cpu_backend_complete_cost_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/benchmark_cpu_backend_complete_cost.cc)
target_include_directories(ce_ccp1_f02_012 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_f02_012 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_f02_012 PRIVATE -Wall -Wextra -Werror)
add_test(NAME ce_ccp1_f02_012 COMMAND ce_ccp1_f02_012)

add_executable(ce_ccp1_f02_013
  deliver_the_first_cpu_object_milestone_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/deliver_the_first_cpu_object_milestone.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/compile_generated_c_into_ordinary_objects.cc)
target_include_directories(ce_ccp1_f02_013 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_f02_013 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_f02_013 PRIVATE -Wall -Wextra -Werror)
add_test(NAME ce_ccp1_f02_013 COMMAND ce_ccp1_f02_013)
