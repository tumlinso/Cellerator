cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_cpu_backend_tests LANGUAGES CXX)

enable_testing()
get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

add_executable(cellerator_backend_provider_abi_test
    backend_provider_abi_test.cc
    mock_backend_provider_v1.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/backend_provider_abi.cc")
target_include_directories(cellerator_backend_provider_abi_test PRIVATE
    "${CELLERATOR_ROOT}/include"
    "${CMAKE_CURRENT_LIST_DIR}")
target_compile_features(cellerator_backend_provider_abi_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_backend_provider_abi_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_f02_001 ALIAS cellerator_backend_provider_abi_test)
add_test(NAME ce_ccp1_f02_001 COMMAND cellerator_backend_provider_abi_test)

add_executable(cellerator_backend_registry_and_selection_test
    backend_registry_and_selection_test.cc
    mock_backend_provider_v1.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/backend_provider_abi.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/backend_registry_and_selection.cc")
target_include_directories(cellerator_backend_registry_and_selection_test PRIVATE
    "${CELLERATOR_ROOT}/include"
    "${CMAKE_CURRENT_LIST_DIR}")
target_compile_features(cellerator_backend_registry_and_selection_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_backend_registry_and_selection_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_f02_002 ALIAS cellerator_backend_registry_and_selection_test)
add_test(NAME ce_ccp1_f02_002 COMMAND cellerator_backend_registry_and_selection_test)

add_executable(cellerator_backend_code_generation_plans_test
    backend_code_generation_plans_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/backend_code_generation_plans.cc")
target_include_directories(cellerator_backend_code_generation_plans_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_backend_code_generation_plans_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_backend_code_generation_plans_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_f02_003 ALIAS cellerator_backend_code_generation_plans_test)
add_test(NAME ce_ccp1_f02_003 COMMAND cellerator_backend_code_generation_plans_test)

add_executable(cellerator_generated_c_representation_test
    generated_c_representation_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/generated_c_representation.cc")
target_include_directories(cellerator_generated_c_representation_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_generated_c_representation_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_generated_c_representation_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_f02_004 ALIAS cellerator_generated_c_representation_test)
add_test(NAME ce_ccp1_f02_004 COMMAND cellerator_generated_c_representation_test)

add_executable(cellerator_generic_cpu_relation_apply_test
  generic_cpu_relation_apply_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/generic_cpu_relation_apply.cc)
target_include_directories(cellerator_generic_cpu_relation_apply_test PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(cellerator_generic_cpu_relation_apply_test PRIVATE cxx_std_17)
target_compile_options(cellerator_generic_cpu_relation_apply_test PRIVATE -Wall -Wextra -Werror)
add_executable(ce_ccp1_f02_005 ALIAS cellerator_generic_cpu_relation_apply_test)
add_test(NAME ce_ccp1_f02_005 COMMAND cellerator_generic_cpu_relation_apply_test)

add_executable(cellerator_cpu_transpose_and_contraction_test
  cpu_transpose_and_contraction_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/cpu_transpose_and_contraction.cc)
target_include_directories(cellerator_cpu_transpose_and_contraction_test PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(cellerator_cpu_transpose_and_contraction_test PRIVATE cxx_std_17)
target_compile_options(cellerator_cpu_transpose_and_contraction_test PRIVATE -Wall -Wextra -Werror)
add_executable(ce_ccp1_f02_006 ALIAS cellerator_cpu_transpose_and_contraction_test)
add_test(NAME ce_ccp1_f02_006 COMMAND cellerator_cpu_transpose_and_contraction_test)

add_executable(cellerator_cpu_segment_gate_update_bundle_and_chain_paths_test
  cpu_segment_gate_update_bundle_and_chain_paths_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/cpu_segment_gate_update_bundle_and_chain_paths.cc)
target_include_directories(cellerator_cpu_segment_gate_update_bundle_and_chain_paths_test PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(cellerator_cpu_segment_gate_update_bundle_and_chain_paths_test PRIVATE cxx_std_17)
target_compile_options(cellerator_cpu_segment_gate_update_bundle_and_chain_paths_test PRIVATE -Wall -Wextra -Werror)
add_executable(ce_ccp1_f02_007 ALIAS cellerator_cpu_segment_gate_update_bundle_and_chain_paths_test)
add_test(NAME ce_ccp1_f02_007 COMMAND cellerator_cpu_segment_gate_update_bundle_and_chain_paths_test)

add_executable(cellerator_cpu_projection_packing_and_order_transforms_test
  cpu_projection_packing_and_order_transforms_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/cpu_projection_packing_and_order_transforms.cc)
target_include_directories(cellerator_cpu_projection_packing_and_order_transforms_test PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(cellerator_cpu_projection_packing_and_order_transforms_test PRIVATE cxx_std_17)
target_compile_options(cellerator_cpu_projection_packing_and_order_transforms_test PRIVATE -Wall -Wextra -Werror)
add_executable(ce_ccp1_f02_008 ALIAS cellerator_cpu_projection_packing_and_order_transforms_test)
add_test(NAME ce_ccp1_f02_008 COMMAND cellerator_cpu_projection_packing_and_order_transforms_test)

add_executable(ce_ccp1_f02_009
  compile_generated_c_into_ordinary_objects_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/compile_generated_c_into_ordinary_objects.cc)
target_include_directories(ce_ccp1_f02_009 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_f02_009 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_f02_009 PRIVATE -Wall -Wextra -Werror)
add_test(NAME ce_ccp1_f02_009 COMMAND ce_ccp1_f02_009)

add_executable(cellerator_host_runtime_binding_abi_test
  host_runtime_binding_abi_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/host_runtime_binding_abi.cc)
target_include_directories(cellerator_host_runtime_binding_abi_test PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(cellerator_host_runtime_binding_abi_test PRIVATE cxx_std_17)
target_compile_options(cellerator_host_runtime_binding_abi_test PRIVATE -Wall -Wextra -Werror)
target_compile_definitions(cellerator_host_runtime_binding_abi_test PRIVATE
  CELLERATOR_TEST_INCLUDE_DIR="${CELLERATOR_ROOT}/include"
  CELLERATOR_TEST_BINDING_OBJECT="$<TARGET_OBJECTS:cellerator_host_runtime_binding_abi_object>")
add_library(cellerator_host_runtime_binding_abi_object OBJECT
  ${CELLERATOR_ROOT}/src/compiler/backend/host_runtime_binding_abi.cc)
target_include_directories(cellerator_host_runtime_binding_abi_object PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(cellerator_host_runtime_binding_abi_object PRIVATE cxx_std_17)
add_executable(ce_ccp1_f02_010 ALIAS cellerator_host_runtime_binding_abi_test)
add_library(ce_ccp1_f02_010_object ALIAS cellerator_host_runtime_binding_abi_object)
add_test(NAME ce_ccp1_f02_010 COMMAND cellerator_host_runtime_binding_abi_test)

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

add_executable(cellerator_first_cpu_object_milestone_test
  first_cpu_object_milestone_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/first_cpu_object_milestone.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/compile_generated_c_into_ordinary_objects.cc)
target_include_directories(cellerator_first_cpu_object_milestone_test PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(cellerator_first_cpu_object_milestone_test PRIVATE cxx_std_17)
target_compile_options(cellerator_first_cpu_object_milestone_test PRIVATE -Wall -Wextra -Werror)
add_executable(ce_ccp1_f02_013 ALIAS cellerator_first_cpu_object_milestone_test)
add_test(NAME ce_ccp1_f02_013 COMMAND cellerator_first_cpu_object_milestone_test)

add_executable(cellerator_cpu_backend_thin_waist_test
  cpu_backend_thin_waist_test.cc
  ${CELLERATOR_ROOT}/src/compiler/backend/cpu_backend_thin_waist.cc)
target_include_directories(cellerator_cpu_backend_thin_waist_test PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(cellerator_cpu_backend_thin_waist_test PRIVATE cxx_std_17)
target_compile_options(cellerator_cpu_backend_thin_waist_test PRIVATE -Wall -Wextra -Werror)
target_compile_definitions(cellerator_cpu_backend_thin_waist_test PRIVATE
  CELLERATOR_TEST_INCLUDE_ROOT="${CELLERATOR_ROOT}/include")
add_executable(ce_ccp1_f02_014 ALIAS cellerator_cpu_backend_thin_waist_test)
add_test(NAME ce_ccp1_f02_014 COMMAND cellerator_cpu_backend_thin_waist_test)
