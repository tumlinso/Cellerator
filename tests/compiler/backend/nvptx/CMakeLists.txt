cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_nvptx_tests LANGUAGES CXX)

enable_testing()
get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

add_executable(ce_ccp1_f04_001
    freeze_clang_cuda_and_nvptx_backend_contracts_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/freeze_clang_cuda_and_nvptx_backend_contracts.cc")
target_include_directories(ce_ccp1_f04_001 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f04_001 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f04_001 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f04_001 COMMAND ce_ccp1_f04_001)

add_executable(ce_ccp1_f04_002
    implement_clang_cuda_action_mapping_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/implement_clang_cuda_action_mapping.cc")
target_include_directories(ce_ccp1_f04_002 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f04_002 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f04_002 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f04_002 COMMAND ce_ccp1_f04_002)

add_executable(ce_ccp1_f04_003
    implement_llvm_nvptx_module_boundary_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/implement_llvm_nvptx_module_boundary.cc")
target_include_directories(ce_ccp1_f04_003 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f04_003 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f04_003 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f04_003 COMMAND ce_ccp1_f04_003)

add_executable(ce_ccp1_f04_004
    define_direct_ptx_typed_operation_model_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/define_direct_ptx_typed_operation_model.cc")
target_include_directories(ce_ccp1_f04_004 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f04_004 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f04_004 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f04_004 COMMAND ce_ccp1_f04_004)

add_executable(ce_ccp1_f04_005
    implement_inline_ptx_native_block_binding_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/implement_inline_ptx_native_block_binding.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/define_direct_ptx_typed_operation_model.cc")
target_include_directories(ce_ccp1_f04_005 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f04_005 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f04_005 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f04_005 COMMAND ce_ccp1_f04_005)

find_program(PTXAS_EXECUTABLE ptxas REQUIRED)
add_executable(ce_ccp1_f04_006
    implement_ptx_emission_and_ptxas_assembly_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/implement_ptx_emission_and_ptxas_assembly.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/define_direct_ptx_typed_operation_model.cc")
target_include_directories(ce_ccp1_f04_006 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f04_006 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f04_006 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f04_006 COMMAND ce_ccp1_f04_006
    "${PTXAS_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/f04_006")

find_program(OBJCOPY_EXECUTABLE objcopy REQUIRED)
set(F04_007_DIR "${CMAKE_CURRENT_BINARY_DIR}/f04_007")
set(F04_007_CUBIN "${F04_007_DIR}/embedded_minimal_kernel.cubin")
set(F04_007_OBJECT "${F04_007_DIR}/embedded_minimal_kernel.o")
add_custom_command(OUTPUT "${F04_007_CUBIN}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${F04_007_DIR}"
    COMMAND "${PTXAS_EXECUTABLE}" -arch=sm_70
        "${CMAKE_CURRENT_LIST_DIR}/embedded_minimal_kernel.ptx" -o "${F04_007_CUBIN}"
    DEPENDS embedded_minimal_kernel.ptx VERBATIM)
add_custom_command(OUTPUT "${F04_007_OBJECT}"
    COMMAND "${OBJCOPY_EXECUTABLE}" --input-target=binary --output-target=elf64-x86-64
        --binary-architecture=i386:x86-64
        --rename-section .data=.nv_cellerator_cubin,alloc,load,readonly,data,contents
        embedded_minimal_kernel.cubin embedded_minimal_kernel.o
    WORKING_DIRECTORY "${F04_007_DIR}"
    DEPENDS "${F04_007_CUBIN}" VERBATIM)
set_source_files_properties("${F04_007_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
add_executable(ce_ccp1_f04_007
    implement_fatbinary_object_embedding_for_direct_ptx_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/implement_fatbinary_object_embedding_for_direct_ptx.cc"
    "${F04_007_OBJECT}")
target_include_directories(ce_ccp1_f04_007 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f04_007 PRIVATE cxx_std_17)
target_link_libraries(ce_ccp1_f04_007 PRIVATE dl)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f04_007 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f04_007 COMMAND ce_ccp1_f04_007)

add_executable(ce_ccp1_f04_008
    map_target_capabilities_and_instruction_families_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/map_target_capabilities_and_instruction_families.cc")
target_include_directories(ce_ccp1_f04_008 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f04_008 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f04_008 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f04_008 COMMAND ce_ccp1_f04_008)

add_executable(ce_ccp1_f04_009
    implement_source_to_ptx_provenance_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/implement_source_to_ptx_provenance.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/implement_ptx_emission_and_ptxas_assembly.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/define_direct_ptx_typed_operation_model.cc")
target_include_directories(ce_ccp1_f04_009 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f04_009 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f04_009 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f04_009 COMMAND ce_ccp1_f04_009)

find_library(CUDA_DRIVER_LIBRARY cuda REQUIRED)
add_executable(ce_ccp1_f04_010
    deliver_a_direct_ptx_hot_path_demonstration_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/deliver_a_direct_ptx_hot_path_demonstration.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/implement_ptx_emission_and_ptxas_assembly.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/define_direct_ptx_typed_operation_model.cc")
target_include_directories(ce_ccp1_f04_010 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f04_010 PRIVATE cxx_std_17)
target_link_libraries(ce_ccp1_f04_010 PRIVATE "${CUDA_DRIVER_LIBRARY}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f04_010 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f04_010 COMMAND ce_ccp1_f04_010
    "${PTXAS_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/f04_010")

find_program(NVCC_EXECUTABLE nvcc REQUIRED)
find_program(CLANG_CUDA_EXECUTABLE clang++-18 REQUIRED)
find_program(PTXAS_MODERN_EXECUTABLE ptxas
    PATHS "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin"
    NO_DEFAULT_PATH REQUIRED)
add_executable(ce_ccp1_f04_011
    compare_nvcc_clang_cuda_and_direct_ptx_routes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/compare_nvcc_clang_cuda_and_direct_ptx_routes.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/deliver_a_direct_ptx_hot_path_demonstration.cc")
target_include_directories(ce_ccp1_f04_011 PRIVATE "${CELLERATOR_ROOT}/include" "${CELLERATOR_ROOT}")
target_compile_features(ce_ccp1_f04_011 PRIVATE cxx_std_17)
target_link_libraries(ce_ccp1_f04_011 PRIVATE "${CUDA_DRIVER_LIBRARY}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f04_011 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f04_011 COMMAND ce_ccp1_f04_011
    "${NVCC_EXECUTABLE}" "${CLANG_CUDA_EXECUTABLE}" "${PTXAS_MODERN_EXECUTABLE}"
    "${CELLERATOR_ROOT}/bench/compiler/backend/nvptx/matched_unit_degree_relation.cu"
    "${CMAKE_CURRENT_BINARY_DIR}/f04_011" "/usr/lib/gcc/x86_64-linux-gnu/12")

add_executable(ce_ccp1_f04_012
    validate_backend_fallback_and_mixed_routes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/validate_backend_fallback_and_mixed_routes.cc")
target_include_directories(ce_ccp1_f04_012 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f04_012 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f04_012 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f04_012 COMMAND ce_ccp1_f04_012)
