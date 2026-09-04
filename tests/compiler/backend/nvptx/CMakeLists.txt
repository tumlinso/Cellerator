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
