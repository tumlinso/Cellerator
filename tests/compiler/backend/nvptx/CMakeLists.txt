cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_nvptx_tests LANGUAGES CXX)

enable_testing()
get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

add_executable(cellerator_clang_cuda_and_nvptx_backend_contracts_test
    clang_cuda_and_nvptx_backend_contracts_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/clang_cuda_and_nvptx_backend_contracts.cc")
target_include_directories(cellerator_clang_cuda_and_nvptx_backend_contracts_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_clang_cuda_and_nvptx_backend_contracts_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_clang_cuda_and_nvptx_backend_contracts_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_clang_cuda_and_nvptx_backend_contracts_test COMMAND cellerator_clang_cuda_and_nvptx_backend_contracts_test)

add_executable(cellerator_clang_cuda_action_mapping_test
    clang_cuda_action_mapping_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/clang_cuda_action_mapping.cc")
target_include_directories(cellerator_clang_cuda_action_mapping_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_clang_cuda_action_mapping_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_clang_cuda_action_mapping_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_clang_cuda_action_mapping_test COMMAND cellerator_clang_cuda_action_mapping_test)

add_executable(cellerator_llvm_nvptx_module_boundary_test
    llvm_nvptx_module_boundary_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/llvm_nvptx_module_boundary.cc")
target_include_directories(cellerator_llvm_nvptx_module_boundary_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_llvm_nvptx_module_boundary_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_llvm_nvptx_module_boundary_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_llvm_nvptx_module_boundary_test COMMAND cellerator_llvm_nvptx_module_boundary_test)

add_executable(cellerator_direct_ptx_typed_operation_model_test
    direct_ptx_typed_operation_model_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/direct_ptx_typed_operation_model.cc")
target_include_directories(cellerator_direct_ptx_typed_operation_model_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_direct_ptx_typed_operation_model_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_direct_ptx_typed_operation_model_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_direct_ptx_typed_operation_model_test COMMAND cellerator_direct_ptx_typed_operation_model_test)

add_executable(cellerator_inline_ptx_native_block_binding_test
    inline_ptx_native_block_binding_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/inline_ptx_native_block_binding.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/direct_ptx_typed_operation_model.cc")
target_include_directories(cellerator_inline_ptx_native_block_binding_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_inline_ptx_native_block_binding_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_inline_ptx_native_block_binding_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_inline_ptx_native_block_binding_test COMMAND cellerator_inline_ptx_native_block_binding_test)

find_program(PTXAS_EXECUTABLE ptxas REQUIRED)
add_executable(cellerator_ptx_emission_and_ptxas_assembly_test
    ptx_emission_and_ptxas_assembly_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/ptx_emission_and_ptxas_assembly.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/direct_ptx_typed_operation_model.cc")
target_include_directories(cellerator_ptx_emission_and_ptxas_assembly_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_ptx_emission_and_ptxas_assembly_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_ptx_emission_and_ptxas_assembly_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_ptx_emission_and_ptxas_assembly_test COMMAND cellerator_ptx_emission_and_ptxas_assembly_test
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
add_executable(cellerator_fatbinary_object_embedding_for_direct_ptx_test
    fatbinary_object_embedding_for_direct_ptx_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/fatbinary_object_embedding_for_direct_ptx.cc"
    "${F04_007_OBJECT}")
target_include_directories(cellerator_fatbinary_object_embedding_for_direct_ptx_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_fatbinary_object_embedding_for_direct_ptx_test PRIVATE cxx_std_17)
target_link_libraries(cellerator_fatbinary_object_embedding_for_direct_ptx_test PRIVATE dl)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_fatbinary_object_embedding_for_direct_ptx_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_fatbinary_object_embedding_for_direct_ptx_test COMMAND cellerator_fatbinary_object_embedding_for_direct_ptx_test)

add_executable(cellerator_map_target_capabilities_and_instruction_families_test
    map_target_capabilities_and_instruction_families_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/map_target_capabilities_and_instruction_families.cc")
target_include_directories(cellerator_map_target_capabilities_and_instruction_families_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_map_target_capabilities_and_instruction_families_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_map_target_capabilities_and_instruction_families_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_map_target_capabilities_and_instruction_families_test COMMAND cellerator_map_target_capabilities_and_instruction_families_test)

add_executable(cellerator_source_to_ptx_provenance_test
    source_to_ptx_provenance_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/source_to_ptx_provenance.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/ptx_emission_and_ptxas_assembly.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/direct_ptx_typed_operation_model.cc")
target_include_directories(cellerator_source_to_ptx_provenance_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_source_to_ptx_provenance_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_source_to_ptx_provenance_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_source_to_ptx_provenance_test COMMAND cellerator_source_to_ptx_provenance_test)

find_library(CUDA_DRIVER_LIBRARY cuda REQUIRED)
add_executable(cellerator_direct_ptx_hot_path_demonstration_test
    direct_ptx_hot_path_demonstration_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/direct_ptx_hot_path_demonstration.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/ptx_emission_and_ptxas_assembly.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/direct_ptx_typed_operation_model.cc")
target_include_directories(cellerator_direct_ptx_hot_path_demonstration_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_direct_ptx_hot_path_demonstration_test PRIVATE cxx_std_17)
target_link_libraries(cellerator_direct_ptx_hot_path_demonstration_test PRIVATE "${CUDA_DRIVER_LIBRARY}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_direct_ptx_hot_path_demonstration_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_direct_ptx_hot_path_demonstration_test COMMAND cellerator_direct_ptx_hot_path_demonstration_test
    "${PTXAS_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/f04_010")

find_program(NVCC_EXECUTABLE nvcc REQUIRED)
find_program(CLANG_CUDA_EXECUTABLE clang++-18 REQUIRED)
find_program(PTXAS_MODERN_EXECUTABLE ptxas
    PATHS "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin"
    NO_DEFAULT_PATH REQUIRED)
add_executable(cellerator_compare_nvcc_clang_cuda_and_direct_ptx_routes_test
    compare_nvcc_clang_cuda_and_direct_ptx_routes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/compare_nvcc_clang_cuda_and_direct_ptx_routes.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/direct_ptx_hot_path_demonstration.cc")
target_include_directories(cellerator_compare_nvcc_clang_cuda_and_direct_ptx_routes_test PRIVATE "${CELLERATOR_ROOT}/include" "${CELLERATOR_ROOT}")
target_compile_features(cellerator_compare_nvcc_clang_cuda_and_direct_ptx_routes_test PRIVATE cxx_std_17)
target_link_libraries(cellerator_compare_nvcc_clang_cuda_and_direct_ptx_routes_test PRIVATE "${CUDA_DRIVER_LIBRARY}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_compare_nvcc_clang_cuda_and_direct_ptx_routes_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_compare_nvcc_clang_cuda_and_direct_ptx_routes_test COMMAND cellerator_compare_nvcc_clang_cuda_and_direct_ptx_routes_test
    "${NVCC_EXECUTABLE}" "${CLANG_CUDA_EXECUTABLE}" "${PTXAS_MODERN_EXECUTABLE}"
    "${CELLERATOR_ROOT}/bench/compiler/backend/nvptx/matched_unit_degree_relation.cu"
    "${CMAKE_CURRENT_BINARY_DIR}/f04_011" "/usr/lib/gcc/x86_64-linux-gnu/12")

add_executable(cellerator_validate_backend_fallback_and_mixed_routes_test
    validate_backend_fallback_and_mixed_routes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/validate_backend_fallback_and_mixed_routes.cc")
target_include_directories(cellerator_validate_backend_fallback_and_mixed_routes_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_validate_backend_fallback_and_mixed_routes_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_validate_backend_fallback_and_mixed_routes_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_validate_backend_fallback_and_mixed_routes_test COMMAND cellerator_validate_backend_fallback_and_mixed_routes_test)

add_executable(cellerator_optional_nvidia_backend_routes_test
    optional_nvidia_backend_routes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/optional_nvidia_backend_routes.cc"
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/clang_cuda_and_nvptx_backend_contracts.cc")
target_include_directories(cellerator_optional_nvidia_backend_routes_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_optional_nvidia_backend_routes_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_optional_nvidia_backend_routes_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_optional_nvidia_backend_routes_test COMMAND cellerator_optional_nvidia_backend_routes_test)
