# Optional CUDA realizations for the native data contracts. This may be
# included by the standalone Framework CUDA SDK (which enables CUDA itself) or
# by the root CUDA build. It intentionally does nothing in a host-only build.
include_guard(GLOBAL)
include(GNUInstallDirs)

if(NOT CMAKE_CUDA_COMPILER_LOADED)
    return()
endif()
find_package(CUDAToolkit REQUIRED)

set(_cellerator_native_data_cuda_export CelleratorNativeDataCUDATargets)
if(NOT TARGET cellerator_memory_cuda)
    cellerator_add_native_data_memory_cuda_owner()
endif()

add_library(cellerator_state_codes_cuda STATIC
    "${PROJECT_SOURCE_DIR}/src/tensor/state_codes/operations.cu"
    "${PROJECT_SOURCE_DIR}/src/tensor/state_codes/owner.cu")
add_library(Cellerator::state_codes_cuda ALIAS cellerator_state_codes_cuda)
set_target_properties(cellerator_state_codes_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME state_codes_cuda
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_compile_features(cellerator_state_codes_cuda PUBLIC cxx_std_20)
target_include_directories(cellerator_state_codes_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_state_codes_cuda PUBLIC
    Cellerator::memory_cuda CUDA::cudart)

add_library(cellerator_block_masked_cuda STATIC
    "${PROJECT_SOURCE_DIR}/src/tensor/block_masked/owner.cc"
    "${PROJECT_SOURCE_DIR}/src/ops/mechanism/packed_gated_product.cu")
add_library(Cellerator::block_masked_cuda ALIAS cellerator_block_masked_cuda)
set_target_properties(cellerator_block_masked_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME block_masked_cuda
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_compile_features(cellerator_block_masked_cuda PUBLIC cxx_std_20)
target_include_directories(cellerator_block_masked_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_block_masked_cuda PUBLIC
    Cellerator::block_masked Cellerator::memory_cuda CUDA::cudart)
target_compile_options(cellerator_block_masked_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

add_library(cellerator_dependency_execution_cuda STATIC
    "${PROJECT_SOURCE_DIR}/src/execution/dependency/transfer.cu"
    "${PROJECT_SOURCE_DIR}/src/execution/dependency/distributed.cu"
    "${PROJECT_SOURCE_DIR}/src/execution/dependency/packed_program.cu")
add_library(Cellerator::dependency_execution_cuda
    ALIAS cellerator_dependency_execution_cuda)
set_target_properties(cellerator_dependency_execution_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME dependency_execution_cuda
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_compile_features(cellerator_dependency_execution_cuda PUBLIC cxx_std_20)
target_include_directories(cellerator_dependency_execution_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_dependency_execution_cuda PUBLIC
    Cellerator::dependency_execution
    Cellerator::block_masked_cuda
    Cellerator::memory_cuda CUDA::cudart)
target_compile_options(cellerator_dependency_execution_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

# A separately selectable native relation executor. This target owns the CUDA
# realization while the general structural packer remains in Cellerator::layout.
# It deliberately has no Torch dependency.
add_library(cellerator_factorized_interaction_cuda STATIC
    "${PROJECT_SOURCE_DIR}/src/modeling/cuda/factorized_interaction.cu"
    "${PROJECT_SOURCE_DIR}/src/modeling/cuda/factorized_interaction_direct.cu"
    "${PROJECT_SOURCE_DIR}/src/modeling/cuda/factorized_interaction_stable.cu"
    "${PROJECT_SOURCE_DIR}/src/modeling/cuda/prepared_factorized_interaction.cu")
add_library(Cellerator::factorized_interaction_cuda
    ALIAS cellerator_factorized_interaction_cuda)
set_target_properties(cellerator_factorized_interaction_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME factorized_interaction_cuda
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_compile_features(cellerator_factorized_interaction_cuda PUBLIC cxx_std_20)
target_include_directories(cellerator_factorized_interaction_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_factorized_interaction_cuda PUBLIC
    Cellerator::layout Cellerator::memory_cuda CUDA::cudart)

# The relation executor is independently selectable.  Its own export contains
# only the selected target; its canonical layout and memory dependencies are
# imported from their shared narrow exports.
install(TARGETS cellerator_factorized_interaction_cuda
    EXPORT CelleratorFactorizedInteractionTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT FactorizedInteractionCUDA)
install(EXPORT CelleratorFactorizedInteractionTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator
    COMPONENT FactorizedInteractionCUDA)
install(FILES
        "${PROJECT_SOURCE_DIR}/include/Cellerator/layout/factorized_interaction.hh"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/layout"
        COMPONENT FactorizedInteractionCUDA)
install(FILES
        "${PROJECT_SOURCE_DIR}/include/Cellerator/modeling/cuda/factorized_interaction.hh"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/modeling/cuda"
        COMPONENT FactorizedInteractionCUDA)

add_library(cellerator_tensor_cuda INTERFACE)
add_library(Cellerator::tensor_cuda ALIAS cellerator_tensor_cuda)
set_target_properties(cellerator_tensor_cuda PROPERTIES EXPORT_NAME tensor_cuda)
target_include_directories(cellerator_tensor_cuda INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(cellerator_tensor_cuda INTERFACE cxx_std_20)
target_link_libraries(cellerator_tensor_cuda INTERFACE
    Cellerator::memory_cuda
    Cellerator::state_codes_cuda
    Cellerator::block_masked_cuda)

add_library(cellerator_native_data_cuda_api INTERFACE)
add_library(Cellerator::native_data_cuda ALIAS cellerator_native_data_cuda_api)
set_target_properties(cellerator_native_data_cuda_api PROPERTIES
    EXPORT_NAME native_data_cuda)
target_include_directories(cellerator_native_data_cuda_api INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(cellerator_native_data_cuda_api INTERFACE
    cxx_std_20 cuda_std_20)
target_link_libraries(cellerator_native_data_cuda_api INTERFACE
    Cellerator::native_data
    Cellerator::tensor_cuda
    Cellerator::dependency_execution_cuda)

install(TARGETS
        cellerator_state_codes_cuda
        cellerator_block_masked_cuda
        cellerator_dependency_execution_cuda
    EXPORT ${_cellerator_native_data_cuda_export}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cellerator_tensor_cuda
    EXPORT ${_cellerator_native_data_cuda_export})
install(TARGETS cellerator_native_data_cuda_api
    EXPORT ${_cellerator_native_data_cuda_export})
install(EXPORT ${_cellerator_native_data_cuda_export} NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)

foreach(_header_subtree IN ITEMS
        tensor/state_codes
        tensor/block_masked)
    get_filename_component(_header_parent "${_header_subtree}" DIRECTORY)
    install(DIRECTORY
        "${PROJECT_SOURCE_DIR}/include/Cellerator/${_header_subtree}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/${_header_parent}"
        COMPONENT NativeDataCUDA)
endforeach()
foreach(_header IN ITEMS
        layout/factorized_interaction.hh
        layout/factorized_interaction_pattern.hh
        modeling/cuda/factorized_interaction.hh
        ops/mechanism/packed_gated_product.cuh
        ops/quantized/api.cuh
        ops/quantized/dispatch.cuh
        ops/quantized/extreme_backend.cuh
        ops/quantized/extreme_dequantize_kernel.cuh
        ops/quantized/extreme_metadata_access.cuh
        ops/quantized/extreme_ptx_primitives.cuh
        ops/quantized/extreme_quantize_kernel.cuh
        ops/quantized/kernels.cuh
        ops/quantized/portable_backend.cuh
        ops/quantized/portable_dequantize_kernel.cuh
        ops/quantized/portable_quantize_kernel.cuh
        tensor/quantized/access.cuh
        tensor/quantized/blocked_ell.cuh
        tensor/quantized/format.cuh
        tensor/quantized/layout.cuh
        tensor/quantized/metadata.cuh
        tensor/quantized/numeric.cuh
        tensor/quantized/packing.cuh
        execution/dependency/distributed.cuh
        execution/dependency/packed_program.cuh
        execution/dependency/transfer.cuh
        interop/formats/dense.cuh
        interop/formats/compressed.cuh
        interop/formats/coo.cuh
        interop/formats/blocked_ell.cuh
        interop/formats/sliced_ell.cuh
        interop/formats/device_views.cuh)
    get_filename_component(_header_parent "${_header}" DIRECTORY)
    install(FILES "${PROJECT_SOURCE_DIR}/include/Cellerator/${_header}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/${_header_parent}"
        COMPONENT NativeDataCUDA)
endforeach()

function(cellerator_append_native_data_cuda_package_config)
    if(DEFINED CELLERATOR_HOST_CONFIG AND EXISTS "${CELLERATOR_HOST_CONFIG}")
        file(APPEND "${CELLERATOR_HOST_CONFIG}"
            "include(CMakeFindDependencyMacro)\n"
            "find_dependency(CUDAToolkit)\n"
            "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorFactorizedInteractionTargets.cmake\")\n"
            "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorNativeDataCUDATargets.cmake\")\n"
            "set(Cellerator_native_data_cuda_FOUND TRUE)\n"
            "set(Cellerator_factorized_interaction_cuda_FOUND TRUE)\n")
    endif()
endfunction()

option(CELLERATOR_BUILD_FACTORIZED_INTERACTION_EXAMPLE
    "Build the native C++ factorized interaction training example" OFF)
if(CELLERATOR_BUILD_FACTORIZED_INTERACTION_EXAMPLE)
    add_executable(cellerator_factorized_interaction_training
        "${PROJECT_SOURCE_DIR}/examples/native_cuda/factorized_interaction_training.cu")
    target_link_libraries(cellerator_factorized_interaction_training PRIVATE
        Cellerator::factorized_interaction_cuda)
    set_target_properties(cellerator_factorized_interaction_training PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()

option(CELLERATOR_BUILD_FACTORIZED_INTERACTION_VALIDATOR
    "Build the independent native CUDA protocol runner" OFF)
if(CELLERATOR_BUILD_FACTORIZED_INTERACTION_VALIDATOR)
    add_subdirectory(
        "${PROJECT_SOURCE_DIR}/bench/factorized_interactions"
        "${CMAKE_CURRENT_BINARY_DIR}/factorized_interactions_validator")
endif()

if(CELLERATOR_BUILD_FACTORIZED_INTERACTION_TESTS)
    add_executable(cellerator_factorized_interaction_cuda_contract_test
        "${PROJECT_SOURCE_DIR}/tests/factorized_interactions/host_contract_test.cc")
    target_link_libraries(cellerator_factorized_interaction_cuda_contract_test PRIVATE
        Cellerator::factorized_interaction_cuda)
    target_compile_features(cellerator_factorized_interaction_cuda_contract_test PRIVATE cxx_std_20)
    target_link_options(cellerator_factorized_interaction_cuda_contract_test PRIVATE
        "LINKER:--wrap=cudaGetDevice"
        "LINKER:--wrap=cudaMemsetAsync")
    add_test(NAME factorized_interaction_cuda_host_contract
        COMMAND cellerator_factorized_interaction_cuda_contract_test)
    set_tests_properties(factorized_interaction_cuda_host_contract PROPERTIES
        LABELS "factorized_interaction;host_only;cuda_compile" TIMEOUT 60)
endif()

# These executables compile device contracts for validation but are not CTest
# cases: running them requires a Scheduler-admitted GPU lease.
if(CELLERATOR_BUILD_NATIVE_DATA_TESTS)
    set(_cellerator_native_data_cuda_test_targets)
    add_executable(cellerator_native_data_transfer_compile_test
        "${PROJECT_SOURCE_DIR}/tests/execution_dependency/transfer_contract_test.cu")
    target_link_libraries(cellerator_native_data_transfer_compile_test PRIVATE
        Cellerator::dependency_execution_cuda)
    set_target_properties(cellerator_native_data_transfer_compile_test PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    list(APPEND _cellerator_native_data_cuda_test_targets
        cellerator_native_data_transfer_compile_test)

    add_executable(cellerator_native_data_block_masked_compile_test
        "${PROJECT_SOURCE_DIR}/tests/tensor/block_masked_cuda_test.cu")
    target_link_libraries(cellerator_native_data_block_masked_compile_test PRIVATE
        Cellerator::block_masked_cuda)
    set_target_properties(cellerator_native_data_block_masked_compile_test PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    list(APPEND _cellerator_native_data_cuda_test_targets
        cellerator_native_data_block_masked_compile_test)

    # The default executable path needs a Scheduler-admitted GPU. This target
    # is compiled and linked only; its separate host-only branch is useful to
    # the owner but is not registered with CTest here.
    add_executable(cellerator_native_owner_cuda_test
        "${PROJECT_SOURCE_DIR}/tests/tensor/native_owner_cuda_test.cu")
    target_link_libraries(cellerator_native_owner_cuda_test PRIVATE
        Cellerator::memory_cuda Cellerator::state_codes_cuda)
    set_target_properties(cellerator_native_owner_cuda_test PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    list(APPEND _cellerator_native_data_cuda_test_targets
        cellerator_native_owner_cuda_test)

    add_executable(tensor_quantized_format_test
        "${PROJECT_SOURCE_DIR}/tests/tensor/quantized_format_test.cu")
    target_include_directories(tensor_quantized_format_test PRIVATE
        "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(tensor_quantized_format_test PRIVATE CUDA::cudart)
    set_target_properties(tensor_quantized_format_test PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    add_test(NAME native_data_quantized_format
        COMMAND tensor_quantized_format_test)
    set_tests_properties(native_data_quantized_format PROPERTIES
        LABELS "native_data;host_only" TIMEOUT 60)
    list(APPEND _cellerator_native_data_cuda_test_targets
        tensor_quantized_format_test)

    # This payload contract test uses NVCC so the public CUDA-aware view
    # headers are parsed in their intended compilation mode, but the test
    # itself exercises host-side descriptors only and does not require a GPU.
    add_executable(tensor_format_payload_contract_test
        "${PROJECT_SOURCE_DIR}/tests/native_data/format_payload_contract_test.cu")
    target_include_directories(tensor_format_payload_contract_test PRIVATE
        "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(tensor_format_payload_contract_test PRIVATE CUDA::cudart)
    set_target_properties(tensor_format_payload_contract_test PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    add_test(NAME native_data_format_payload_contract
        COMMAND tensor_format_payload_contract_test)
    set_tests_properties(native_data_format_payload_contract PROPERTIES
        LABELS "native_data;host_only" TIMEOUT 60)
    list(APPEND _cellerator_native_data_cuda_test_targets
        tensor_format_payload_contract_test)

    add_executable(cellerator_native_data_distributed_cuda_test
        "${PROJECT_SOURCE_DIR}/tests/execution_dependency/distributed_cuda_test.cu")
    target_link_libraries(cellerator_native_data_distributed_cuda_test PRIVATE
        Cellerator::native_data_cuda)
    set_target_properties(cellerator_native_data_distributed_cuda_test PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    list(APPEND _cellerator_native_data_cuda_test_targets
        cellerator_native_data_distributed_cuda_test)

    # This two-device contract executable checks 32-way distributed binding
    # and VJPs. It is intentionally build-only: execution requires an
    # explicitly admitted two-device Scheduler lease.
    add_executable(cellerator_native_data_distributed_32way_test
        "${PROJECT_SOURCE_DIR}/tests/execution_dependency/distributed_32way_test.cu")
    target_link_libraries(cellerator_native_data_distributed_32way_test PRIVATE
        Cellerator::native_data_cuda)
    set_target_properties(cellerator_native_data_distributed_32way_test PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    list(APPEND _cellerator_native_data_cuda_test_targets
        cellerator_native_data_distributed_32way_test)

    # The PackedProgram contract test exercises real prepared-program
    # admission on two devices. Build and link it here, but leave execution to
    # an explicitly admitted Scheduler job rather than registering it with
    # CTest.
    add_executable(cellerator_native_data_packed_program_contract_test
        "${PROJECT_SOURCE_DIR}/tests/execution_dependency/packed_program_contract_test.cu")
    target_link_libraries(cellerator_native_data_packed_program_contract_test PRIVATE
        Cellerator::native_data_cuda)
    set_target_properties(cellerator_native_data_packed_program_contract_test PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    list(APPEND _cellerator_native_data_cuda_test_targets
        cellerator_native_data_packed_program_contract_test)

    add_custom_target(cellerator_native_data_cuda_test
        DEPENDS ${_cellerator_native_data_cuda_test_targets})
endif()

add_custom_target(cellerator_native_data_cuda
    DEPENDS
        cellerator_memory_cuda
        cellerator_state_codes_cuda
        cellerator_block_masked_cuda
        cellerator_dependency_execution_cuda)
if(TARGET cellerator_native_data_cuda_test)
    add_dependencies(cellerator_native_data_cuda
        cellerator_native_data_cuda_test)
endif()
