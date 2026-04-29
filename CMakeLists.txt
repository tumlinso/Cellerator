cmake_minimum_required(VERSION 3.20)

# Choose the CUDA toolchain before enabling the CUDA language.
#
# On this host CMake can resolve `/usr/bin/nvcc` from the distro CUDA 12.0
# package even when a newer HPC SDK nvcc is earlier on PATH. That older nvcc
# also trips over the default GCC 13 host compiler on Ubuntu 24.04, producing
# the recurring `_Float32` / `_Float64` failures during compiler detection.
#
# Respect explicit user/tooling choices first:
# - `-DCMAKE_CUDA_COMPILER=...`
# - `-DCMAKE_CUDA_HOST_COMPILER=...`
# - `CUDACXX`
# - `CUDAHOSTCXX`
#
# Otherwise prefer the known-good local toolchain pair.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(DEFINED ENV{CUDACXX} AND NOT "$ENV{CUDACXX}" STREQUAL "")
        set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "CUDA compiler" FORCE)
    elseif(EXISTS "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc")
        set(CMAKE_CUDA_COMPILER
            "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc"
            CACHE FILEPATH "CUDA compiler" FORCE)
        message(STATUS "Defaulting CUDA compiler to the local HPC SDK nvcc")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    if(DEFINED ENV{CUDAHOSTCXX} AND NOT "$ENV{CUDAHOSTCXX}" STREQUAL "")
        set(CMAKE_CUDA_HOST_COMPILER "$ENV{CUDAHOSTCXX}" CACHE FILEPATH "CUDA host compiler" FORCE)
    elseif(EXISTS "/usr/bin/g++-12")
        set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12" CACHE FILEPATH "CUDA host compiler" FORCE)
        message(STATUS "Defaulting CUDA host compiler to /usr/bin/g++-12 for nvcc compatibility")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

project(Cellerator LANGUAGES CXX CUDA)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

option(CELLERATOR_ENABLE_TORCH_MODELS "Enable libtorch-backed model targets" ON)
option(CELLERATOR_ENABLE_CUDA_LINEINFO "Emit CUDA line mapping for Nsight profiling" OFF)
set(CELLERATOR_CUDA_MODE "generic" CACHE STRING
    "CUDA optimization mode: generic, native, or native-extreme")
set_property(CACHE CELLERATOR_CUDA_MODE PROPERTY STRINGS generic native native-extreme)
set(CELLERATOR_PREFERRED_LIBTORCH_ROOT "/usr/local" CACHE PATH
    "Preferred root for the source-built libtorch installation")
set(CELLERATOR_CELLSHARD_SOURCE_DIR "" CACHE PATH
    "Optional local CellShard source checkout. Defaults to ../CellShard before find_package(CellShard).")

add_library(cellerator_quantized_formats INTERFACE)
add_library(Cellerator::quantized_formats ALIAS cellerator_quantized_formats)
target_include_directories(cellerator_quantized_formats INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_quantized_formats INTERFACE cxx_std_17)
target_link_libraries(cellerator_quantized_formats INTERFACE CUDA::cudart)

function(cellerator_add_cellshard_source cellshard_source_dir)
    if(NOT EXISTS "${cellshard_source_dir}/CMakeLists.txt")
        message(FATAL_ERROR "CELLERATOR_CELLSHARD_SOURCE_DIR does not point to a CellShard source tree: ${cellshard_source_dir}")
    endif()
    if(NOT EXISTS "${cellshard_source_dir}/include/CellShard/CellShard.hh")
        message(FATAL_ERROR
            "CellShard source tree must expose the package include layout at include/CellShard. "
            "Update the CellShard checkout or set CELLERATOR_CELLSHARD_SOURCE_DIR to a compatible source tree: "
            "${cellshard_source_dir}")
    endif()
    if(DEFINED CELLSHARD_ENABLE_CUDA AND NOT CELLSHARD_ENABLE_CUDA)
        message(FATAL_ERROR
            "Cellerator requires the CUDA CellShard runtime. "
            "Do not configure with CELLSHARD_ENABLE_CUDA=OFF.")
    endif()
    if(NOT DEFINED CELLSHARD_ENABLE_CUDA)
        set(CELLSHARD_ENABLE_CUDA ON CACHE BOOL
            "Enable the CUDA CellShard runtime required by Cellerator.")
    endif()
    if(NOT DEFINED CELLSHARD_ENABLE_CELLERATOR_QUANTIZED)
        set(CELLSHARD_ENABLE_CELLERATOR_QUANTIZED ON CACHE BOOL
            "Use the Cellerator-owned quantized format ABI inside CellShard.")
    endif()
    if(NOT DEFINED CELLSHARD_ENABLE_CUDA_LINEINFO)
        set(CELLSHARD_ENABLE_CUDA_LINEINFO ${CELLERATOR_ENABLE_CUDA_LINEINFO} CACHE BOOL
            "Emit CUDA line mapping for Nsight profiling inside CellShard.")
    endif()
    if(NOT DEFINED CELLSHARD_CUDA_MODE)
        set(CELLSHARD_CUDA_MODE ${CELLERATOR_CUDA_MODE} CACHE STRING
            "CellShard CUDA optimization mode: generic, native, or native-extreme.")
    endif()
    add_subdirectory("${cellshard_source_dir}" "${CMAKE_CURRENT_BINARY_DIR}/CellShard" EXCLUDE_FROM_ALL)
endfunction()

if(CELLERATOR_CELLSHARD_SOURCE_DIR)
    cellerator_add_cellshard_source("${CELLERATOR_CELLSHARD_SOURCE_DIR}")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../CellShard/CMakeLists.txt")
    cellerator_add_cellshard_source("${CMAKE_CURRENT_SOURCE_DIR}/../CellShard")
else()
    find_package(CellShard CONFIG REQUIRED)
endif()

if(NOT TARGET CellShard::runtime)
    message(FATAL_ERROR "Cellerator requires CellShard::runtime")
endif()

if(CELLERATOR_ENABLE_TORCH_MODELS)
    if(DEFINED ENV{LIBTORCH_PATH} AND NOT "$ENV{LIBTORCH_PATH}" STREQUAL "")
        list(PREPEND CMAKE_PREFIX_PATH "$ENV{LIBTORCH_PATH}")
    elseif(EXISTS "${CELLERATOR_PREFERRED_LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
        if(NOT Torch_DIR OR Torch_DIR MATCHES ".*/site-packages/torch/share/cmake/Torch/?$")
            set(Torch_DIR
                "${CELLERATOR_PREFERRED_LIBTORCH_ROOT}/share/cmake/Torch"
                CACHE PATH "Torch CMake package directory" FORCE)
            message(STATUS "Defaulting Torch_DIR to source-built libtorch at ${Torch_DIR}")
        endif()
        if(EXISTS "${CELLERATOR_PREFERRED_LIBTORCH_ROOT}/share/cmake/Caffe2/Caffe2Config.cmake")
            if(NOT Caffe2_DIR OR Caffe2_DIR MATCHES ".*/site-packages/torch/share/cmake/Caffe2/?$")
                set(Caffe2_DIR
                    "${CELLERATOR_PREFERRED_LIBTORCH_ROOT}/share/cmake/Caffe2"
                    CACHE PATH "Caffe2 CMake package directory" FORCE)
                message(STATUS "Defaulting Caffe2_DIR to source-built libtorch at ${Caffe2_DIR}")
            endif()
        endif()
        foreach(cellerator_torch_cache_var
                IN ITEMS
                    TORCH_LIBRARY
                    C10_CUDA_LIBRARY
                    c10_LIBRARY
                    kineto_LIBRARY)
            if(DEFINED ${cellerator_torch_cache_var}
               AND "${${cellerator_torch_cache_var}}" MATCHES ".*/site-packages/torch/.*")
                unset(${cellerator_torch_cache_var} CACHE)
            endif()
        endforeach()
    endif()
endif()

if(CELLERATOR_ENABLE_TORCH_MODELS)
    find_package(Torch REQUIRED)
endif()

find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS
        /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/lib
)

function(cellerator_link_nccl_if_available target)
    if(NCCL_LIBRARY)
        target_link_libraries(${target} PRIVATE ${NCCL_LIBRARY})
    endif()
endfunction()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT CELLERATOR_CUDA_MODE STREQUAL "generic"
   AND NOT CELLERATOR_CUDA_MODE STREQUAL "native"
   AND NOT CELLERATOR_CUDA_MODE STREQUAL "native-extreme")
    message(FATAL_ERROR "CELLERATOR_CUDA_MODE must be one of: generic, native, native-extreme")
endif()

if(CELLERATOR_CUDA_MODE STREQUAL "native-extreme")
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING
        "Native-extreme mode hard-locks the build to Volta sm_70." FORCE)
    set(cellerator_extreme_arch_count 0)
    foreach(cellerator_cuda_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        math(EXPR cellerator_extreme_arch_count "${cellerator_extreme_arch_count} + 1")
        string(STRIP "${cellerator_cuda_arch}" cellerator_cuda_arch_clean)
        if(NOT cellerator_cuda_arch_clean MATCHES "^(70|7\\.0)(-real|-virtual)?$")
            message(FATAL_ERROR
                "CELLERATOR_CUDA_MODE=native-extreme is a Volta-only path and currently requires "
                "CMAKE_CUDA_ARCHITECTURES to contain only sm_70 entries.")
        endif()
    endforeach()
    if(cellerator_extreme_arch_count EQUAL 0)
        message(FATAL_ERROR
            "CELLERATOR_CUDA_MODE=native-extreme requires at least one explicit sm_70 architecture entry.")
    endif()
endif()

set(CELLERATOR_CUDA_MODE_GENERIC 0)
set(CELLERATOR_CUDA_MODE_NATIVE 0)
set(CELLERATOR_CUDA_MODE_NATIVE_EXTREME 0)
set(CELLERATOR_CUDA_MODE_ENUM generic)
if(CELLERATOR_CUDA_MODE STREQUAL "generic")
    set(CELLERATOR_CUDA_MODE_GENERIC 1)
elseif(CELLERATOR_CUDA_MODE STREQUAL "native")
    set(CELLERATOR_CUDA_MODE_NATIVE 1)
    set(CELLERATOR_CUDA_MODE_ENUM native)
else()
    set(CELLERATOR_CUDA_MODE_NATIVE_EXTREME 1)
    set(CELLERATOR_CUDA_MODE_ENUM native_extreme)
endif()

message(STATUS "Cellerator CUDA mode: ${CELLERATOR_CUDA_MODE}")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/support/cuda_mode.hh.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/cellerator_cuda_mode.hh
    @ONLY
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(cellerator_cuda_mode INTERFACE)
target_include_directories(cellerator_cuda_mode INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_compile_definitions(cellerator_cuda_mode INTERFACE
    CELLERATOR_CUDA_MODE_GENERIC=${CELLERATOR_CUDA_MODE_GENERIC}
    CELLERATOR_CUDA_MODE_NATIVE=${CELLERATOR_CUDA_MODE_NATIVE}
    CELLERATOR_CUDA_MODE_NATIVE_EXTREME=${CELLERATOR_CUDA_MODE_NATIVE_EXTREME}
)
target_compile_features(cellerator_cuda_mode INTERFACE cxx_std_17)

function(cellerator_enable_perf target)
    set(cellerator_cuda_perf_flags
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xptxas=-O3,-dlcm=ca
    )
    if(CELLERATOR_ENABLE_CUDA_LINEINFO)
        list(APPEND cellerator_cuda_perf_flags -lineinfo)
    endif()

    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG -march=native -mtune=native>
        $<$<COMPILE_LANGUAGE:CUDA>:${cellerator_cuda_perf_flags}>
    )
    target_link_libraries(${target} PRIVATE cellerator_cuda_mode)
    set_target_properties(${target} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE YES
    )
endfunction()

if(CELLERATOR_ENABLE_TORCH_MODELS)
    add_library(cellerator_torch INTERFACE)
    target_link_libraries(cellerator_torch INTERFACE CUDA::cudart ${TORCH_LIBRARIES})
    target_compile_features(cellerator_torch INTERFACE cxx_std_17)
    if(DEFINED TORCH_INCLUDE_DIRS)
        target_include_directories(cellerator_torch SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})
    endif()
    if(TORCH_CXX_FLAGS)
        separate_arguments(CELLERATOR_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(cellerator_torch INTERFACE ${CELLERATOR_TORCH_CXX_FLAGS})
    endif()

    add_library(cellerator_torch_bindings INTERFACE)
    target_include_directories(cellerator_torch_bindings INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(cellerator_torch_bindings INTERFACE cellerator_torch CellShard::runtime)
    target_compile_features(cellerator_torch_bindings INTERFACE cxx_std_17)

    add_library(cellerator_compute_model_ops STATIC
        src/compute/ml/model_ops/model_ops.cu
    )
    target_include_directories(cellerator_compute_model_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(cellerator_compute_model_ops PRIVATE cellerator_torch CUDA::cudart)
    set_target_properties(cellerator_compute_model_ops PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )
    cellerator_enable_perf(cellerator_compute_model_ops)
endif()

add_executable(quantizedMatrixTest
    tests/quantized_matrix_test.cu
)
target_include_directories(quantizedMatrixTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedMatrixTest PRIVATE CUDA::cudart)
set_target_properties(quantizedMatrixTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedMatrixTest)

add_executable(quantizedMatrixBench
    bench/quantized_matrix_bench.cu
)
target_include_directories(quantizedMatrixBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedMatrixBench PRIVATE CUDA::cudart)
set_target_properties(quantizedMatrixBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedMatrixBench)

add_executable(quantizedTransferSpmmBench
    bench/quantized_transfer_spmm_bench.cu
)
target_include_directories(quantizedTransferSpmmBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedTransferSpmmBench PRIVATE cellerator_compute_autograd CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(quantizedTransferSpmmBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedTransferSpmmBench)

add_executable(trajectoryCompileTest
    tests/trajectory_compile_test.cu
)
target_include_directories(trajectoryCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trajectoryCompileTest PRIVATE CUDA::cudart)
set_target_properties(trajectoryCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(trajectoryCompileTest)

add_executable(trajectoryRuntimeTest
    tests/trajectory_runtime_test.cu
)
target_include_directories(trajectoryRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trajectoryRuntimeTest PRIVATE CUDA::cudart)
set_target_properties(trajectoryRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(trajectoryRuntimeTest)

add_library(cellerator_compute_exact_search STATIC
    src/compute/neighbors/exact_search/exact_search.cu
)
target_include_directories(cellerator_compute_exact_search PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_exact_search PRIVATE CUDA::cudart)
set_target_properties(cellerator_compute_exact_search PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_exact_search)

add_library(cellerator_neighbor_math INTERFACE)
target_link_libraries(cellerator_neighbor_math INTERFACE cellerator_compute_exact_search)

add_library(cellerator_compute_autograd STATIC
    src/compute/ml/autograd/runtime.cu
    src/compute/ml/autograd/kernels/base_sparse.cu
    src/compute/ml/autograd/kernels/dist_sparse.cu
)
target_include_directories(cellerator_compute_autograd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_autograd PUBLIC CellShard::runtime)
target_link_libraries(cellerator_compute_autograd PRIVATE CUDA::cudart CUDA::cusparse)
cellerator_link_nccl_if_available(cellerator_compute_autograd)
set_target_properties(cellerator_compute_autograd PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_autograd)

add_library(cellerator_sparse_math INTERFACE)
target_link_libraries(cellerator_sparse_math INTERFACE cellerator_compute_autograd)
add_library(cellerator_sparse_linalg INTERFACE)
target_link_libraries(cellerator_sparse_linalg INTERFACE cellerator_compute_autograd)
add_library(cellerator_sparse_ml INTERFACE)
target_link_libraries(cellerator_sparse_ml INTERFACE cellerator_compute_autograd)

add_executable(computeAutogradBench
    bench/compute_autograd_bench.cu
)
target_include_directories(computeAutogradBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(computeAutogradBench PRIVATE cellerator_compute_autograd CUDA::cudart CUDA::cusparse)
cellerator_link_nccl_if_available(computeAutogradBench)
set_target_properties(computeAutogradBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(computeAutogradBench)

add_executable(exactSearchRuntimeTest
    tests/exact_search_runtime_test.cu
)
target_include_directories(exactSearchRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(exactSearchRuntimeTest PRIVATE cellerator_compute_exact_search CUDA::cudart)
set_target_properties(exactSearchRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(exactSearchRuntimeTest)

add_executable(computeAutogradRuntimeTest
    tests/compute_autograd_runtime_test.cu
)
target_include_directories(computeAutogradRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(computeAutogradRuntimeTest PRIVATE cellerator_compute_autograd CellShard::runtime CUDA::cudart CUDA::cusparse)
cellerator_link_nccl_if_available(computeAutogradRuntimeTest)
set_target_properties(computeAutogradRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(computeAutogradRuntimeTest)

add_library(cellerator_state_reduce_cuda STATIC
    src/models/state_reduce/state_reduce_cuda.cu
)
target_include_directories(cellerator_state_reduce_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_state_reduce_cuda PRIVATE cellerator_compute_autograd CellShard::runtime CUDA::cudart CUDA::cusparse)
cellerator_link_nccl_if_available(cellerator_state_reduce_cuda)
set_target_properties(cellerator_state_reduce_cuda PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_state_reduce_cuda)

add_executable(stateReduceRuntimeTest
    tests/state_reduce_runtime_test.cu
)
target_include_directories(stateReduceRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stateReduceRuntimeTest PRIVATE cellerator_state_reduce_cuda cellerator_compute_autograd CellShard::runtime CUDA::cudart CUDA::cusparse)
cellerator_link_nccl_if_available(stateReduceRuntimeTest)
set_target_properties(stateReduceRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(stateReduceRuntimeTest)

add_library(cellerator_developmental_time_cuda STATIC
    src/models/developmental_time/dT_model.cu
)
target_include_directories(cellerator_developmental_time_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_developmental_time_cuda PRIVATE cellerator_compute_autograd CellShard::runtime CUDA::cudart CUDA::cusparse)
cellerator_link_nccl_if_available(cellerator_developmental_time_cuda)
set_target_properties(cellerator_developmental_time_cuda PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_developmental_time_cuda)

add_library(cellerator_developmental_time_trajectory_cuda STATIC
    src/models/developmental_time_trajectory/dTT_model.cu
)
target_include_directories(cellerator_developmental_time_trajectory_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_developmental_time_trajectory_cuda PRIVATE cellerator_developmental_time_cuda cellerator_compute_autograd CellShard::runtime CUDA::cudart CUDA::cusparse)
cellerator_link_nccl_if_available(cellerator_developmental_time_trajectory_cuda)
set_target_properties(cellerator_developmental_time_trajectory_cuda PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_developmental_time_trajectory_cuda)

add_executable(developmentalTimeCompileTest
    tests/developmental_time_compile_test.cc
)
target_include_directories(developmentalTimeCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(developmentalTimeCompileTest PRIVATE CellShard::runtime CUDA::cudart)
set_target_properties(developmentalTimeCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(developmentalTimeCompileTest)

add_executable(developmentalTimeRuntimeTest
    tests/developmental_time_runtime_test.cu
)
target_include_directories(developmentalTimeRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(developmentalTimeRuntimeTest PRIVATE cellerator_developmental_time_cuda cellerator_compute_autograd CellShard::runtime CUDA::cudart CUDA::cusparse)
cellerator_link_nccl_if_available(developmentalTimeRuntimeTest)
set_target_properties(developmentalTimeRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(developmentalTimeRuntimeTest)

add_executable(developmentalTimeTrajectoryRuntimeTest
    tests/developmental_time_trajectory_runtime_test.cu
)
target_include_directories(developmentalTimeTrajectoryRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(developmentalTimeTrajectoryRuntimeTest PRIVATE cellerator_developmental_time_trajectory_cuda cellerator_developmental_time_cuda cellerator_compute_autograd CellShard::runtime CUDA::cudart CUDA::cusparse)
cellerator_link_nccl_if_available(developmentalTimeTrajectoryRuntimeTest)
set_target_properties(developmentalTimeTrajectoryRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(developmentalTimeTrajectoryRuntimeTest)

if(CELLERATOR_ENABLE_TORCH_MODELS)
    add_library(cellerator_quantize_cuda STATIC
        src/models/quantize/quantize_cuda.cu
    )
    target_include_directories(cellerator_quantize_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(cellerator_quantize_cuda PRIVATE cellerator_torch cellerator_compute_autograd CUDA::cudart)
    set_target_properties(cellerator_quantize_cuda PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    cellerator_enable_perf(cellerator_quantize_cuda)

    add_executable(torchBindingsCompileTest
        tests/torch_bindings_compile_test.cc
    )
    target_include_directories(torchBindingsCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(torchBindingsCompileTest PRIVATE cellerator_torch_bindings)
    set_target_properties(torchBindingsCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(torchBindingsCompileTest)

    add_executable(denseReduceCompileTest
        tests/dense_reduce_compile_test.cc
    )
    target_include_directories(denseReduceCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(denseReduceCompileTest PRIVATE cellerator_torch cellerator_compute_model_ops CellShard::runtime)
    set_target_properties(denseReduceCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(denseReduceCompileTest)

    add_executable(quantizeModelTest
        tests/quantize_model_test.cc
    )
    target_include_directories(quantizeModelTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(quantizeModelTest PRIVATE cellerator_torch cellerator_compute_model_ops cellerator_quantize_cuda CUDA::cudart)
    set_target_properties(quantizeModelTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(quantizeModelTest)

    add_executable(modelCustomOpsTest
        tests/model_custom_ops_test.cc
    )
    target_include_directories(modelCustomOpsTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(modelCustomOpsTest PRIVATE cellerator_torch cellerator_compute_model_ops CellShard::runtime)
    set_target_properties(modelCustomOpsTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(modelCustomOpsTest)
endif()
