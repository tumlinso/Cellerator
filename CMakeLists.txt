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

find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

option(CELLERATOR_ENABLE_TORCH_MODELS "Enable libtorch-backed model targets" ON)
option(CELLERATOR_ENABLE_CUDA_LINEINFO "Emit CUDA line mapping for Nsight profiling" OFF)

if(CELLERATOR_ENABLE_TORCH_MODELS AND NOT Torch_DIR AND DEFINED ENV{LIBTORCH_PATH})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{LIBTORCH_PATH}")
endif()

if(CELLERATOR_ENABLE_TORCH_MODELS)
    find_package(Torch REQUIRED)
endif()

find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS
        /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/lib
)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
    target_link_libraries(cellerator_torch_bindings INTERFACE cellerator_torch)
    target_compile_features(cellerator_torch_bindings INTERFACE cxx_std_17)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix/matrix_io.cu)
    add_executable(matrixTest
        tests/matrix_test.cc
        src/matrix/matrix_io.cu
    )
    set_source_files_properties(
        tests/matrix_test.cc
        src/matrix/matrix_io.cu
        PROPERTIES LANGUAGE CUDA
    )
    target_include_directories(matrixTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(matrixTest PRIVATE CUDA::cusparse)
    target_compile_features(matrixTest PRIVATE cxx_std_17)
    set_target_properties(matrixTest PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    cellerator_enable_perf(matrixTest)

    add_executable(convertEmbryoMtxShards
        src/load_embryo_mtx_shards.cu
        src/matrix/matrix_io.cu
    )
    target_include_directories(convertEmbryoMtxShards PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(convertEmbryoMtxShards PRIVATE CUDA::cusparse)
    target_compile_features(convertEmbryoMtxShards PRIVATE cxx_std_17)
    set_target_properties(convertEmbryoMtxShards PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    cellerator_enable_perf(convertEmbryoMtxShards)
    target_compile_options(convertEmbryoMtxShards PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp,-march=native,-mtune=native>
    )
    target_link_options(convertEmbryoMtxShards PRIVATE -fopenmp)
else()
    message(WARNING "Skipping legacy matrix targets because src/matrix/matrix_io.cu is not present in this checkout.")
endif()

add_executable(scrnaPreprocessBench
    bench/scrna_preprocess_bench.cu
    extern/CellShard/src/disk/matrix.cu
    extern/CellShard/src/sharded/disk.cu
    extern/CellShard/src/sharded/shard_paths.cu
    extern/CellShard/src/sharded/series_h5.cc
)
target_include_directories(scrnaPreprocessBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(scrnaPreprocessBench PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(scrnaPreprocessBench PRIVATE CUDA::cusparse ${HDF5_LIBRARIES})
if(NCCL_LIBRARY)
    target_link_libraries(scrnaPreprocessBench PRIVATE ${NCCL_LIBRARY})
endif()
set_target_properties(scrnaPreprocessBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(scrnaPreprocessBench)

add_executable(cellShardSeriesH5Test
    tests/cellshard_series_h5_test.cu
    extern/CellShard/src/disk/matrix.cu
    extern/CellShard/src/sharded/disk.cu
    extern/CellShard/src/sharded/shard_paths.cu
    extern/CellShard/src/sharded/series_h5.cc
)
target_include_directories(cellShardSeriesH5Test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cellShardSeriesH5Test PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(cellShardSeriesH5Test PRIVATE ${HDF5_LIBRARIES})
set_target_properties(cellShardSeriesH5Test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellShardSeriesH5Test)

add_executable(quantizedMatrixTest
    tests/quantized_matrix_test.cu
)
target_include_directories(quantizedMatrixTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedMatrixTest PRIVATE CUDA::cudart)
set_target_properties(quantizedMatrixTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedMatrixTest)

add_library(seriesIngestCompileTest OBJECT
    tests/series_ingest_compile_test.cu
)
target_include_directories(seriesIngestCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(seriesIngestCompileTest PRIVATE ${HDF5_INCLUDE_DIRS})
set_target_properties(seriesIngestCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(seriesIngestCompileTest)

add_executable(quantizedMatrixBench
    bench/quantized_matrix_bench.cu
)
target_include_directories(quantizedMatrixBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedMatrixBench PRIVATE CUDA::cudart)
set_target_properties(quantizedMatrixBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedMatrixBench)

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

if(CELLERATOR_ENABLE_TORCH_MODELS)
    add_executable(developmentalTimeCompileTest
        tests/developmental_time_compile_test.cc
    )
    target_include_directories(developmentalTimeCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(developmentalTimeCompileTest PRIVATE cellerator_torch)
    set_target_properties(developmentalTimeCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(developmentalTimeCompileTest)

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
    target_link_libraries(denseReduceCompileTest PRIVATE cellerator_torch)
    set_target_properties(denseReduceCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(denseReduceCompileTest)

    add_executable(forwardNeighborsCompileTest
        tests/forward_neighbors_compile_test.cc
    )
    target_include_directories(forwardNeighborsCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(forwardNeighborsCompileTest PRIVATE cellerator_torch)
    set_target_properties(forwardNeighborsCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(forwardNeighborsCompileTest)

    add_executable(quantizeModelTest
        tests/quantize_model_test.cc
    )
    target_include_directories(quantizeModelTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(quantizeModelTest PRIVATE cellerator_torch)
    set_target_properties(quantizeModelTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(quantizeModelTest)
endif()
