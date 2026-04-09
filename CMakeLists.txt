cmake_minimum_required(VERSION 3.20)

project(Cellerator LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS
        /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/lib
)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

function(cellerator_enable_perf target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG -march=native -mtune=native>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda -Xptxas=-O3,-dlcm=ca>
    )
    set_target_properties(${target} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE YES
    )
endfunction()

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

add_library(seriesIngestCompileTest OBJECT
    tests/series_ingest_compile_test.cu
)
target_include_directories(seriesIngestCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(seriesIngestCompileTest PRIVATE ${HDF5_INCLUDE_DIRS})
set_target_properties(seriesIngestCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(seriesIngestCompileTest)
