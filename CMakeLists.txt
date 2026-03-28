cmake_minimum_required(VERSION 3.20)

project(Cellerator LANGUAGES CXX CUDA)

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
target_compile_features(matrixTest PRIVATE cxx_std_17)
set_target_properties(matrixTest PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)

add_executable(convertEmbryoMtxShards
    src/load_embryo_mtx_shards.cu
    src/matrix/matrix_io.cu
)
target_include_directories(convertEmbryoMtxShards PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(convertEmbryoMtxShards PRIVATE cxx_std_17)
set_target_properties(convertEmbryoMtxShards PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_compile_options(convertEmbryoMtxShards PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)
target_link_options(convertEmbryoMtxShards PRIVATE -fopenmp)
