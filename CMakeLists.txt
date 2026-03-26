cmake_minimum_required(VERSION 3.20)

project(Cellerator LANGUAGES CXX CUDA)

add_executable(matrixTest
    tests/matrix_test.cc
    src/matrix/matrix_io.cc
)
set_source_files_properties(
    tests/matrix_test.cc
    src/matrix/matrix_io.cc
    PROPERTIES LANGUAGE CUDA
)
target_include_directories(matrixTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(matrixTest PRIVATE cxx_std_17)
set_target_properties(matrixTest PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
