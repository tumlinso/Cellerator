cmake_minimum_required(VERSION 3.20)

project(Cellerator LANGUAGES CXX)

add_executable(matrixTest
    tests/matrix_test.cc
    src/matrix/matrix_io.cc
)
target_include_directories(matrixTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(matrixTest PRIVATE cxx_std_17)
