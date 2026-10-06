cmake_minimum_required(VERSION 3.24)
project(cellerator_learning_installed_consumer LANGUAGES CXX)
find_package(Cellerator CONFIG REQUIRED COMPONENTS torch)
add_executable(cellerator_learning_installed_consumer
    ../../tests/bindings/torch/program_ops_test.cc)
target_link_libraries(cellerator_learning_installed_consumer PRIVATE Cellerator::torch)
target_compile_features(cellerator_learning_installed_consumer PRIVATE cxx_std_20)
