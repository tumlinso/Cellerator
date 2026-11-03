cmake_minimum_required(VERSION 3.24)

project(CelleratorFactorizedInteractionRunner LANGUAGES CXX CUDA)

if(NOT TARGET Cellerator::factorized_interaction_cuda)
    find_package(Cellerator CONFIG REQUIRED COMPONENTS factorized_interaction_cuda)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

add_executable(cellerator_factorized_interaction_runner native_runner.cu)
target_compile_features(cellerator_factorized_interaction_runner PRIVATE cxx_std_17)
set_target_properties(cellerator_factorized_interaction_runner PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
target_link_libraries(cellerator_factorized_interaction_runner
    PRIVATE Cellerator::factorized_interaction_cuda)
