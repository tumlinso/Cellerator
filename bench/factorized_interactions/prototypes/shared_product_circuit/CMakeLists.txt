cmake_minimum_required(VERSION 3.24)
project(CelleratorSharedProductCircuit LANGUAGES CXX CUDA)

if(NOT DEFINED CELLERATOR_SOURCE_ROOT)
    get_filename_component(CELLERATOR_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70)
find_package(CUDAToolkit REQUIRED)

# Deliberately private standalone build: no public targets, Torch, package
# install, hardware discovery, or repository-wide CMake registration.
add_executable(shared_product_circuit_runner
    circuit_runner.cu
    "${CELLERATOR_SOURCE_ROOT}/src/layout/factorized_interaction.cc"
    "${CELLERATOR_SOURCE_ROOT}/src/memory/cuda/resources.cu"
    "${CELLERATOR_SOURCE_ROOT}/src/modeling/cuda/factorized_interaction.cu"
    "${CELLERATOR_SOURCE_ROOT}/src/modeling/cuda/factorized_interaction_direct.cu"
    "${CELLERATOR_SOURCE_ROOT}/src/modeling/cuda/factorized_interaction_stable.cu"
    "${CELLERATOR_SOURCE_ROOT}/src/modeling/cuda/prepared_factorized_interaction.cu")
target_include_directories(shared_product_circuit_runner PRIVATE
    "${CELLERATOR_SOURCE_ROOT}/include")
target_compile_options(shared_product_circuit_runner PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--ftz=false;--prec-div=true;--prec-sqrt=true>)
target_link_libraries(shared_product_circuit_runner PRIVATE CUDA::cudart)
