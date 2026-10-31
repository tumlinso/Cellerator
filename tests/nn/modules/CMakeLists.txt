cmake_minimum_required(VERSION 3.28)

project(CelleratorNNModuleConsumer LANGUAGES CXX)

option(CELLERATOR_NN_MODULES_TEST_CUDA
    "Compile the ordinary-header ABI bridge with NVCC; this does not launch CUDA" OFF)

if(CELLERATOR_NN_MODULES_TEST_CUDA)
    # Torch's Caffe2 config selects CMAKE_CXX_COMPILER as NVCC's host
    # compiler when Clang is used for CXX. Preserve the explicitly selected
    # host compiler across package discovery, then enable CUDA with that ABI.
    set(_cellerator_cuda_host_compiler "${CMAKE_CUDA_HOST_COMPILER}")
    find_package(Cellerator CONFIG REQUIRED COMPONENTS nn nn_cuda)
    if(_cellerator_cuda_host_compiler)
        set(CMAKE_CUDA_HOST_COMPILER "${_cellerator_cuda_host_compiler}")
    endif()
    enable_language(CUDA)
else()
    find_package(Cellerator CONFIG REQUIRED COMPONENTS nn)
endif()
find_package(CelleratorNNModules CONFIG REQUIRED)
if(CELLERATOR_NN_MODULES_TEST_CUDA AND NOT TARGET Cellerator::nn_cuda)
    message(FATAL_ERROR
        "The Cellerator package must enable CELLERATOR_ENABLE_NN_CUDA")
endif()

add_executable(cellerator_nn_module_consumer
    import_consumer.cc
    module_only.cc
    include_bridge.cc)
if(CELLERATOR_NN_MODULES_TEST_CUDA)
    target_sources(cellerator_nn_module_consumer PRIVATE cuda_bridge.cu)
    target_compile_definitions(cellerator_nn_module_consumer
        PRIVATE CELLERATOR_NN_MODULES_TEST_CUDA=1)
    set_target_properties(cellerator_nn_module_consumer PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED YES)
endif()

target_compile_features(cellerator_nn_module_consumer PRIVATE cxx_std_20)
target_link_libraries(cellerator_nn_module_consumer
    PRIVATE
        CelleratorNNModules::nn)

enable_testing()
add_test(NAME cellerator_nn_module_consumer
         COMMAND cellerator_nn_module_consumer)
