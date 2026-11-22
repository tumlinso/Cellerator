cmake_minimum_required(VERSION 3.28)

project(CelleratorNNModuleConsumer LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

option(CELLERATOR_NN_MODULES_TEST_CUDA
    "Compile the ordinary-header ABI bridge with NVCC; this does not launch CUDA" OFF)

if(CELLERATOR_NN_MODULES_TEST_CUDA)
    find_package(Cellerator CONFIG REQUIRED COMPONENTS nn nn_cuda)
    enable_language(CUDA)
else()
    find_package(Cellerator CONFIG REQUIRED COMPONENTS nn)
endif()
if(CELLERATOR_NN_MODULES_TEST_CUDA AND NOT TARGET Cellerator::nn_cuda)
    message(FATAL_ERROR
        "The Cellerator package must enable CELLERATOR_ENABLE_NN_CUDA")
endif()

add_executable(cellerator_nn_module_consumer
    import_consumer.cc
    module_only.cc
    include_bridge.cc)
set_source_files_properties(import_consumer.cc module_only.cc
    PROPERTIES CXX_SCAN_FOR_MODULES ON)
if(CELLERATOR_NN_MODULES_TEST_CUDA)
    target_sources(cellerator_nn_module_consumer PRIVATE cuda_bridge.cu)
    target_compile_definitions(cellerator_nn_module_consumer
        PRIVATE CELLERATOR_NN_MODULES_TEST_CUDA=1)
    set_target_properties(cellerator_nn_module_consumer PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED YES)
endif()

target_compile_features(cellerator_nn_module_consumer PRIVATE cxx_std_20)
target_link_libraries(cellerator_nn_module_consumer PRIVATE Cellerator::nn)

enable_testing()
add_test(NAME cellerator_nn_module_consumer
         COMMAND cellerator_nn_module_consumer)
