cmake_minimum_required(VERSION 3.28)

project(CelleratorTensorModuleConsumer LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

option(CELLERATOR_TENSOR_MODULE_TEST_CUDA
    "Compile the tensor header bridge with NVCC" OFF)

if(NOT TARGET Cellerator::tensor)
    find_package(Cellerator CONFIG REQUIRED COMPONENTS native_data)
endif()
add_executable(cellerator_tensor_module_consumer module_consumer.cc header_bridge.cc)
set_source_files_properties(module_consumer.cc PROPERTIES CXX_SCAN_FOR_MODULES ON)
target_link_libraries(cellerator_tensor_module_consumer PRIVATE Cellerator::tensor)

if(CELLERATOR_TENSOR_MODULE_TEST_CUDA)
    enable_language(CUDA)
    target_sources(cellerator_tensor_module_consumer PRIVATE cuda_bridge.cu)
    target_compile_definitions(cellerator_tensor_module_consumer PRIVATE CELLERATOR_TENSOR_MODULE_TEST_CUDA)
    set_target_properties(cellerator_tensor_module_consumer PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
endif()

enable_testing()
add_test(NAME cellerator_tensor_module_consumer COMMAND cellerator_tensor_module_consumer)
