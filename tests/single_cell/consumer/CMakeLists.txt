cmake_minimum_required(VERSION 3.28)
project(CelleratorMethodConsumer LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
enable_testing()

find_package(Cellerator CONFIG REQUIRED COMPONENTS preprocess velocity)
if(TARGET anndatapp::anndatapp OR TARGET Cellerator::anndata)
    message(FATAL_ERROR "Native method discovery imported the optional AnnData dependency")
endif()

add_executable(cellerator_method_consumer ../methods.cpp)
target_link_libraries(cellerator_method_consumer PRIVATE Cellerator::preprocess Cellerator::velocity)
add_test(NAME cellerator-method-consumer COMMAND cellerator_method_consumer)
