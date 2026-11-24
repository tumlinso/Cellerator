cmake_minimum_required(VERSION 3.28)
project(ccell_consumer LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
enable_testing()

find_package(ccell CONFIG REQUIRED)
find_package(ccell CONFIG REQUIRED COMPONENTS preprocess velocity)
if(TARGET anndatapp::anndatapp OR TARGET Cellerator::anndata)
  message(FATAL_ERROR "Default CCell discovery imported an optional data dependency.")
endif()

add_executable(consumer_umbrella ../umbrella.cpp)
target_link_libraries(consumer_umbrella PRIVATE ccell::ccell)
add_test(NAME consumer-umbrella COMMAND consumer_umbrella)

add_executable(consumer_components ../components.cpp)
target_link_libraries(consumer_components PRIVATE ccell::preprocess ccell::velocity)
add_test(NAME consumer-components COMMAND consumer_components)

add_executable(consumer_modeling ../../examples/modeling.cpp)
target_link_libraries(consumer_modeling PRIVATE ccell::preprocess)
add_test(NAME consumer-modeling COMMAND consumer_modeling)

add_executable(consumer_velocity ../../src/velocity/tests/consumer/main.cpp)
target_link_libraries(consumer_velocity PRIVATE ccell::velocity)
add_test(NAME consumer-velocity COMMAND consumer_velocity)

if(CCELL_TEST_ANNDATA_MODELING)
  find_package(Cellerator CONFIG REQUIRED COMPONENTS anndata)
  add_executable(consumer_anndata ../anndata_modeling.cpp)
  target_link_libraries(consumer_anndata PRIVATE Cellerator::anndata ccell::preprocess ccell::velocity)
  add_test(NAME consumer-anndata COMMAND consumer_anndata)
endif()
