cmake_minimum_required(VERSION 3.24)
project(cellerator_learning_installed_consumer LANGUAGES CXX)
find_package(CelleraTorch CONFIG REQUIRED)
add_executable(ce_ml2_installed_consumer
    ../../components/CelleraTorch/tests/mechanism_test.cc)
target_link_libraries(ce_ml2_installed_consumer PRIVATE CelleraTorch::mechanism)
target_compile_features(ce_ml2_installed_consumer PRIVATE cxx_std_20)
