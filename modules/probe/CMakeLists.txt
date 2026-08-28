cmake_minimum_required(VERSION 3.28)

project(CelleratorModuleProbe LANGUAGES CXX)

add_library(cellerator_module_probe)
target_compile_features(cellerator_module_probe PUBLIC cxx_std_20)
target_sources(cellerator_module_probe
    PUBLIC
        FILE_SET CXX_MODULES FILES
            module_probe.ccm)

add_executable(cellerator_module_probe_test module_probe_test.cc)
target_compile_features(cellerator_module_probe_test PRIVATE cxx_std_20)
target_link_libraries(cellerator_module_probe_test
    PRIVATE cellerator_module_probe)
