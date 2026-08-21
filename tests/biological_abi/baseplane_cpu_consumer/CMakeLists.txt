cmake_minimum_required(VERSION 3.24)
project(CelleratorBiologicalAbiBaseplaneCpuConsumer LANGUAGES CXX)

if(NOT CELLERATOR_INCLUDE_DIR OR NOT BASEPLANE_INCLUDE_DIR)
    message(FATAL_ERROR
        "Set CELLERATOR_INCLUDE_DIR and BASEPLANE_INCLUDE_DIR")
endif()

add_executable(celleratorBiologicalAbiBaseplaneCpuConsumer main.cc)
target_include_directories(celleratorBiologicalAbiBaseplaneCpuConsumer PRIVATE
    ${CELLERATOR_INCLUDE_DIR}
    ${BASEPLANE_INCLUDE_DIR}
)
target_compile_features(celleratorBiologicalAbiBaseplaneCpuConsumer PRIVATE
    cxx_std_17)
