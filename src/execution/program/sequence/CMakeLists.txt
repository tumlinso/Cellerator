add_library(cellerator_sequence_integration STATIC
    src/execution/planning/sequence/strategy.cc
    src/execution/program/sequence/regulatory.cu)
target_link_libraries(cellerator_sequence_integration PUBLIC
    Cellerator::operation_core Cellerator::sequence CUDA::cudart)
target_include_directories(cellerator_sequence_integration PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
set_target_properties(cellerator_sequence_integration PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
