# CE-ARCH-40: Cellerator-owned preparation and execution over frozen Baseplane
# predicate contracts. Baseplane remains independently buildable and owns no
# Cellerator runtime or numerical policy.
add_library(cellerator_sequence_integration STATIC
    src/interop/baseplane/baseplane_strategy.cc
    src/interop/baseplane/baseplane_integration.cu
)
add_library(Cellerator::sequence_integration ALIAS
    cellerator_sequence_integration)
target_include_directories(cellerator_sequence_integration PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_sequence_integration PUBLIC
    Cellerator::operation_core
    Baseplane::seq
    CUDA::cudart
)
set_target_properties(cellerator_sequence_integration PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
