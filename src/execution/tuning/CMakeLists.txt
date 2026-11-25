add_library(cellerator_execution_tuning_cuda STATIC
    src/execution/tuning/candidate_measurement.cu
)
add_library(Cellerator::execution_tuning_cuda ALIAS cellerator_execution_tuning_cuda)
target_include_directories(cellerator_execution_tuning_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_execution_tuning_cuda PUBLIC
    Cellerator::execution_planning
    Cellerator::operation_core
    CUDA::cudart
)
set_target_properties(cellerator_execution_tuning_cuda PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
