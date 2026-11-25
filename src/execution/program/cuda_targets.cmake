add_library(cellerator_execution_core INTERFACE)
add_library(Cellerator::execution_core ALIAS cellerator_execution_core)
target_link_libraries(cellerator_execution_core INTERFACE
    Cellerator::execution_projection_binding
    Cellerator::builtin_candidate_catalog
    Cellerator::relation_catalog
    Cellerator::preparation_factory
    Cellerator::compute_candidate_inventory
)

add_library(cellerator_execution_program STATIC
    src/execution/program/operation.cc
    $<TARGET_OBJECTS:cellerator_execution_stage_program>
)
add_library(Cellerator::execution_program ALIAS
    cellerator_execution_program)
target_include_directories(cellerator_execution_program PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_execution_program PUBLIC
    Cellerator::execution_core
    Cellerator::execution_planning
)
target_compile_features(cellerator_execution_program PUBLIC cxx_std_20)

add_library(cellerator_execution_feature_major_training STATIC
    src/execution/program/feature_major_training.cu
)
add_library(Cellerator::execution_feature_major_training ALIAS cellerator_execution_feature_major_training)
target_include_directories(cellerator_execution_feature_major_training PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_execution_feature_major_training PUBLIC
    Cellerator::native_training_slice
    Cellerator::feature_major_projection
    Cellerator::transpose_projection
    Cellerator::runtime
    CUDA::cudart
)
set_target_properties(cellerator_execution_feature_major_training PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)

add_library(cellerator_execution_training_stages STATIC
    src/execution/program/training/graph_capture.cc
    src/execution/program/training/program.cc
)
add_library(Cellerator::execution_training_stages ALIAS cellerator_execution_training_stages)
target_include_directories(cellerator_execution_training_stages PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(cellerator_execution_training_stages PUBLIC cxx_std_20)
