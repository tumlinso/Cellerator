add_library(cellerator_planner STATIC
    src/planner/end_to_end_planner.cc
    src/planner/distributed/hierarchy.cc
    src/geometry/optimization/objective_v2.cc
    src/planner/objective_v2_calibration.cc
)
add_library(Cellerator::planner ALIAS cellerator_planner)
target_include_directories(cellerator_planner PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_planner PUBLIC
    Cellerator::operation_core
    CellPack::semantic_geometry
    CellPack::alternating_refinement
)
target_compile_features(cellerator_planner PUBLIC cxx_std_17)

add_library(cellerator_planner_input_fixtures STATIC
    tests/fixtures/planner/live_planner_inputs.cc
)
target_include_directories(cellerator_planner_input_fixtures PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellerator_planner_input_fixtures PUBLIC
    Cellerator::planner
)
target_compile_features(cellerator_planner_input_fixtures PUBLIC cxx_std_17)

add_library(cellerator_quantitative_relation_fixtures STATIC
    tests/fixtures/model_execution/quantitative_relation_fixture.cc
)
target_include_directories(cellerator_quantitative_relation_fixtures PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellerator_quantitative_relation_fixtures PUBLIC
    Cellerator::identity_registry
)
target_compile_features(cellerator_quantitative_relation_fixtures PUBLIC cxx_std_17)

add_library(cellerator_candidate_measurement STATIC
    src/planner/candidate_measurement.cu
)
add_library(Cellerator::candidate_measurement ALIAS
    cellerator_candidate_measurement)
target_include_directories(cellerator_candidate_measurement PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_candidate_measurement PUBLIC
    Cellerator::planner
    Cellerator::operation_core
    CUDA::cudart
)
set_target_properties(cellerator_candidate_measurement PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
