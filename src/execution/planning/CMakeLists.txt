add_library(cellerator_execution_planning STATIC
    src/execution/planning/operation_planning.cc
    src/execution/planning/distributed/hierarchy.cc
    src/execution/planning/layout_cost.cc
    src/execution/planning/layout_cost_calibration.cc
    src/execution/planning/candidates/connected_economics_v1.cc
    src/execution/planning/candidates/candidate_workspace_v1.cc
    src/execution/planning/candidates/local_pareto_frontier_v1.cc
    src/execution/planning/candidates/pareto_portfolio_v1.cc
    src/execution/planning/candidates/validation_v1.cc
    src/execution/planning/resource/planning_resources_v1.cc
    src/execution/planning/external_cost/compiler_exchange_v1.cc
    src/execution/planning/external_cost/complete_cost_v1.cc
    src/execution/planning/external_cost/frontier_v1.cc
    src/execution/planning/external_cost/geometry_objective_v1.cc
    src/execution/planning/external_cost/pricing_oracle_v1.cc
    src/execution/planning/external_cost/vector_v1.cc
)
add_library(Cellerator::execution_planning ALIAS cellerator_execution_planning)
target_include_directories(cellerator_execution_planning PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_execution_planning PUBLIC
    Cellerator::operation_core
    Cellerator::cellpack_geometry
    CellPack::alternating_refinement
)
target_compile_features(cellerator_execution_planning PUBLIC cxx_std_20)
