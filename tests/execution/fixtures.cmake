add_library(cellerator_planner_input_fixtures STATIC ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    tests/fixtures/planner/live_planner_inputs.cc
)
target_include_directories(cellerator_planner_input_fixtures PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellerator_planner_input_fixtures PUBLIC
    Cellerator::execution_planning
)
target_compile_features(cellerator_planner_input_fixtures PUBLIC cxx_std_20)

add_library(cellerator_quantitative_relation_fixtures STATIC ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    tests/fixtures/model_execution/quantitative_relation_fixture.cc
)
target_include_directories(cellerator_quantitative_relation_fixtures PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellerator_quantitative_relation_fixtures PUBLIC
    Cellerator::identity_registry
)
target_compile_features(cellerator_quantitative_relation_fixtures PUBLIC cxx_std_20)
