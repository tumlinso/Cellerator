add_library(cellpack_hardware_cost_model STATIC
    ${PROJECT_SOURCE_DIR}/src/execution/planning/packing/hardware_cost_model.cc
)
add_library(CellPack::hardware_cost_model ALIAS cellpack_hardware_cost_model)
target_include_directories(cellpack_hardware_cost_model PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_hardware_cost_model PUBLIC CellPack::cellpack)
set_target_properties(cellpack_hardware_cost_model PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_hardware_cost_model)
