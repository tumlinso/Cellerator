add_library(cellpack STATIC
    src/planner.cc
)
add_library(CellPack::cellpack ALIAS cellpack)
target_include_directories(cellpack PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellpack PUBLIC cxx_std_17)
set_target_properties(cellpack PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellpack)
endif()

add_executable(cellPackFormatTest
    tests/format_test.cc
)
target_link_libraries(cellPackFormatTest PRIVATE CellPack::cellpack)
set_target_properties(cellPackFormatTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackFormatTest)
endif()

add_executable(cellPackPlannerTest
    tests/planner_test.cc
)
target_link_libraries(cellPackPlannerTest PRIVATE CellPack::cellpack)
set_target_properties(cellPackPlannerTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackPlannerTest)
endif()
