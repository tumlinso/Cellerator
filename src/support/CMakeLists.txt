add_library(cellerator_support STATIC
    adaptive_support.cc
    proposal_discovery.cc
    "${PROJECT_SOURCE_DIR}/src/geometry/dependency/compact_support.cc"
    "${PROJECT_SOURCE_DIR}/src/execution/dependency/support_lifecycle.cc")
add_library(Cellerator::support ALIAS cellerator_support)
target_include_directories(cellerator_support PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(cellerator_support PUBLIC cxx_std_20)
set_target_properties(cellerator_support PROPERTIES POSITION_INDEPENDENT_CODE ON)
