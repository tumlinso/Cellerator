add_library(cellerator_indexing STATIC
    temporal_occurrences.cc
    "${PROJECT_SOURCE_DIR}/src/containers/posting_index.cc"
    "${PROJECT_SOURCE_DIR}/src/geometry/dependency/posting_index.cc")
add_library(Cellerator::indexing ALIAS cellerator_indexing)
target_include_directories(cellerator_indexing PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_indexing PUBLIC Cellerator::support)
target_compile_features(cellerator_indexing PUBLIC cxx_std_17)
set_target_properties(cellerator_indexing PROPERTIES POSITION_INDEPENDENT_CODE ON)
