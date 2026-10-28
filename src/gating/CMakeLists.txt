add_library(cellerator_gating STATIC
    condition_program.cc
    derivative_sampling.cc)
add_library(Cellerator::gating ALIAS cellerator_gating)
target_include_directories(cellerator_gating PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_gating PUBLIC Cellerator::support Cellerator::indexing)
target_compile_features(cellerator_gating PUBLIC cxx_std_17)
set_target_properties(cellerator_gating PROPERTIES POSITION_INDEPENDENT_CODE ON)
