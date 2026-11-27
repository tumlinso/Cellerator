add_library(cellerator_preparation_factory STATIC
    src/execution/preparation/preparation_factory.cc
)
add_library(Cellerator::preparation_factory ALIAS
    cellerator_preparation_factory)
target_include_directories(cellerator_preparation_factory PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_preparation_factory PUBLIC
    Cellerator::builtin_candidate_catalog
    Cellerator::execution_projection_binding
    Cellerator::runtime
)
target_compile_features(cellerator_preparation_factory PUBLIC cxx_std_20)
