add_library(cellerator_sparse_support STATIC
    logical_support.cc
    adaptive_support.cc
    ../indexing/reverse_postings.cc
    ../indexing/temporal_occurrences.cc
    ../gating/condition_program.cc
    ../gating/derivative_sampling.cc
    proposal_discovery.cc
    ../projection/physical_mma_hybrid_builder.cc
    ../projection/mma_physical_cover_validation.cc)
add_library(Cellerator::sparse_support ALIAS cellerator_sparse_support)
target_include_directories(cellerator_sparse_support PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(cellerator_sparse_support PUBLIC cxx_std_17)
set_target_properties(cellerator_sparse_support PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
