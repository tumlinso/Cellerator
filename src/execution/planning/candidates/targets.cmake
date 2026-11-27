add_library(cellerator_compute_candidate_inventory STATIC
    src/execution/planning/candidates/compute_candidate_inventory.cc)
add_library(Cellerator::compute_candidate_inventory ALIAS
    cellerator_compute_candidate_inventory)
target_include_directories(cellerator_compute_candidate_inventory PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(cellerator_compute_candidate_inventory PUBLIC
    Cellerator::architecture_provider)
target_compile_features(cellerator_compute_candidate_inventory PUBLIC cxx_std_20)

add_executable(cellerator_compute_candidate_inventory_test ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    src/execution/planning/candidates/compute_candidate_inventory_test.cc)
target_link_libraries(cellerator_compute_candidate_inventory_test PRIVATE
    Cellerator::compute_candidate_inventory)
target_compile_features(cellerator_compute_candidate_inventory_test PRIVATE cxx_std_20)
target_compile_options(cellerator_compute_candidate_inventory_test PRIVATE -UNDEBUG)

set(CELLERATOR_OPERATION_FOCUSED_TESTS
    residual_portfolio_v1_test
    transpose_candidates_v1_test
    transpose_cover_v1_test
    transpose_integration_v1_test
    transpose_validation_v1_test
    transpose_value_map_v1_test)
foreach(test_name IN LISTS CELLERATOR_OPERATION_FOCUSED_TESTS)
    if(test_name STREQUAL "residual_portfolio_v1_test")
        set(test_source
            src/ops/relation/cuda/sm70/residual/portfolio_v1_test.cc)
    else()
        set(test_source
            src/ops/relation/cuda/sm70/transpose/${test_name}.cc)
    endif()
    add_executable(cellerator_operation_${test_name} ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL} ${test_source})
    target_link_libraries(cellerator_operation_${test_name} PRIVATE
        Cellerator::compute_candidate_inventory)
    if(test_name STREQUAL "transpose_integration_v1_test")
        target_link_libraries(cellerator_operation_${test_name} PRIVATE
            Cellerator::builtin_candidate_catalog Cellerator::execution_planning)
    endif()
    target_compile_features(cellerator_operation_${test_name} PRIVATE cxx_std_20)
    target_compile_options(cellerator_operation_${test_name} PRIVATE -UNDEBUG)
endforeach()

add_executable(cellerator_operation_relation_apply_v1_test ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    src/ops/relation/cuda/sm70/relation_apply/tests/relation_apply_v1_test.cu)
target_link_libraries(cellerator_operation_relation_apply_v1_test PRIVATE
    Cellerator::compute_candidate_inventory)
set_target_properties(cellerator_operation_relation_apply_v1_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES)
target_compile_options(cellerator_operation_relation_apply_v1_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>
    $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>)

# Final CE-GEO source catalog fan-in. This cold target only validates and
# exposes immutable provider/relation fragments; it never owns selection,
# preparation, device queries, or execution resources.
add_library(cellerator_relation_catalog STATIC
    src/execution/planning/candidates/relation_catalog_readiness.cc)
add_library(Cellerator::relation_catalog ALIAS cellerator_relation_catalog)
target_include_directories(cellerator_relation_catalog PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(cellerator_relation_catalog PUBLIC
    Cellerator::architecture_provider
    Cellerator::relation_algebra)
target_compile_features(cellerator_relation_catalog PUBLIC cxx_std_20)

add_library(cellerator_builtin_candidate_catalog STATIC
    src/execution/planning/candidates/builtin_catalog.cc
    src/execution/planning/candidates/v3/catalog_v3.cc
    src/execution/planning/candidates/v3/discovery_v3.cc
    src/execution/planning/candidates/v3/inventory_v3.cc
    src/execution/planning/candidates/v3/validation_v3.cc
)
add_library(Cellerator::builtin_candidate_catalog ALIAS
    cellerator_builtin_candidate_catalog)
target_include_directories(cellerator_builtin_candidate_catalog PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_builtin_candidate_catalog PUBLIC
    Cellerator::row_masked_n1_candidate
    Cellerator::csr_fallback_candidate
    Cellerator::feature_major_small_n_candidate
    Cellerator::transpose_backward_candidate
)
target_compile_features(cellerator_builtin_candidate_catalog PUBLIC cxx_std_20)

target_link_libraries(cellerator_compute_candidate_inventory PUBLIC
    Cellerator::relation_algebra)

target_link_libraries(cellerator_compute_candidate_inventory PUBLIC
    Cellerator::fusion_training)
