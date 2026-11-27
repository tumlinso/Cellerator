# The public relation algebra remains independent of model, trainer, and
# framework ownership. Its candidate-catalog-v2 declarations are source-linked
# here, while schema-v2-only operations continue to fail closed under the
# frozen operation-core schema v1.
include(src/ops/segment/sources.cmake)
add_library(cellerator_relation_algebra STATIC
    src/execution/planning/candidates/relation_algebra_catalog.cc
    src/execution/planning/candidates/relation_algebra_assembly.cc
    src/ops/relation/relation_apply_compatibility.cc
    src/ops/condition/edge_map_or_gate.cu
    src/ops/relation/relation_bundle.cc
    src/ops/condition/edge/dynamic_support_mask_v1.cu
    src/ops/condition/edge/gate_update_validation_v1.cc
    src/ops/condition/edge/general_edge_map_v1.cu
    src/ops/condition/edge/indexed_gates_v1.cu
    src/ops/relation/sparse_axis_update/sparse_axis_update_v1.cu
    ${CELLERATOR_RELATION_SEGMENT_SOURCES}
)
add_library(Cellerator::relation_algebra ALIAS cellerator_relation_algebra)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/layout/projection/active/active_projection_v1.cu")
    target_sources(cellerator_relation_algebra PRIVATE
        src/layout/projection/active/active_projection_v1.cu)
endif()
target_include_directories(cellerator_relation_algebra PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_relation_algebra PUBLIC
    Cellerator::operation_core
    CUDA::cudart
)
set_target_properties(cellerator_relation_algebra PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)

set(CELLERATOR_OPERATION_RELATION_HOST_TEST_SOURCES
    src/ops/relation/bundle/moments_test.cc
    src/ops/relation/bundle/plan_validation_test.cc
    src/ops/relation/bundle/portfolio_validation_test.cc
    src/ops/relation/bundle/sequential_grouped_test.cc
    src/ops/relation/bundle/shared_owner_test.cc
    src/ops/relation/chain/two_hop_test.cc
    ${CELLERATOR_RELATION_SEGMENT_TEST_SOURCES})
foreach(test_source IN LISTS CELLERATOR_OPERATION_RELATION_HOST_TEST_SOURCES)
    get_filename_component(test_stem ${test_source} NAME_WE)
    get_filename_component(test_parent ${test_source} DIRECTORY)
    # Preserve the existing generated test target names after moving their sources.
    if(test_parent STREQUAL "src/ops/relation/bundle")
        set(test_parent_hash 3befcdbbcf3acfe34ad5b85515442af2)
    elseif(test_parent STREQUAL "src/ops/relation/chain")
        set(test_parent_hash 41e486b1a5d5c738cecbb0b440b18b93)
    elseif(test_parent STREQUAL "src/ops/segment")
        set(test_parent_hash dca9a81c9f12e29ab6890190570c51a8)
    else()
        string(MD5 test_parent_hash ${test_parent})
    endif()
    set(test_target cellerator_operation_relation_${test_parent_hash}_${test_stem})
    add_executable(${test_target} ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL} ${test_source})
    target_link_libraries(${test_target} PRIVATE Cellerator::relation_algebra)
    target_compile_features(${test_target} PRIVATE cxx_std_20)
    target_compile_options(${test_target} PRIVATE -UNDEBUG)
    list(APPEND CELLERATOR_OPERATION_RELATION_HOST_TEST_TARGETS ${test_target})
endforeach()

add_library(cellerator_operation_relation_sm70_compile OBJECT ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    src/ops/relation/bundle/sm70_compile_test.cu)
target_include_directories(cellerator_operation_relation_sm70_compile PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(cellerator_operation_relation_sm70_compile PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES)
