# Repository-local semantic foundation; no installed SDK or runtime ownership.
add_library(cellerator_relation_semantics STATIC
    src/ops/relation/relation_semantics.cc)
add_library(Cellerator::relation_semantics ALIAS cellerator_relation_semantics)
target_include_directories(cellerator_relation_semantics PUBLIC
    "${PROJECT_SOURCE_DIR}/include")
target_compile_features(cellerator_relation_semantics PUBLIC cxx_std_20)

# The prepared pair always consumes calculus and the RU1 runtime support.
add_library(cellerator_relation_calculus STATIC src/ops/relation/relation_calculus.cc)
add_library(Cellerator::relation_calculus ALIAS cellerator_relation_calculus)
target_link_libraries(cellerator_relation_calculus PUBLIC Cellerator::relation_semantics)
target_compile_features(cellerator_relation_calculus PUBLIC cxx_std_20)
add_library(cellerator_relation_gradient_cover STATIC
    src/ops/relation/cuda/sm70/edge_value_gradient/gradient_cover.cc)
target_include_directories(cellerator_relation_gradient_cover PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(cellerator_relation_gradient_cover PUBLIC cxx_std_20)

add_executable(ceSpineCoreTest ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    tests/semantic_spine/core/descriptor_test.cc)
target_link_libraries(ceSpineCoreTest PRIVATE Cellerator::relation_semantics)
target_compile_options(ceSpineCoreTest PRIVATE -UNDEBUG)
add_executable(ceSpineMathematicalDualityTest ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    tests/semantic_spine/core/mathematical_duality_test.cc)
target_compile_features(ceSpineMathematicalDualityTest PRIVATE cxx_std_20)
target_compile_options(ceSpineMathematicalDualityTest PRIVATE -UNDEBUG)
set_target_properties(ceSpineCoreTest ceSpineMathematicalDualityTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Repository-local native execution and source-origin bridge.
# The semantic foundation remains available to the host-only configuration.
if(NOT CELLERATOR_ENABLE_CUDA STREQUAL "OFF")
    add_library(cellerator_relation_update_support STATIC
        src/execution/preparation/readiness/relation_value_readiness.cu
        src/ops/relation/cuda/sm70/edge_value_gradient/relation_gradient.cu
        src/ops/relation/cuda/sm70/edge_value_gradient/relation_value_update.cu
        src/ops/relation/cuda/sm70/edge_value_gradient/gradient_pack.cu
        src/ops/relation/cuda/sm70/edge_value_gradient/hybrid_gradient.cu
        src/ops/relation/cuda/sm70/edge_value_gradient/gradient_dispatch.cc)
    target_include_directories(cellerator_relation_update_support PRIVATE
        ${PROJECT_SOURCE_DIR}/src/runtime)
    target_link_libraries(cellerator_relation_update_support PUBLIC
        Cellerator::relation_calculus cellerator_relation_gradient_cover
        Cellerator::architecture_provider Cellerator::transpose_backward_candidate
        Cellerator::feature_major_small_n_candidate CUDA::cudart CUDA::cuda_driver)
    target_compile_features(cellerator_relation_update_support PUBLIC cxx_std_20)
    set_target_properties(cellerator_relation_update_support PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
    add_library(cellerator_prepared_relation_cuda STATIC
        src/execution/preparation/prepared_relation.cu)
    add_library(Cellerator::prepared_relation_cuda ALIAS cellerator_prepared_relation_cuda)
    add_library(cellerator_relation_update_core ALIAS cellerator_prepared_relation_cuda)
    target_link_libraries(cellerator_prepared_relation_cuda PUBLIC
        cellerator_relation_update_support
        Cellerator::feature_major_small_n_candidate
        Cellerator::transpose_backward_candidate CUDA::cudart)
    target_link_libraries(cellerator_prepared_relation_cuda PRIVATE
        Cellerator::csr_multiply Cellerator::feature_major_tiles_cuda CUDA::cusparse)
    target_compile_features(cellerator_prepared_relation_cuda PUBLIC cxx_std_20)
    set_target_properties(cellerator_prepared_relation_cuda PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)

    cellerator_include_checks(tests/semantic_spine/CMakeLists.txt)
    cellerator_include_checks(examples/semantic_spine_v1/CMakeLists.txt)
endif()
