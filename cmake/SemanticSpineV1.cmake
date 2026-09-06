# Repository-local semantic foundation; no installed SDK or runtime ownership.
add_library(cellerator_relation_semantics STATIC
    src/compute/operation/relation_semantics.cc)
add_library(Cellerator::relation_semantics ALIAS cellerator_relation_semantics)
target_include_directories(cellerator_relation_semantics PUBLIC
    "${PROJECT_SOURCE_DIR}/include")
target_compile_features(cellerator_relation_semantics PUBLIC cxx_std_17)

# The prepared pair always consumes calculus and the RU1 runtime support.
add_library(cellerator_relation_calculus STATIC src/compute/operation/relation_calculus.cc)
add_library(Cellerator::relation_calculus ALIAS cellerator_relation_calculus)
target_link_libraries(cellerator_relation_calculus PUBLIC Cellerator::relation_semantics)
target_compile_features(cellerator_relation_calculus PUBLIC cxx_std_17)
add_library(cellerator_relation_gradient_cover STATIC
    src/compute/architecture/providers/nvidia/sm70/edge_value_gradient/gradient_cover.cc)
target_include_directories(cellerator_relation_gradient_cover PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(cellerator_relation_gradient_cover PUBLIC cxx_std_17)

if(CELLERATOR_BUILD_RELATION_UPDATE_SPINE_V1)
    add_library(cellerator_relation_update_compiler STATIC
        src/compiler/ir/semantic/implement_gradient_and_publication_operations.cc
        src/compiler/sema/relation_update_spine_bridge.cc
        src/compiler/ir/realization/relation_update_spine.cc
        src/compiler/frontend/source/build_a_lossless_raw_token_stream.cc
        src/compiler/frontend/parser/parse_relation_application.cc)
    target_link_libraries(cellerator_relation_update_compiler PUBLIC Cellerator::relation_calculus)
    target_compile_features(cellerator_relation_update_compiler PUBLIC cxx_std_17)
endif()

if(CELLERATOR_BUILD_TESTS)
    add_executable(ceSpineCoreTest tests/semantic_spine/core/descriptor_test.cc)
    target_link_libraries(ceSpineCoreTest PRIVATE Cellerator::relation_semantics)
    target_compile_options(ceSpineCoreTest PRIVATE -UNDEBUG)
    add_executable(ceSpineMathematicalDualityTest
        tests/semantic_spine/core/mathematical_duality_test.cc)
    target_compile_features(ceSpineMathematicalDualityTest PRIVATE cxx_std_17)
    target_compile_options(ceSpineMathematicalDualityTest PRIVATE -UNDEBUG)
    set_target_properties(ceSpineCoreTest ceSpineMathematicalDualityTest PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

# Repository-local native execution and source-origin bridge.
# The semantic foundation remains available to the host-only configuration.
if(NOT CELLERATOR_ENABLE_CUDA STREQUAL "OFF")
    add_library(cellerator_relation_update_support STATIC
        src/runtime/relation_value_readiness.cu
        src/compute/architecture/providers/nvidia/sm70/edge_value_gradient/relation_gradient.cu
        src/compute/architecture/providers/nvidia/sm70/edge_value_gradient/relation_value_update.cu
        src/compute/architecture/providers/nvidia/sm70/edge_value_gradient/gradient_pack.cu
        src/compute/architecture/providers/nvidia/sm70/edge_value_gradient/hybrid_gradient.cu
        src/compute/architecture/providers/nvidia/sm70/edge_value_gradient/gradient_dispatch.cc)
    target_link_libraries(cellerator_relation_update_support PUBLIC
        Cellerator::relation_calculus cellerator_relation_gradient_cover
        Cellerator::architecture_provider Cellerator::transpose_backward_candidate
        Cellerator::feature_major_small_n_candidate CUDA::cudart)
    target_compile_features(cellerator_relation_update_support PUBLIC cxx_std_17)
    set_target_properties(cellerator_relation_update_support PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    add_library(cellerator_prepared_relation_cuda STATIC
        src/compute/operation/prepared_relation.cu)
    add_library(Cellerator::prepared_relation_cuda ALIAS cellerator_prepared_relation_cuda)
    add_library(cellerator_relation_update_core ALIAS cellerator_prepared_relation_cuda)
    target_link_libraries(cellerator_prepared_relation_cuda PUBLIC
        cellerator_relation_update_support
        Cellerator::feature_major_small_n_candidate
        Cellerator::transpose_backward_candidate CUDA::cudart)
    target_compile_features(cellerator_prepared_relation_cuda PUBLIC cxx_std_17)
    set_target_properties(cellerator_prepared_relation_cuda PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)

    add_library(cellerator_spine_frontend STATIC
        src/compiler/ir/semantic/implement_relation_apply_and_transpose_operations.cc
        src/compiler/ir/semantic/implement_relation_ir_types.cc
        src/compiler/ir/semantic/implement_state_and_value_plane_ir_types.cc
        src/compiler/ir/semantic/implement_domain_and_axis_ir_types.cc
        src/compiler/sema/relation_spine_bridge.cc
        src/compiler/sema/implement_numerical_tuple_semantics.cc
        src/compiler/frontend/parser/parse_compiler_semantic_declarations.cc
        src/compiler/frontend/parser/parse_biological_type_constructors_and_qualifiers.cc
        src/compiler/frontend/parser/parse_relation_application.cc
        src/compiler/frontend/parser/parse_non_relation_operation_families.cc)
    target_link_libraries(cellerator_spine_frontend PUBLIC
        Cellerator::relation_semantics CUDA::cudart)
    target_compile_features(cellerator_spine_frontend PUBLIC cxx_std_17)

    if(CELLERATOR_BUILD_TESTS)
        include(tests/semantic_spine/CMakeLists.txt)
    endif()
    include(examples/semantic_spine_v1/CMakeLists.txt)
endif()
