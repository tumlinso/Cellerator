# CE-EXOP-278: exact fusion and generic training-program fan-in. Optimizer,
# loss, model, and framework policy remain caller-owned; these targets expose
# only prepared stage graphs, projection-primary values, and exact validation.
add_library(cellerator_fusion_training STATIC
    src/ops/relation/fusion/bundle_moments_v1.cu
    src/ops/relation/fusion/contract_compositions_v1.cu
    src/ops/relation/fusion/fusion_validation_v1.cc
    src/ops/relation/fusion/mma_same_owner_residual_v1.cu
    src/ops/relation/fusion/pack_apply_epilogue_v1.cu
    src/execution/preparation/prepared_stage_graph_v1.cc
    src/ops/relation/fusion/sparse_exchange_v1.cu
    src/execution/program/training/v2/generation_readiness.cc
    src/execution/program/training/v2/relation_closure.cc
    src/execution/program/training/v2/value_modes.cc)
add_library(Cellerator::fusion_training ALIAS cellerator_fusion_training)
target_include_directories(cellerator_fusion_training PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(cellerator_fusion_training PUBLIC
    Cellerator::execution_training_stages
    Cellerator::relation_algebra
    CUDA::cudart)
set_target_properties(cellerator_fusion_training PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES 70)

add_executable(cellerator_operation_fusion_training_integration_test ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    src/execution/program/training/fusion_training_integration_test.cc)
target_link_libraries(cellerator_operation_fusion_training_integration_test PRIVATE
    Cellerator::compute_candidate_inventory)
target_compile_features(cellerator_operation_fusion_training_integration_test
    PRIVATE cxx_std_20)
target_compile_options(cellerator_operation_fusion_training_integration_test
    PRIVATE -UNDEBUG)

set(CELLERATOR_OPERATION_TRAINING_TESTS
    generation_readiness
    relation_closure
    value_modes)
foreach(test_name IN LISTS CELLERATOR_OPERATION_TRAINING_TESTS)
    add_executable(cellerator_operation_training_${test_name}_test ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
        src/execution/program/training/v2/${test_name}_test.cc)
    target_link_libraries(cellerator_operation_training_${test_name}_test PRIVATE
        Cellerator::fusion_training)
    target_compile_features(cellerator_operation_training_${test_name}_test
        PRIVATE cxx_std_20)
    target_compile_options(cellerator_operation_training_${test_name}_test
        PRIVATE -UNDEBUG)
endforeach()

set(CELLERATOR_OPERATION_TRAINING_PROGRAM_TESTS
    graph_capture
    program)
foreach(test_name IN LISTS CELLERATOR_OPERATION_TRAINING_PROGRAM_TESTS)
    add_executable(cellerator_operation_training_${test_name}_test ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
        src/execution/program/training/${test_name}_test.cc)
    target_link_libraries(cellerator_operation_training_${test_name}_test PRIVATE
        Cellerator::fusion_training)
    target_compile_features(cellerator_operation_training_${test_name}_test
        PRIVATE cxx_std_20)
    target_compile_options(cellerator_operation_training_${test_name}_test
        PRIVATE -UNDEBUG)
endforeach()

add_library(cellerator_operation_fusion_sm70_compile OBJECT ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    src/ops/relation/fusion/bundle_moments_v1.cu
    src/ops/relation/fusion/contract_compositions_v1.cu
    src/ops/relation/fusion/mma_same_owner_residual_v1.cu
    src/ops/relation/fusion/pack_apply_epilogue_v1.cu
    src/ops/relation/fusion/sparse_exchange_v1.cu)
target_include_directories(cellerator_operation_fusion_sm70_compile PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(cellerator_operation_fusion_sm70_compile PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES 70)

add_library(cellerator_native_training_slice STATIC
    src/execution/program/training/native_training_slice.cu
)
add_library(Cellerator::native_training_slice ALIAS
    cellerator_native_training_slice)
target_include_directories(cellerator_native_training_slice PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_native_training_slice PUBLIC
    cellerator_transpose_backward_candidate
    Cellerator::feature_major_projection
    Cellerator::transpose_projection
    Cellerator::runtime
    CUDA::cudart
)
set_target_properties(cellerator_native_training_slice PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
