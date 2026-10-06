if(NOT CELLERATOR_ENABLE_TORCH OR NOT CELLERATOR_ENABLE_CELLSHARD)
    message(FATAL_ERROR
        "examples/torch requires CELLERATOR_ENABLE_TORCH and CELLERATOR_ENABLE_CELLSHARD")
endif()

add_library(cellerator_torch_examples INTERFACE)
target_include_directories(cellerator_torch_examples INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cellerator_torch_examples INTERFACE
    Cellerator::torch
    CellShard::runtime
    Cellerator::torch_model_ops)

add_library(cellerator_torch_quantize STATIC
    src/quantize/quantize_cuda.cu)
target_include_directories(cellerator_torch_quantize PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cellerator_torch_quantize PUBLIC
    Cellerator::torch_runtime
    Cellerator::runtime_fleet
    Cellerator::model_ops)
target_link_libraries(cellerator_torch_quantize PRIVATE
    cellerator_compute_sparse_ops
    CUDA::cudart)
target_compile_features(cellerator_torch_quantize PUBLIC cxx_std_17)
set_target_properties(cellerator_torch_quantize PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_torch_quantize)

if(CELLERATOR_BUILD_TESTS)
    add_executable(cellerator_torch_dense_reduce_compile_test
        ${PROJECT_SOURCE_DIR}/tests/bindings/torch/dense_reduce_compile_test.cc)
    target_link_libraries(cellerator_torch_dense_reduce_compile_test PRIVATE
        cellerator_torch_examples)
    target_compile_features(cellerator_torch_dense_reduce_compile_test PRIVATE cxx_std_17)

    add_executable(cellerator_torch_cellshard_views_compile_test
        ${PROJECT_SOURCE_DIR}/tests/examples/torch/bindings_compile_test.cc)
    target_link_libraries(cellerator_torch_cellshard_views_compile_test PRIVATE
        Cellerator::torch_runtime
        CellShard::runtime)
    target_include_directories(cellerator_torch_cellshard_views_compile_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(cellerator_torch_cellshard_views_compile_test PRIVATE cxx_std_17)

    add_executable(cellerator_torch_quantize_primitive_test
        ${PROJECT_SOURCE_DIR}/tests/bindings/torch/quantize_primitive_test.cc)
    target_link_libraries(cellerator_torch_quantize_primitive_test PRIVATE
        cellerator_torch_quantize
        Cellerator::torch_model_ops
        CUDA::cudart)
    target_compile_features(cellerator_torch_quantize_primitive_test PRIVATE cxx_std_17)
endif()
