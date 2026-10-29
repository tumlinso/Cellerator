if(TARGET Cellerator::sparse_support_numeric)
    add_executable(cellerator_sparse_support_numeric_reference numeric_reference.cc)
    target_link_libraries(cellerator_sparse_support_numeric_reference PRIVATE
        Cellerator::sparse_support_numeric)
    target_compile_features(cellerator_sparse_support_numeric_reference PRIVATE cxx_std_20)
    target_compile_options(cellerator_sparse_support_numeric_reference PRIVATE -UNDEBUG)
    add_test(NAME cellerator_sparse_support_numeric_reference
        COMMAND cellerator_sparse_support_numeric_reference)
    add_executable(cellerator_sparse_support_numeric_cuda numeric_cuda_test.cu)
    target_link_libraries(cellerator_sparse_support_numeric_cuda PRIVATE
        Cellerator::sparse_support_numeric CUDA::cudart)
    target_include_directories(cellerator_sparse_support_numeric_cuda PRIVATE
        ${PROJECT_SOURCE_DIR})
    set_target_properties(cellerator_sparse_support_numeric_cuda PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
    target_compile_options(cellerator_sparse_support_numeric_cuda PRIVATE -UNDEBUG)
    add_test(NAME cellerator_sparse_support_numeric_cuda
        COMMAND cellerator_sparse_support_numeric_cuda)
    set_tests_properties(cellerator_sparse_support_numeric_cuda PROPERTIES
        LABELS "cuda;sparse_support" SKIP_RETURN_CODE 77)
endif()

add_executable(cellerator_sparse_support_indexing_test
    indexing_support_gating_test.cc)
target_link_libraries(cellerator_sparse_support_indexing_test PRIVATE
    Cellerator::sparse_support)
target_compile_features(cellerator_sparse_support_indexing_test PRIVATE cxx_std_17)
target_compile_options(cellerator_sparse_support_indexing_test PRIVATE -UNDEBUG)
add_test(NAME cellerator_sparse_support_indexing
    COMMAND cellerator_sparse_support_indexing_test)

add_executable(cellerator_sparse_support_image_builder physical_image_builder_test.cc)
target_link_libraries(cellerator_sparse_support_image_builder PRIVATE
    Cellerator::sparse_support)
target_compile_features(cellerator_sparse_support_image_builder PRIVATE cxx_std_17)
target_compile_options(cellerator_sparse_support_image_builder PRIVATE -UNDEBUG)
add_test(NAME cellerator_sparse_support_image_builder
    COMMAND cellerator_sparse_support_image_builder)

add_executable(cellerator_sparse_support_derivative_sampling
    derivative_sampling_test.cc)
target_link_libraries(cellerator_sparse_support_derivative_sampling PRIVATE
    Cellerator::sparse_support)
target_compile_features(cellerator_sparse_support_derivative_sampling PRIVATE cxx_std_17)
target_compile_options(cellerator_sparse_support_derivative_sampling PRIVATE -UNDEBUG)
add_test(NAME cellerator_sparse_support_derivative_sampling
    COMMAND cellerator_sparse_support_derivative_sampling)

if(TARGET Cellerator::sparse_support_cuda)
    add_executable(cellerator_sparse_support_cuda_worksets cuda_worksets_test.cu)
    target_link_libraries(cellerator_sparse_support_cuda_worksets PRIVATE
        Cellerator::sparse_support_cuda CUDA::cudart)
    set_target_properties(cellerator_sparse_support_cuda_worksets PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    target_compile_options(cellerator_sparse_support_cuda_worksets PRIVATE -UNDEBUG)
    add_test(NAME cellerator_sparse_support_cuda_worksets
        COMMAND cellerator_sparse_support_cuda_worksets)
    set_tests_properties(cellerator_sparse_support_cuda_worksets PROPERTIES
        LABELS "cuda;sparse_support" SKIP_RETURN_CODE 77)
endif()
