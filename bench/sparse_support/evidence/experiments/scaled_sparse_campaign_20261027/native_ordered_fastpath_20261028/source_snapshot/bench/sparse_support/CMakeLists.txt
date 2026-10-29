if(NOT TARGET Cellerator::sparse_support)
    message(FATAL_ERROR
        "The sparse-support benchmark requires Cellerator::sparse_support")
endif()

add_executable(cellerator_sparse_support_indexing_bench
    indexing_bench.cc)
target_link_libraries(cellerator_sparse_support_indexing_bench PRIVATE
    Cellerator::sparse_support)
target_compile_features(cellerator_sparse_support_indexing_bench PRIVATE
    cxx_std_17)

if(TARGET Cellerator::sparse_support_cuda)
    add_executable(cellerator_sparse_support_cuda_indexing_bench
        cuda_indexing_bench.cu)
    target_link_libraries(cellerator_sparse_support_cuda_indexing_bench PRIVATE
        Cellerator::sparse_support_cuda Cellerator::sparse_support CUDA::cudart)
    target_compile_options(cellerator_sparse_support_cuda_indexing_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    set_target_properties(cellerator_sparse_support_cuda_indexing_bench PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
endif()
