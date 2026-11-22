add_executable(cellerator_sequence_bench "${CMAKE_CURRENT_LIST_DIR}/sequence_bench.cc")
target_link_libraries(cellerator_sequence_bench PRIVATE Cellerator::sequence_ops)
target_compile_features(cellerator_sequence_bench PRIVATE cxx_std_20)

if(TARGET Cellerator::sequence_ops_cuda)
    add_executable(cellerator_sequence_cuda_bench "${CMAKE_CURRENT_LIST_DIR}/sequence_bench.cu")
    target_link_libraries(cellerator_sequence_cuda_bench PRIVATE Cellerator::sequence_ops_cuda CUDA::cudart)
    set_target_properties(cellerator_sequence_cuda_bench PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
endif()
