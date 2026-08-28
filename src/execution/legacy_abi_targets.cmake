add_library(cellerator_abi STATIC
    src/abi/abi.cu
)
target_include_directories(cellerator_abi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(cellerator_abi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_abi PUBLIC cellerator_sparse_ml)
target_link_libraries(cellerator_abi PRIVATE CUDA::cudart CUDA::cusparse CUDA::cublas)
set_target_properties(cellerator_abi PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_abi)
