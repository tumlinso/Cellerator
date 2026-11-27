add_library(cellerator_compute_sparse_ops STATIC
    src/ops/matrix/sparse/kernels/base_sparse.cu
    src/ops/matrix/sparse/kernels/dist_sparse.cu
    src/ops/matrix/sparse/row_transforms.cu
    src/ops/matrix/sparse/column_moments.cu
)
add_library(Cellerator::sparse_ops ALIAS cellerator_compute_sparse_ops)
target_include_directories(cellerator_compute_sparse_ops
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(cellerator_compute_sparse_ops PUBLIC Cellerator::runtime_fleet)
target_link_libraries(cellerator_compute_sparse_ops PRIVATE Cellerator::csr_multiply CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_compute_sparse_ops PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_sparse_ops)

add_library(cellerator_compute_sparse_project STATIC
    src/ops/matrix/sparse/project/project.cu
)
target_include_directories(cellerator_compute_sparse_project PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_sparse_project PUBLIC cellerator_compute_sparse_ops Cellerator::runtime_fleet)
target_link_libraries(cellerator_compute_sparse_project PRIVATE Cellerator::csr_multiply CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_compute_sparse_project PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_sparse_project)
