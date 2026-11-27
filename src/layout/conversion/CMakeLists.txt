add_library(cellerator_compute_matrix_convert STATIC
    src/layout/conversion/bucket.cu
    src/layout/conversion/compressed.cu
)
add_library(Cellerator::compute_matrix_convert ALIAS cellerator_compute_matrix_convert)
target_include_directories(cellerator_compute_matrix_convert PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_matrix_convert PUBLIC Cellerator::runtime)
target_link_libraries(cellerator_compute_matrix_convert PRIVATE CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_compute_matrix_convert PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
