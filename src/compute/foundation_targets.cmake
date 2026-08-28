# CE-ARCH-22: backend-neutral operation preparation and direct dispatch over
# the biological execution contracts and the single Cellerator runtime.
add_library(cellerator_operation_core STATIC
    src/compute/operation/operation_core.cc
)
add_library(Cellerator::operation_core ALIAS cellerator_operation_core)
target_include_directories(cellerator_operation_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_operation_core PUBLIC cxx_std_17)
target_link_libraries(cellerator_operation_core PUBLIC
    Cellerator::biological_abi
    Cellerator::runtime
)
set_target_properties(cellerator_operation_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

add_library(cellerator_compute_matrix_convert STATIC
    src/compute/matrix/convert/bucket.cu
    src/compute/matrix/convert/compressed.cu
)
add_library(Cellerator::compute_matrix_convert ALIAS cellerator_compute_matrix_convert)
target_include_directories(cellerator_compute_matrix_convert PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_matrix_convert PUBLIC Cellerator::runtime)
target_link_libraries(cellerator_compute_matrix_convert PRIVATE CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_compute_matrix_convert PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
