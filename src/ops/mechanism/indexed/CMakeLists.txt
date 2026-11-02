include(GNUInstallDirs)
find_package(CUDAToolkit REQUIRED)

add_library(cellerator_indexed_mechanism STATIC
    incidence.cc
    evaluators.cc
    evaluators.cu
    product_training.cu
    product_kernels.cu
    parameter_owner.cu)
add_library(Cellerator::indexed_mechanism ALIAS cellerator_indexed_mechanism)
target_include_directories(cellerator_indexed_mechanism PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(cellerator_indexed_mechanism PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    Cellerator::runtime
    Cellerator::executable_program)
set_target_properties(cellerator_indexed_mechanism PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME indexed_mechanism
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_compile_features(cellerator_indexed_mechanism PUBLIC cxx_std_20)
