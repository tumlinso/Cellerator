if(NOT CMAKE_CUDA_COMPILER_LOADED)
    message(FATAL_ERROR
        "Native-data benchmark builds require an enabled CUDA compiler")
endif()

# Torch's CUDA component discovery should use the already selected CUDA
# toolkit, not whichever toolkit happens to be on PATH.
find_package(CUDAToolkit REQUIRED)
include("${PROJECT_SOURCE_DIR}/cmake/CelleratorTorchDependency.cmake")

# This is the conventional eager ATen comparison side. The Cellerator runner
# is a separate target because it links the native CUDA owner and uses the same
# workload/result schema after its adapter lands.
add_executable(cellerator_native_data_aten_bench
    benchmark_main.cc
    workload.cc
    reference_workload.cc
    aten_baseline.cc)
target_include_directories(cellerator_native_data_aten_bench PRIVATE
    "${PROJECT_SOURCE_DIR}/bench/native_data"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/tests/native_data/reference")
target_compile_features(cellerator_native_data_aten_bench PRIVATE cxx_std_20)
target_link_libraries(cellerator_native_data_aten_bench PRIVATE
    ${CELLERATOR_TORCH_LINK_TARGETS}
    CUDA::cudart)
if(CELLERATOR_TORCH_CXX_OPTIONS)
    target_compile_options(cellerator_native_data_aten_bench PRIVATE
        ${CELLERATOR_TORCH_CXX_OPTIONS})
endif()
set_target_properties(cellerator_native_data_aten_bench PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)

add_executable(cellerator_native_data_specialist_bench
    benchmark_main.cc
    workload.cc
    reference_workload.cc
    distributed_adapter.cc)
target_include_directories(cellerator_native_data_specialist_bench PRIVATE
    "${PROJECT_SOURCE_DIR}/bench/native_data"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/tests/native_data/reference")
target_compile_features(cellerator_native_data_specialist_bench PRIVATE cxx_std_20)
target_compile_definitions(cellerator_native_data_specialist_bench PRIVATE
    CELLERATOR_NATIVE_DATA_SPECIALIST=1)
target_link_libraries(cellerator_native_data_specialist_bench PRIVATE
    Cellerator::native_data_cuda
    ${CELLERATOR_TORCH_LINK_TARGETS}
    CUDA::cudart)
if(CELLERATOR_TORCH_CXX_OPTIONS)
    target_compile_options(cellerator_native_data_specialist_bench PRIVATE
        ${CELLERATOR_TORCH_CXX_OPTIONS})
endif()
set_target_properties(cellerator_native_data_specialist_bench PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
