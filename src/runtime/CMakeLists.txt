add_library(cellerator_runtime_multi_gpu INTERFACE)
add_library(Cellerator::runtime_multi_gpu ALIAS cellerator_runtime_multi_gpu)
add_library(cellerator_dist ALIAS cellerator_runtime_multi_gpu)
add_library(Cellerator::dist ALIAS cellerator_runtime_multi_gpu)
target_include_directories(cellerator_runtime_multi_gpu INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_runtime_multi_gpu INTERFACE cxx_std_17)
target_link_libraries(cellerator_runtime_multi_gpu INTERFACE CUDA::cudart)
if(NCCL_LIBRARY)
    target_compile_definitions(cellerator_runtime_multi_gpu INTERFACE CELLERATOR_DIST_HAS_NCCL=1)
    target_link_libraries(cellerator_runtime_multi_gpu INTERFACE ${NCCL_LIBRARY})
    if(NCCL_INCLUDE_DIR)
        target_include_directories(cellerator_runtime_multi_gpu SYSTEM INTERFACE ${NCCL_INCLUDE_DIR})
    endif()
else()
    target_compile_definitions(cellerator_runtime_multi_gpu INTERFACE CELLERATOR_DIST_HAS_NCCL=0)
endif()
add_library(cellerator_runtime STATIC
    src/runtime/runtime.cu
    src/runtime/device_descriptor.cu
    src/runtime/memory/allocation.cc
    src/runtime/memory/copy.cu
    src/runtime/memory/generation_marks.cc
    src/runtime/memory/image.cc
    src/runtime/memory/session_memory.cu
    src/runtime/value_readiness.cu
)
add_library(Cellerator::runtime ALIAS cellerator_runtime)
target_include_directories(cellerator_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_runtime PUBLIC cxx_std_17)
target_link_libraries(cellerator_runtime PUBLIC CUDA::cudart)
target_link_libraries(cellerator_runtime PRIVATE CUDA::cublas CUDA::cusparse)
set_target_properties(cellerator_runtime PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
add_library(cellerator_runtime_fleet STATIC
    src/runtime/multi_gpu/fleet.cu
    src/runtime/multi_gpu/reduce.cu
)
add_library(Cellerator::runtime_fleet ALIAS cellerator_runtime_fleet)
target_include_directories(cellerator_runtime_fleet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_runtime_fleet PUBLIC Cellerator::runtime Cellerator::runtime_multi_gpu CUDA::cublas)
target_link_libraries(cellerator_runtime_fleet PRIVATE CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_runtime_fleet PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
# Compatibility target names for pre-remap consumers. Canonical targets must
# link Cellerator::runtime or Cellerator::runtime_fleet directly.
add_library(cellerator_legacy_execution_context ALIAS cellerator_runtime_fleet)
add_library(Cellerator::legacy_execution_context ALIAS cellerator_runtime_fleet)
add_library(cellerator_compute_runtime ALIAS cellerator_runtime_fleet)
