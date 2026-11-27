add_library(cellerator_runtime_multi_gpu INTERFACE)
add_library(Cellerator::runtime_multi_gpu ALIAS cellerator_runtime_multi_gpu)
target_include_directories(cellerator_runtime_multi_gpu INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_runtime_multi_gpu INTERFACE cxx_std_20)
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
add_subdirectory(src/memory)
include("${CMAKE_CURRENT_LIST_DIR}/sources.cmake")
add_library(cellerator_runtime STATIC)
cellerator_target_runtime_sources(cellerator_runtime)
add_library(Cellerator::runtime ALIAS cellerator_runtime)
target_include_directories(cellerator_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_runtime PUBLIC cxx_std_20)
target_link_libraries(cellerator_runtime PUBLIC
    CUDA::cudart Cellerator::memory_contract Cellerator::value_readiness)
target_link_libraries(cellerator_runtime PRIVATE CUDA::cublas CUDA::cusparse)
set_target_properties(cellerator_runtime PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
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
set_target_properties(cellerator_runtime_fleet PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
