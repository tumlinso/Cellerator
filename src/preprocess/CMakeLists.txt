add_library(cellerator_compute_preprocess STATIC
    src/preprocess/fleet.cu
    src/preprocess/fused_workflows.cu
    src/preprocess/gene_stats.cu
    src/preprocess/normalize.cu
    src/preprocess/preprocess.cu
    src/preprocess/qc_metrics.cu
    src/preprocess/workspace.cu
)
add_library(Cellerator::compute_preprocess ALIAS cellerator_compute_preprocess)
target_include_directories(cellerator_compute_preprocess PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(cellerator_compute_preprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_preprocess PUBLIC
    Cellerator::runtime_fleet
    CellShard::runtime
    Cellerator::dist
    CUDA::cudart
)
set_target_properties(cellerator_compute_preprocess PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_preprocess)

add_library(cellerator_preprocess STATIC
    src/interop/cellshard/preprocess_session.cu
    src/preprocess/runtime.cc
)
add_library(Cellerator::preprocess ALIAS cellerator_preprocess)
target_include_directories(cellerator_preprocess PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(cellerator_preprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_preprocess PUBLIC Cellerator::compute_preprocess CellShard::export)
set_target_properties(cellerator_preprocess PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_preprocess)
