add_library(cellpack STATIC
    candidate_relation.cc
    cell_block_records.cc
    evaluator.cc
    gating.cc
    layout_metrics.cc
    layout_selector.cc
    merge_cost.cc
    optimizer.cc
    packing_plan.cc
    pack.cc
    planner.cc
)

# CE-GEO portable semantic-geometry compiler. CSG1 construction and
# validation stay host-only and architecture-neutral; physical providers link
# this target without becoming geometry authorities.
add_library(cellerator_semantic_geometry_v1 STATIC
    compiler/admissibility.cc
    compiler/compile_geometry.cc
    compiler/cpbp_v1_compatibility.cc
    compiler/identity_strategy.cc
    compiler/relation_cover.cc
    compiler/strategy_registry.cc
    compiler/work_layout.cc
    persistence/semantic_geometry_image_v1.cc
    persistence/semantic_geometry_support_sections.cc
    strategy/exact_relation_rescan.cc
    strategy/rectangular_affinity.cc
    strategy/support_affinity.cc
    strategy/support_multiresolution.cc
    support_atlas.cc
)
add_library(Cellerator::semantic_geometry_v1 ALIAS
    cellerator_semantic_geometry_v1)
target_include_directories(cellerator_semantic_geometry_v1 PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_semantic_geometry_v1 PUBLIC
    Cellerator::biological_abi
    Cellerator::compute_sampling
)
target_compile_features(cellerator_semantic_geometry_v1 PUBLIC cxx_std_17)
add_library(CellPack::cellpack ALIAS cellpack)
target_include_directories(cellpack PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(cellpack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_compile_features(cellpack PUBLIC cxx_std_17)
set_target_properties(cellpack PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellpack)
endif()

add_library(cellpack_gating_cuda STATIC
    gating_cuda.cu
)
add_library(CellPack::gating_cuda ALIAS cellpack_gating_cuda)
target_include_directories(cellpack_gating_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellpack_gating_cuda PUBLIC CellPack::cellpack CUDA::cudart)
target_compile_features(cellpack_gating_cuda PUBLIC cxx_std_17)
set_target_properties(cellpack_gating_cuda PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellpack_gating_cuda)
endif()

add_library(cellpack_merge_cost_cuda STATIC
    merge_cost_cuda.cu
)
add_library(CellPack::merge_cost_cuda ALIAS cellpack_merge_cost_cuda)
target_include_directories(cellpack_merge_cost_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(cellpack_merge_cost_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellpack_merge_cost_cuda PUBLIC CellPack::cellpack CUDA::cudart)
target_compile_features(cellpack_merge_cost_cuda PUBLIC cxx_std_17)
set_target_properties(cellpack_merge_cost_cuda PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellpack_merge_cost_cuda)
endif()

add_library(cellpack_cell_block_records_cuda STATIC
    cell_block_records_cuda.cu
)
add_library(CellPack::cell_block_records_cuda ALIAS cellpack_cell_block_records_cuda)
target_include_directories(cellpack_cell_block_records_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellpack_cell_block_records_cuda PUBLIC CellPack::cellpack CUDA::cudart)
target_compile_features(cellpack_cell_block_records_cuda PUBLIC cxx_std_17)
set_target_properties(cellpack_cell_block_records_cuda PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellpack_cell_block_records_cuda)
endif()

# CP-BP-08 host warp-tile ABI/reference. CP-BP-07's local-order target is
# defined by the parent Cellerator build after this subdirectory is added.
add_library(cellpack_warp_tiles STATIC
    warp_tiles.cc
)
add_library(CellPack::warp_tiles ALIAS cellpack_warp_tiles)
target_include_directories(cellpack_warp_tiles PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellpack_warp_tiles PUBLIC
    CellPack::cellpack
    cellpack_local_cell_ordering
)
target_compile_features(cellpack_warp_tiles PUBLIC cxx_std_17)
set_target_properties(cellpack_warp_tiles PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellpack_warp_tiles)
endif()

# CP-BP-08 Phase D asynchronous CUDA tile construction. The Phase C host
# target remains the exact semantic oracle and owns the frozen logical ABI.
add_library(cellpack_warp_tiles_cuda STATIC
    warp_tiles_cuda.cu
)
add_library(CellPack::warp_tiles_cuda ALIAS cellpack_warp_tiles_cuda)
target_include_directories(cellpack_warp_tiles_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellpack_warp_tiles_cuda PUBLIC
    CellPack::warp_tiles
    CUDA::cudart
)
target_compile_features(cellpack_warp_tiles_cuda PUBLIC cxx_std_17)
set_target_properties(cellpack_warp_tiles_cuda PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellpack_warp_tiles_cuda)
endif()

# CP-BP-09 Phase E: asynchronous direct configured-precision warp-tile
# weighted-row reduction. The root-owned Phase D host target remains frozen.
add_library(cellpack_feature_weighted_row_reduction_cuda STATIC
    feature_weighted_row_reduction_cuda.cu
)
add_library(CellPack::feature_weighted_row_reduction_cuda ALIAS
    cellpack_feature_weighted_row_reduction_cuda)
target_include_directories(cellpack_feature_weighted_row_reduction_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellpack_feature_weighted_row_reduction_cuda PUBLIC
    cellpack_feature_weighted_row_reduction
    CUDA::cudart
)
target_compile_features(cellpack_feature_weighted_row_reduction_cuda PUBLIC cxx_std_17)
set_target_properties(cellpack_feature_weighted_row_reduction_cuda PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellpack_feature_weighted_row_reduction_cuda)
endif()

add_executable(cellPackFormatTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/format_test.cc
)
target_link_libraries(cellPackFormatTest PRIVATE CellPack::cellpack)
set_target_properties(cellPackFormatTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackFormatTest)
endif()

add_executable(cellPackPlannerTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/planner_test.cc
)
target_link_libraries(cellPackPlannerTest PRIVATE CellPack::cellpack)
set_target_properties(cellPackPlannerTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackPlannerTest)
endif()

add_executable(cellPackMatrixViewTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/matrix_view_test.cc
)
target_link_libraries(cellPackMatrixViewTest PRIVATE CellPack::cellpack)
set_target_properties(cellPackMatrixViewTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackMatrixViewTest)
endif()

add_executable(cellPackReconstructionTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/reconstruction_test.cc
)
target_link_libraries(cellPackReconstructionTest PRIVATE CellPack::cellpack)
set_target_properties(cellPackReconstructionTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackReconstructionTest)
endif()

add_executable(cellPackCellBlockRecordsTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/cell_block_records_test.cc
)
target_link_libraries(cellPackCellBlockRecordsTest PRIVATE CellPack::cellpack)
set_target_properties(cellPackCellBlockRecordsTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackCellBlockRecordsTest)
endif()

# CP-BP-08 Phase C host warp-tile ABI/reference validation.
add_executable(cellPackWarpTilesTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/warp_tiles_test.cc
)
target_link_libraries(cellPackWarpTilesTest PRIVATE CellPack::warp_tiles)
set_target_properties(cellPackWarpTilesTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackWarpTilesTest)
endif()

# CP-BP-08 Phase D exact CPU/CUDA construction validation.
add_executable(cellPackWarpTilesCudaTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/warp_tiles_cuda_test.cu
)
target_link_libraries(cellPackWarpTilesCudaTest PRIVATE CellPack::warp_tiles_cuda)
set_target_properties(cellPackWarpTilesCudaTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackWarpTilesCudaTest)
endif()

# CP-BP-09 Phase E direct-runtime contract and numerical validation.
add_executable(cellPackFeatureWeightedRowReductionCudaTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/feature_weighted_row_reduction_cuda_test.cu
)
target_link_libraries(cellPackFeatureWeightedRowReductionCudaTest PRIVATE
    CellPack::feature_weighted_row_reduction_cuda
)
set_target_properties(cellPackFeatureWeightedRowReductionCudaTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackFeatureWeightedRowReductionCudaTest)
endif()

add_executable(cellPackLayoutMetricsTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/layout_metrics_test.cc
)
target_link_libraries(cellPackLayoutMetricsTest PRIVATE CellPack::cellpack)
set_target_properties(cellPackLayoutMetricsTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackLayoutMetricsTest)
endif()

add_executable(cellPackEvaluatorTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/evaluator_test.cc
)
target_link_libraries(cellPackEvaluatorTest PRIVATE CellPack::cellpack)
set_target_properties(cellPackEvaluatorTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackEvaluatorTest)
endif()

add_executable(cellPackLayoutSelectorTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/layout_selector_test.cc
)
target_link_libraries(cellPackLayoutSelectorTest PRIVATE CellPack::cellpack)
set_target_properties(cellPackLayoutSelectorTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackLayoutSelectorTest)
endif()

add_executable(cellPackGatingTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/gating_test.cc
)
target_link_libraries(cellPackGatingTest PRIVATE CellPack::cellpack)
set_target_properties(cellPackGatingTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackGatingTest)
endif()

add_executable(cellPackGatingCudaTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/gating_cuda_test.cu
)
target_link_libraries(cellPackGatingCudaTest PRIVATE CellPack::gating_cuda)
set_target_properties(cellPackGatingCudaTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackGatingCudaTest)
endif()

add_executable(cellPackOptimizerTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/optimizer_test.cc
)
target_include_directories(cellPackOptimizerTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellPackOptimizerTest PRIVATE CellPack::cellpack)
set_target_properties(cellPackOptimizerTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackOptimizerTest)
endif()

add_executable(cellPackMergeCostTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/merge_cost_test.cu
)
target_link_libraries(cellPackMergeCostTest PRIVATE CellPack::merge_cost_cuda)
set_target_properties(cellPackMergeCostTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackMergeCostTest)
endif()

add_executable(cellPackCellBlockRecordsCudaTest
    ${PROJECT_SOURCE_DIR}/tests/geometry/cell_block_records_cuda_test.cu
)
target_link_libraries(cellPackCellBlockRecordsCudaTest PRIVATE CellPack::cell_block_records_cuda)
set_target_properties(cellPackCellBlockRecordsCudaTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackCellBlockRecordsCudaTest)
endif()

add_executable(cellPackLayoutBench
    ${PROJECT_SOURCE_DIR}/bench/geometry/cellpack_layout_bench.cc
)
target_include_directories(cellPackLayoutBench PRIVATE ${CMAKE_SOURCE_DIR}/bench)
target_link_libraries(cellPackLayoutBench PRIVATE CellPack::cellpack)
set_target_properties(cellPackLayoutBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackLayoutBench)
endif()

add_executable(cellPackEvaluatorBench
    ${PROJECT_SOURCE_DIR}/bench/geometry/cellpack_evaluator_bench.cc
)
target_include_directories(cellPackEvaluatorBench PRIVATE ${CMAKE_SOURCE_DIR}/bench)
target_link_libraries(cellPackEvaluatorBench PRIVATE CellPack::cellpack)
set_target_properties(cellPackEvaluatorBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackEvaluatorBench)
endif()

add_executable(cellPackGatingBench
    ${PROJECT_SOURCE_DIR}/bench/geometry/cellpack_gating_bench.cu
)
target_include_directories(cellPackGatingBench PRIVATE ${CMAKE_SOURCE_DIR}/bench)
target_link_libraries(cellPackGatingBench PRIVATE CellPack::gating_cuda)
set_target_properties(cellPackGatingBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackGatingBench)
endif()

add_executable(cellPackOptimizerBench
    ${PROJECT_SOURCE_DIR}/bench/geometry/cellpack_optimizer_bench.cc
)
target_include_directories(cellPackOptimizerBench PRIVATE ${CMAKE_SOURCE_DIR}/bench)
target_link_libraries(cellPackOptimizerBench PRIVATE CellPack::cellpack)
set_target_properties(cellPackOptimizerBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackOptimizerBench)
endif()

add_executable(cellPackMergeCostBench
    ${PROJECT_SOURCE_DIR}/bench/geometry/cellpack_merge_cost_bench.cu
)
target_include_directories(cellPackMergeCostBench PRIVATE ${CMAKE_SOURCE_DIR}/bench)
target_link_libraries(cellPackMergeCostBench PRIVATE CellPack::merge_cost_cuda)
set_target_properties(cellPackMergeCostBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackMergeCostBench)
endif()

add_executable(cellPackCellBlockRecordsBench
    ${PROJECT_SOURCE_DIR}/bench/geometry/cell_block_records_bench.cu
)
target_include_directories(cellPackCellBlockRecordsBench PRIVATE ${CMAKE_SOURCE_DIR}/bench)
target_link_libraries(cellPackCellBlockRecordsBench PRIVATE CellPack::cell_block_records_cuda)
set_target_properties(cellPackCellBlockRecordsBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackCellBlockRecordsBench)
endif()

# CP-BP-08 Phase D serialized V100 construction benchmark.
add_executable(cellPackWarpTilesBench
    ${PROJECT_SOURCE_DIR}/bench/geometry/warp_tiles_bench.cu
)
target_include_directories(cellPackWarpTilesBench PRIVATE ${CMAKE_SOURCE_DIR}/bench)
target_link_libraries(cellPackWarpTilesBench PRIVATE CellPack::warp_tiles_cuda)
set_target_properties(cellPackWarpTilesBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackWarpTilesBench)
endif()

# CP-BP-09 Phase E serialized V100 direct-runtime benchmark.
add_executable(cellPackFeatureWeightedRowReductionBench
    ${PROJECT_SOURCE_DIR}/bench/geometry/feature_weighted_row_reduction_bench.cu
)
target_include_directories(cellPackFeatureWeightedRowReductionBench PRIVATE
    ${CMAKE_SOURCE_DIR}/bench
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(cellPackFeatureWeightedRowReductionBench PRIVATE
    CellPack::feature_weighted_row_reduction_cuda
    cellerator_compute_sparse_ops
    Cellerator::runtime_fleet
    CUDA::cudart
    CUDA::cusparse
)
set_target_properties(cellPackFeatureWeightedRowReductionBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackFeatureWeightedRowReductionBench)
endif()

# CP-BP-12 serialized measured-cost campaign. This is a host-side calibration
# consumer of the stable CP-BP-09 runtime, not a new or tuned kernel path.
add_executable(cellPackHardwareCostAutotuneBench
    ${PROJECT_SOURCE_DIR}/bench/geometry/hardware_cost_autotune_bench.cu
)
target_include_directories(cellPackHardwareCostAutotuneBench PRIVATE
    ${CMAKE_SOURCE_DIR}/bench
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(cellPackHardwareCostAutotuneBench PRIVATE
    CellPack::hardware_cost_model
    CellPack::feature_weighted_row_reduction_cuda
    cellerator_compute_sparse_ops
    Cellerator::runtime_fleet
    CUDA::cudart
    CUDA::cusparse
)
set_target_properties(cellPackHardwareCostAutotuneBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellPackHardwareCostAutotuneBench)
endif()
