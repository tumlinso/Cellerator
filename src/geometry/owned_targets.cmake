add_subdirectory(src/geometry)

add_library(cellpack_statistical_validation STATIC
    src/geometry/statistical_validation.cc
)
add_library(CellPack::statistical_validation ALIAS cellpack_statistical_validation)
target_include_directories(cellpack_statistical_validation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_statistical_validation
    PUBLIC CellPack::cellpack
    PRIVATE Cellerator::compute_sampling
)
set_target_properties(cellpack_statistical_validation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_statistical_validation)

add_executable(cellPackStatisticalValidationTest
    tests/geometry/statistical_validation_test.cc
)
target_link_libraries(cellPackStatisticalValidationTest PRIVATE
    CellPack::statistical_validation
)
set_target_properties(cellPackStatisticalValidationTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackStatisticalValidationTest)

# CP-BP-11 Phase C: frozen-plan, record-level held-out/null validation. This
# remains in the root CMake seam so concurrent CP-BP-08 owns component CMake.
add_library(cellpack_record_statistical_validation STATIC
    src/geometry/record_statistical_validation.cc
)
add_library(CellPack::record_statistical_validation ALIAS
    cellpack_record_statistical_validation)
target_include_directories(cellpack_record_statistical_validation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_record_statistical_validation PUBLIC
    CellPack::statistical_validation
)
set_target_properties(cellpack_record_statistical_validation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_record_statistical_validation)

add_executable(cellPackRecordStatisticalValidationTest
    tests/geometry/record_statistical_validation_test.cc
)
target_link_libraries(cellPackRecordStatisticalValidationTest PRIVATE
    CellPack::record_statistical_validation
    CellPack::apply_plan
)
set_target_properties(cellPackRecordStatisticalValidationTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackRecordStatisticalValidationTest)

# CP-BP-11 Phase E: allocation-free frozen-plan tile held-out/null/bootstrap
# validation. This remains in the root CMake seam while concurrent CP-BP-09
# owns the component-CMake CUDA-consumer blocks.
add_library(cellpack_tile_statistical_validation STATIC
    src/geometry/tile_statistical_validation.cc
)
add_library(CellPack::tile_statistical_validation ALIAS
    cellpack_tile_statistical_validation)
target_include_directories(cellpack_tile_statistical_validation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_tile_statistical_validation PUBLIC
    CellPack::record_statistical_validation
    CellPack::warp_tiles
)
set_target_properties(cellpack_tile_statistical_validation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_tile_statistical_validation)

add_executable(cellPackTileStatisticalValidationTest
    tests/geometry/tile_statistical_validation_test.cc
)
target_link_libraries(cellPackTileStatisticalValidationTest PRIVATE
    CellPack::tile_statistical_validation
    CellPack::apply_plan
)
set_target_properties(cellPackTileStatisticalValidationTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackTileStatisticalValidationTest)

# CP-BP-10 Phase F: bounded host-side held-out alternating controller.
add_library(cellpack_alternating_refinement STATIC
    src/geometry/alternating_refinement.cc
)
add_library(CellPack::alternating_refinement ALIAS
    cellpack_alternating_refinement)
target_include_directories(cellpack_alternating_refinement PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_alternating_refinement PUBLIC
    CellPack::record_statistical_validation
)
set_target_properties(cellpack_alternating_refinement PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_alternating_refinement)

add_executable(cellPackAlternatingRefinementTest
    tests/geometry/alternating_refinement_test.cc
)
target_link_libraries(cellPackAlternatingRefinementTest PRIVATE
    CellPack::alternating_refinement
)
set_target_properties(cellPackAlternatingRefinementTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackAlternatingRefinementTest)

# CP-BP-11 Phase F: bootstrap-relearned mapping and measured-runtime stability.
add_library(cellpack_runtime_statistical_validation STATIC
    src/geometry/runtime_statistical_validation.cc
)
add_library(CellPack::runtime_statistical_validation ALIAS
    cellpack_runtime_statistical_validation)
target_include_directories(cellpack_runtime_statistical_validation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_runtime_statistical_validation PUBLIC
    CellPack::alternating_refinement
    CellPack::tile_statistical_validation
)
set_target_properties(cellpack_runtime_statistical_validation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_runtime_statistical_validation)

add_executable(cellPackRuntimeStatisticalValidationTest
    tests/geometry/runtime_statistical_validation_test.cc
)
target_link_libraries(cellPackRuntimeStatisticalValidationTest PRIVATE
    CellPack::runtime_statistical_validation
)
set_target_properties(cellPackRuntimeStatisticalValidationTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackRuntimeStatisticalValidationTest)

# CP-BP-07: bounded local cell ordering. Kept in the root CMake seam so the
# concurrently leased CP-BP-06 component-CMake blocks remain disjoint.
add_library(cellpack_local_cell_ordering STATIC
    src/geometry/local_cell_ordering.cc
    src/geometry/local_cell_ordering_cuda.cu
)
add_library(CellPack::local_cell_ordering ALIAS cellpack_local_cell_ordering)
target_include_directories(cellpack_local_cell_ordering PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_local_cell_ordering PUBLIC CellPack::cellpack CUDA::cudart)
set_target_properties(cellpack_local_cell_ordering PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_local_cell_ordering)

add_executable(cellPackLocalCellOrderingTest
    tests/geometry/local_cell_ordering_test.cu
)
target_link_libraries(cellPackLocalCellOrderingTest PRIVATE CellPack::local_cell_ordering)
set_target_properties(cellPackLocalCellOrderingTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackLocalCellOrderingTest)

add_executable(cellPackLocalCellOrderingBench
    bench/geometry/local_cell_ordering_bench.cu
)
target_include_directories(cellPackLocalCellOrderingBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)
target_link_libraries(cellPackLocalCellOrderingBench PRIVATE CellPack::local_cell_ordering)
set_target_properties(cellPackLocalCellOrderingBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackLocalCellOrderingBench)

add_library(cellpack_apply_plan STATIC
    src/geometry/apply_plan.cc
    src/geometry/apply_plan_cuda.cu
)
add_library(CellPack::apply_plan ALIAS cellpack_apply_plan)
target_include_directories(cellpack_apply_plan PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_apply_plan PUBLIC CellPack::cellpack CUDA::cudart)
set_target_properties(cellpack_apply_plan PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_apply_plan)

# CP-BP-09 Phase D: configured-precision canonical/record/direct-tile host
# references and the pointer-first consumer contract. CUDA runtime work remains
# separately gated by CP08_DEVICE_READY and Barrier D.
add_library(cellpack_feature_weighted_row_reduction STATIC
    src/geometry/feature_weighted_row_reduction.cc
)
add_library(CellPack::feature_weighted_row_reduction ALIAS
    cellpack_feature_weighted_row_reduction)
target_include_directories(cellpack_feature_weighted_row_reduction PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_feature_weighted_row_reduction PUBLIC
    CellPack::warp_tiles
    CellPack::apply_plan
)
set_target_properties(cellpack_feature_weighted_row_reduction PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_feature_weighted_row_reduction)

# CP-BP-13: pointer-free persistent CellPack execution image. CellShard owns
# the outer CSPACK envelope and transfer; this target owns semantic validation
# and pointer rebinding for the frozen plan/order/tile payload.
add_library(cellpack_persistent_packing_payload STATIC
    src/geometry/persistent_packing_payload.cc
)
add_library(CellPack::persistent_packing_payload ALIAS
    cellpack_persistent_packing_payload)
target_include_directories(cellpack_persistent_packing_payload PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_persistent_packing_payload PUBLIC
    CellPack::feature_weighted_row_reduction
)
set_target_properties(cellpack_persistent_packing_payload PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_persistent_packing_payload)

add_executable(cellPackPersistentPackingPayloadTest
    tests/geometry/persistent_packing_payload_test.cu
)
target_link_libraries(cellPackPersistentPackingPayloadTest PRIVATE
    CellPack::persistent_packing_payload
    CellPack::feature_weighted_row_reduction_cuda
    CellShard::inspect
    CUDA::cudart
)
set_target_properties(cellPackPersistentPackingPayloadTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackPersistentPackingPayloadTest)

# CE-ARCH-60: recovered architecture foundations are reusable targets rather
# than standalone source bursts. CellPack remains the owner of semantic
# geometry and relocatable execution images; Cellerator owns planning.
add_library(cellpack_semantic_geometry STATIC
    src/geometry/semantic_geometry.cc
)
add_library(CellPack::semantic_geometry ALIAS cellpack_semantic_geometry)
target_include_directories(cellpack_semantic_geometry PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_semantic_geometry PUBLIC
    CellPack::cellpack
    Cellerator::biological_abi
)
target_compile_features(cellpack_semantic_geometry PUBLIC cxx_std_17)

add_executable(cellPackSemanticGeometryAdapterTest
    tests/geometry/semantic_geometry_adapter_test.cc
)
target_link_libraries(cellPackSemanticGeometryAdapterTest PRIVATE
    CellPack::semantic_geometry
)
set_target_properties(cellPackSemanticGeometryAdapterTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
add_library(cellpack_execution_image_v2 STATIC
    src/geometry/persistence/execution_image_v2.cc
    src/geometry/persistence/execution_image_v2_cpk1.cc
)
add_library(CellPack::execution_image_v2 ALIAS cellpack_execution_image_v2)
target_include_directories(cellpack_execution_image_v2 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_execution_image_v2 PUBLIC
    CellPack::persistent_packing_payload
    CellPack::semantic_geometry
    Cellerator::biological_abi
)
target_compile_features(cellpack_execution_image_v2 PUBLIC cxx_std_17)

add_library(cellerator_opaque_execution_artifact STATIC
    src/execution/opaque_artifact.cc
)
add_library(Cellerator::opaque_execution_artifact ALIAS
    cellerator_opaque_execution_artifact)
target_include_directories(cellerator_opaque_execution_artifact PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_opaque_execution_artifact PUBLIC
    CellPack::execution_image_v2
)
target_compile_features(cellerator_opaque_execution_artifact PUBLIC cxx_std_17)

add_executable(cellPackExecutionImageV2Test
    tests/geometry/persistence/execution_image_v2_test.cc
)
target_link_libraries(cellPackExecutionImageV2Test PRIVATE
    CellPack::execution_image_v2
)
set_target_properties(cellPackExecutionImageV2Test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

# The device proof is intentionally a separately scheduled executable. It
# performs one opaque CellShard upload and consumes a CPE2-prebound projection
# directly on the caller stream; resource-aware CUDA gates launch it.
add_executable(cellPackExecutionImageV2DeviceTest
    tests/geometry/persistence/execution_image_v2_device_test.cu
)
target_link_libraries(cellPackExecutionImageV2DeviceTest PRIVATE
    CellPack::execution_image_v2
    CellShard::inspect
    CUDA::cudart
)
set_target_properties(cellPackExecutionImageV2DeviceTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_executable(celleratorOpaqueExecutionArtifactTest
    tests/persistence/opaque_execution_artifact_test.cu
)
target_link_libraries(celleratorOpaqueExecutionArtifactTest PRIVATE
    Cellerator::opaque_execution_artifact
    CellShard::inspect
    CUDA::cudart
)
set_target_properties(celleratorOpaqueExecutionArtifactTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
