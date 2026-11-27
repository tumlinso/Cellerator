add_subdirectory(src/geometry)

add_library(cellpack_statistical_validation STATIC
    src/geometry/validation/statistical_validation.cc
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
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_statistical_validation)

add_executable(cellPackStatisticalValidationTest ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    tests/geometry/statistical_validation_test.cc
)
target_link_libraries(cellPackStatisticalValidationTest PRIVATE
    CellPack::statistical_validation
)
set_target_properties(cellPackStatisticalValidationTest PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackStatisticalValidationTest)

# CP-BP-11 Phase C: frozen-plan, record-level held-out/null validation. This
# remains in the root CMake seam so concurrent CP-BP-08 owns component CMake.
add_library(cellpack_record_statistical_validation STATIC
    src/geometry/validation/record_statistical_validation.cc
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
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_record_statistical_validation)

add_executable(cellPackRecordStatisticalValidationTest ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    tests/geometry/record_statistical_validation_test.cc
)
target_link_libraries(cellPackRecordStatisticalValidationTest PRIVATE
    CellPack::record_statistical_validation
    CellPack::apply_plan
)
set_target_properties(cellPackRecordStatisticalValidationTest PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackRecordStatisticalValidationTest)

# CP-BP-11 Phase E: allocation-free frozen-plan tile held-out/null/bootstrap
# validation. This remains in the root CMake seam while concurrent CP-BP-09
# owns the component-CMake CUDA-consumer blocks.
add_library(cellpack_tile_statistical_validation STATIC
    src/geometry/validation/tile_statistical_validation.cc
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
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_tile_statistical_validation)

add_executable(cellPackTileStatisticalValidationTest ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    tests/geometry/tile_statistical_validation_test.cc
)
target_link_libraries(cellPackTileStatisticalValidationTest PRIVATE
    CellPack::tile_statistical_validation
    CellPack::apply_plan
)
set_target_properties(cellPackTileStatisticalValidationTest PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackTileStatisticalValidationTest)

# CP-BP-10 Phase F: bounded host-side held-out alternating controller.
add_library(cellpack_alternating_refinement STATIC
    src/geometry/packing/alternating_refinement.cc
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
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_alternating_refinement)

add_executable(cellPackAlternatingRefinementTest ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    tests/geometry/alternating_refinement_test.cc
)
target_link_libraries(cellPackAlternatingRefinementTest PRIVATE
    CellPack::alternating_refinement
)
set_target_properties(cellPackAlternatingRefinementTest PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackAlternatingRefinementTest)

# CP-BP-11 Phase F: bootstrap-relearned mapping and measured-runtime stability.
add_library(cellpack_runtime_statistical_validation STATIC
    src/geometry/validation/runtime_statistical_validation.cc
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
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_runtime_statistical_validation)

add_executable(cellPackRuntimeStatisticalValidationTest ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    tests/geometry/runtime_statistical_validation_test.cc
)
target_link_libraries(cellPackRuntimeStatisticalValidationTest PRIVATE
    CellPack::runtime_statistical_validation
)
set_target_properties(cellPackRuntimeStatisticalValidationTest PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackRuntimeStatisticalValidationTest)

add_executable(cellPackLocalCellOrderingTest ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    tests/geometry/local_cell_ordering_test.cu
)
target_link_libraries(cellPackLocalCellOrderingTest PRIVATE CellPack::local_cell_ordering)
set_target_properties(cellPackLocalCellOrderingTest PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackLocalCellOrderingTest)

add_executable(cellPackLocalCellOrderingBench ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    bench/geometry/local_cell_ordering_bench.cu
)
target_include_directories(cellPackLocalCellOrderingBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)
target_link_libraries(cellPackLocalCellOrderingBench PRIVATE CellPack::local_cell_ordering)
set_target_properties(cellPackLocalCellOrderingBench PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackLocalCellOrderingBench)

# CP-BP physical metadata summaries and their execution association are native
# host contracts. CellPack continues to own its packing payload and codecs.
add_library(cellerator_cellpack_geometry STATIC
    src/layout/cell_packing_geometry.cc
    src/execution/preparation/cell_packing_binding.cc
)
add_library(Cellerator::cellpack_geometry ALIAS cellerator_cellpack_geometry)
set_target_properties(cellerator_cellpack_geometry PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME cellpack_geometry)
target_include_directories(cellerator_cellpack_geometry PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_cellpack_geometry PUBLIC
    CellPack::cellpack
    Cellerator::biological_abi
)
target_compile_features(cellerator_cellpack_geometry PUBLIC cxx_std_20)

add_executable(cellPackingBindingTest ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    tests/layout/cell_packing_binding_test.cc
)
target_link_libraries(cellPackingBindingTest PRIVATE
    Cellerator::cellpack_geometry
)
set_target_properties(cellPackingBindingTest PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
if(CELLERATOR_BUILD_TESTS)
    add_test(NAME cell_packing_binding COMMAND cellPackingBindingTest)
endif()
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
    Cellerator::biological_abi
)
target_compile_features(cellpack_execution_image_v2 PUBLIC cxx_std_20)

add_executable(cellPackExecutionImageV2Test ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    tests/geometry/persistence/execution_image_v2_test.cc
)
target_link_libraries(cellPackExecutionImageV2Test PRIVATE
    CellPack::execution_image_v2
)
set_target_properties(cellPackExecutionImageV2Test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
