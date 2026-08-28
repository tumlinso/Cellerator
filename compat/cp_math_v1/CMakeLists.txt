# The surviving CP-Math v1 pieces are validation and compatibility evidence,
# not a runtime or planner. They remain explicit, EXCLUDE_FROM_ALL targets so
# old referees and projection adapters stay reproducible without entering the
# supported operation/session graph.
add_library(cellerator_math_v1_evidence STATIC EXCLUDE_FROM_ALL
    compat/cp_math_v1/src/operation.cc
    compat/cp_math_v1/src/operation_signature.cc
    src/compute/projection/physical_csr.cc
    compat/cp_math_v1/src/physical_bell.cc
    compat/cp_math_v1/src/physical_bell_candidates.cc
    src/compute/projection/native_tile_view.cc
    compat/cp_math_v1/src/referee.cc
)
target_include_directories(cellerator_math_v1_evidence PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/compat/cp_math_v1/include>
)
target_link_libraries(cellerator_math_v1_evidence PUBLIC
    CellPack::persistent_packing_payload
    CUDA::cudart
)
target_compile_features(cellerator_math_v1_evidence PUBLIC cxx_std_17)

add_library(cellerator_packed_dense_operand STATIC
    compat/cp_math_v1/src/packed_dense_operand.cu
)
add_library(Cellerator::packed_dense_operand ALIAS
    cellerator_packed_dense_operand)
target_include_directories(cellerator_packed_dense_operand PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_packed_dense_operand PUBLIC
    cellerator_math_v1_evidence
    CUDA::cudart
)
set_target_properties(cellerator_packed_dense_operand PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_executable(cpMathOperationContractTest EXCLUDE_FROM_ALL
    tests/math_operation_contract_test.cc
)
target_link_libraries(cpMathOperationContractTest PRIVATE
    cellerator_math_v1_evidence
)

add_executable(cpMathRefereeFoundationTest EXCLUDE_FROM_ALL
    tests/math_referee_foundation_test.cc
)
target_include_directories(cpMathRefereeFoundationTest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(cpMathRefereeFoundationTest PRIVATE
    cellerator_math_v1_evidence
)

add_executable(cpMathNativeTileAdapterTest EXCLUDE_FROM_ALL
    tests/math_native_tile_adapter_test.cc
)
target_link_libraries(cpMathNativeTileAdapterTest PRIVATE
    cellerator_math_v1_evidence
)

add_executable(cpMathBellLoweringTest EXCLUDE_FROM_ALL
    tests/math_bell_lowering_test.cc
)
target_link_libraries(cpMathBellLoweringTest PRIVATE
    cellerator_math_v1_evidence
)

add_executable(cpMathExecutionCsrTest EXCLUDE_FROM_ALL
    tests/math_execution_csr_test.cu
)
target_link_libraries(cpMathExecutionCsrTest PRIVATE
    Cellerator::packed_dense_operand
    CUDA::cudart
)
set_target_properties(cpMathExecutionCsrTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_executable(cellPackFeatureWeightedRowReductionTest
    tests/geometry/feature_weighted_row_reduction_test.cc
)
target_link_libraries(cellPackFeatureWeightedRowReductionTest PRIVATE
    CellPack::feature_weighted_row_reduction
)
set_target_properties(cellPackFeatureWeightedRowReductionTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackFeatureWeightedRowReductionTest)

# CP-BP-12: versioned measured-hardware cost model and deterministic
# storage/runtime autotune selection. The measurement harness is wired in the
# CellPack component CMake file beside the CUDA consumer it compares.
add_library(cellpack_hardware_cost_model STATIC
    src/geometry/hardware_cost_model.cc
)
add_library(CellPack::hardware_cost_model ALIAS cellpack_hardware_cost_model)
target_include_directories(cellpack_hardware_cost_model PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_hardware_cost_model PUBLIC CellPack::cellpack)
set_target_properties(cellpack_hardware_cost_model PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_hardware_cost_model)

add_executable(cellPackHardwareCostModelTest
    tests/geometry/hardware_cost_model_test.cc
)
target_link_libraries(cellPackHardwareCostModelTest PRIVATE
    CellPack::hardware_cost_model
)
set_target_properties(cellPackHardwareCostModelTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackHardwareCostModelTest)

add_executable(cellPackApplyPlanTest
    tests/geometry/apply_plan_test.cu
)
target_link_libraries(cellPackApplyPlanTest PRIVATE CellPack::apply_plan)
set_target_properties(cellPackApplyPlanTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackApplyPlanTest)

add_executable(cellPackInferredPackingPipelineTest
    tests/geometry/inferred_packing_pipeline_test.cc
)
target_link_libraries(cellPackInferredPackingPipelineTest PRIVATE
    Cellerator::compute_gene_candidates
    CellPack::merge_cost_cuda
    CellPack::apply_plan
)
set_target_properties(cellPackInferredPackingPipelineTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackInferredPackingPipelineTest)

add_executable(cellPackApplyPlanBench
    bench/geometry/cellpack_apply_plan_bench.cu
)
target_include_directories(cellPackApplyPlanBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)
target_link_libraries(cellPackApplyPlanBench PRIVATE CellPack::apply_plan)
set_target_properties(cellPackApplyPlanBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackApplyPlanBench)

# CP-MATH-03 benchmark promoted from its standalone validation command so
# background evidence can rebuild it from an immutable source snapshot.
add_executable(cpMathPackedDenseOperandBench
    bench/math/packed_dense_operand_bench.cu
)
target_include_directories(cpMathPackedDenseOperandBench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/bench
)
target_link_libraries(cpMathPackedDenseOperandBench PRIVATE
    Cellerator::packed_dense_operand
)
set_target_properties(cpMathPackedDenseOperandBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cpMathPackedDenseOperandBench)

if(CELLERATOR_ENABLE_TORCH_MODELS)
    add_subdirectory(components/CelleraTorch)
endif()
