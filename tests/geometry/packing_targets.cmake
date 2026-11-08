add_executable(cellPackFeatureWeightedRowReductionTest
    tests/geometry/feature_weighted_row_reduction_test.cc
)
target_link_libraries(cellPackFeatureWeightedRowReductionTest PRIVATE
    CellPack::feature_weighted_row_reduction
)
set_target_properties(cellPackFeatureWeightedRowReductionTest PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackFeatureWeightedRowReductionTest)

# CP-BP-12: versioned measured-hardware cost model and deterministic
# storage/runtime autotune selection. The measurement harness stays with
# its benchmark owner and the cost model with src/geometry.
add_executable(cellPackHardwareCostModelTest
    tests/geometry/hardware_cost_model_test.cc
)
target_link_libraries(cellPackHardwareCostModelTest PRIVATE
    CellPack::hardware_cost_model
)
set_target_properties(cellPackHardwareCostModelTest PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackHardwareCostModelTest)

add_executable(cellPackApplyPlanTest
    tests/geometry/apply_plan_test.cu
)
target_link_libraries(cellPackApplyPlanTest PRIVATE CellPack::apply_plan)
set_target_properties(cellPackApplyPlanTest PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
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
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackInferredPackingPipelineTest)
