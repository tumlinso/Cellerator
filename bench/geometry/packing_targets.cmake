add_executable(cellPackApplyPlanBench
    bench/geometry/cellpack_apply_plan_bench.cu
)
target_include_directories(cellPackApplyPlanBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)
target_link_libraries(cellPackApplyPlanBench PRIVATE CellPack::apply_plan)
set_target_properties(cellPackApplyPlanBench PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellPackApplyPlanBench)
