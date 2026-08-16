add_library(cellpack STATIC
    src/candidate_relation.cc
    src/evaluator.cc
    src/gating.cc
    src/layout_metrics.cc
    src/layout_selector.cc
    src/merge_cost.cc
    src/optimizer.cc
    src/packing_plan.cc
    src/pack.cc
    src/planner.cc
)
add_library(CellPack::cellpack ALIAS cellpack)
target_include_directories(cellpack PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
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
    src/gating_cuda.cu
)
add_library(CellPack::gating_cuda ALIAS cellpack_gating_cuda)
target_include_directories(cellpack_gating_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
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
    src/merge_cost_cuda.cu
)
add_library(CellPack::merge_cost_cuda ALIAS cellpack_merge_cost_cuda)
target_include_directories(cellpack_merge_cost_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(cellpack_merge_cost_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
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

add_executable(cellPackFormatTest
    tests/format_test.cc
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
    tests/planner_test.cc
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
    tests/matrix_view_test.cc
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
    tests/reconstruction_test.cc
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

add_executable(cellPackLayoutMetricsTest
    tests/layout_metrics_test.cc
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
    tests/evaluator_test.cc
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
    tests/layout_selector_test.cc
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
    tests/gating_test.cc
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
    tests/gating_cuda_test.cu
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
    tests/optimizer_test.cc
)
target_include_directories(cellPackOptimizerTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
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
    tests/merge_cost_test.cu
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

add_executable(cellPackLayoutBench
    bench/cellpack_layout_bench.cc
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
    bench/cellpack_evaluator_bench.cc
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
    bench/cellpack_gating_bench.cu
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
    bench/cellpack_optimizer_bench.cc
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
    bench/cellpack_merge_cost_bench.cu
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
