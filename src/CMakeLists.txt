add_executable(quantizedMatrixTest
    tests/quantized_matrix_test.cu
)
target_include_directories(quantizedMatrixTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedMatrixTest PRIVATE CUDA::cudart)
set_target_properties(quantizedMatrixTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedMatrixTest)

add_executable(cellshardAccessAdapterCompileTest
    tests/cellshard_access_adapter_compile_test.cc
)
target_include_directories(cellshardAccessAdapterCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellshardAccessAdapterCompileTest PRIVATE Cellerator::runtime CellShard::headers)
set_target_properties(cellshardAccessAdapterCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellshardAccessAdapterCompileTest)

add_executable(quantizedMatrixBench
    bench/quantized_matrix_bench.cu
)
target_include_directories(quantizedMatrixBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedMatrixBench PRIVATE CUDA::cudart)
set_target_properties(quantizedMatrixBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedMatrixBench)

add_executable(quantizedTransferSpmmBench
    bench/quantized_transfer_spmm_bench.cu
)
target_include_directories(quantizedTransferSpmmBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedTransferSpmmBench PRIVATE cellerator_compute_sparse_project cellerator_compute_sparse_ops cellerator_compute_runtime CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(quantizedTransferSpmmBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedTransferSpmmBench)

add_executable(trajectoryCompileTest
    tests/trajectory_compile_test.cu
)
target_include_directories(trajectoryCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trajectoryCompileTest PRIVATE CUDA::cudart)
set_target_properties(trajectoryCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(trajectoryCompileTest)

add_executable(trajectoryRuntimeTest
    tests/trajectory_runtime_test.cu
)
target_include_directories(trajectoryRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trajectoryRuntimeTest PRIVATE CUDA::cudart)
set_target_properties(trajectoryRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(trajectoryRuntimeTest)

add_library(cellerator_compute_exact_search STATIC
    src/compute/neighbors/exact_search/exact_search.cu
)
target_include_directories(cellerator_compute_exact_search PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_exact_search PRIVATE CUDA::cudart)
set_target_properties(cellerator_compute_exact_search PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_exact_search)

add_library(cellerator_compute_forward_neighbors STATIC
    src/compute/neighbors/forward_neighbors/forward_neighbors.cu
)
add_library(Cellerator::forward_neighbors ALIAS cellerator_compute_forward_neighbors)
target_include_directories(cellerator_compute_forward_neighbors
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(cellerator_compute_forward_neighbors
    PUBLIC
        cellerator_compute_exact_search
        CellShard::runtime
        CUDA::cudart
        CUDA::cusparse
)
set_target_properties(cellerator_compute_forward_neighbors PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_forward_neighbors)

add_library(cellerator_neighbor_math INTERFACE)
target_link_libraries(cellerator_neighbor_math INTERFACE cellerator_compute_exact_search)

add_library(cellerator_legacy_execution_context STATIC
    src/runtime/legacy/execution_context.cu
    src/runtime/legacy/reduce.cu
)
add_library(Cellerator::legacy_execution_context ALIAS cellerator_legacy_execution_context)
add_library(cellerator_compute_runtime ALIAS cellerator_legacy_execution_context)
target_include_directories(cellerator_legacy_execution_context PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_legacy_execution_context PUBLIC CellShard::runtime Cellerator::runtime Cellerator::runtime_multi_gpu CUDA::cublas)
target_link_libraries(cellerator_legacy_execution_context PRIVATE CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_legacy_execution_context PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_legacy_execution_context)

add_library(cellerator_compute_dataset STATIC
    src/compute/dataset/dataset.cc
)
add_library(Cellerator::compute_dataset ALIAS cellerator_compute_dataset)
target_include_directories(cellerator_compute_dataset PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(cellerator_compute_dataset PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_dataset PUBLIC
    cellerator_compute_runtime
    CellShard::export
    CUDA::cudart
)
set_target_properties(cellerator_compute_dataset PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_dataset)

add_library(cellerator_compute_sampling STATIC
    src/compute/dataset/sampling.cc
    src/compute/dataset/sampling_materialization.cc
)
add_library(Cellerator::compute_sampling ALIAS cellerator_compute_sampling)
target_include_directories(cellerator_compute_sampling PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_compute_sampling PUBLIC CellShard::export)
set_target_properties(cellerator_compute_sampling PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_sampling)

add_library(cellerator_compute_gene_support STATIC
    src/compute/dataset/gene_support_bitset.cu
)
add_library(Cellerator::compute_gene_support ALIAS cellerator_compute_gene_support)
target_include_directories(cellerator_compute_gene_support PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_compute_gene_support PUBLIC cellerator_compute_sampling CUDA::cudart)
set_target_properties(cellerator_compute_gene_support PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellerator_compute_gene_support)

add_library(cellerator_compute_gene_candidates STATIC
    src/geometry/candidate_discovery/gene_candidate_discovery.cc
    src/geometry/candidate_discovery/gene_candidate_minhash.cu
    src/geometry/candidate_discovery/gene_candidate_discovery_cuda.cu
)
add_library(Cellerator::compute_gene_candidates ALIAS cellerator_compute_gene_candidates)
target_include_directories(cellerator_compute_gene_candidates PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_compute_gene_candidates PUBLIC
    cellerator_compute_gene_support
    CUDA::cudart
)
set_target_properties(cellerator_compute_gene_candidates PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellerator_compute_gene_candidates)

add_library(cellerator_compute_sparse_ops STATIC
    src/compute/operators/sparse/kernels/base_sparse.cu
    src/compute/operators/sparse/kernels/dist_sparse.cu
)
target_include_directories(cellerator_compute_sparse_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_sparse_ops PUBLIC cellerator_compute_runtime)
target_link_libraries(cellerator_compute_sparse_ops PRIVATE CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_compute_sparse_ops PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_sparse_ops)

add_library(cellerator_compute_sparse_project STATIC
    src/compute/candidate/sparse/project.cu
)
target_include_directories(cellerator_compute_sparse_project PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_sparse_project PUBLIC cellerator_compute_sparse_ops cellerator_compute_runtime)
target_link_libraries(cellerator_compute_sparse_project PRIVATE CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_compute_sparse_project PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_sparse_project)

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
    cellerator_compute_runtime
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

if(CELLERATOR_BUILD_PREPROCESS_WORKBENCH AND PkgConfig_FOUND)
    pkg_check_modules(CELLERATOR_NCURSESW QUIET ncursesw)
    if(CELLERATOR_NCURSESW_FOUND)
        add_executable(celleratorPreprocessWorkbench
            tools/preprocess_workbench/workbench_main.cc
        )
        target_include_directories(celleratorPreprocessWorkbench PRIVATE
            ${CELLERATOR_NCURSESW_INCLUDE_DIRS}
        )
        target_compile_options(celleratorPreprocessWorkbench PRIVATE
            ${CELLERATOR_NCURSESW_CFLAGS_OTHER}
        )
        target_link_directories(celleratorPreprocessWorkbench PRIVATE
            ${CELLERATOR_NCURSESW_LIBRARY_DIRS}
        )
        target_link_libraries(celleratorPreprocessWorkbench PRIVATE
            Cellerator::preprocess
            ${CELLERATOR_NCURSESW_LIBRARIES}
        )
        set_target_properties(celleratorPreprocessWorkbench PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED YES
        )
    endif()
endif()

if(CELLERATOR_ENABLE_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE CELLERATOR_PYBIND11_CMAKE_DIR
            RESULT_VARIABLE CELLERATOR_PYBIND11_CMAKE_DIR_STATUS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(CELLERATOR_PYBIND11_CMAKE_DIR_STATUS EQUAL 0 AND NOT "${CELLERATOR_PYBIND11_CMAKE_DIR}" STREQUAL "")
            list(PREPEND CMAKE_PREFIX_PATH "${CELLERATOR_PYBIND11_CMAKE_DIR}")
            find_package(pybind11 CONFIG QUIET)
        endif()
    endif()
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.6
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    if(NOT TARGET CellShard::export)
        message(FATAL_ERROR "CELLERATOR_ENABLE_PYTHON requires CellShard::export")
    endif()

    pybind11_add_module(_cellerator
        python/module.cc
    )
    target_include_directories(_cellerator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(_cellerator PRIVATE
        Cellerator::preprocess
        CellShard::export
        CUDA::cudart
    )
    set_target_properties(_cellerator PROPERTIES
        OUTPUT_NAME _cellerator
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )
    cellerator_enable_perf(_cellerator)

    install(TARGETS _cellerator
        LIBRARY DESTINATION cellerator
        RUNTIME DESTINATION cellerator
    )
    install(FILES
        python/cellerator/__init__.py
        python/cellerator/pp.py
        DESTINATION cellerator
    )
endif()

add_library(cellerator_sparse_math INTERFACE)
target_link_libraries(cellerator_sparse_math INTERFACE cellerator_compute_sparse_project cellerator_compute_sparse_ops cellerator_compute_runtime)
add_library(cellerator_sparse_linalg INTERFACE)
target_link_libraries(cellerator_sparse_linalg INTERFACE cellerator_compute_sparse_project cellerator_compute_sparse_ops cellerator_compute_runtime)
add_library(cellerator_sparse_ml INTERFACE)
target_link_libraries(cellerator_sparse_ml INTERFACE cellerator_compute_gene_candidates cellerator_compute_gene_support cellerator_compute_sampling cellerator_compute_dataset cellerator_compute_sparse_project cellerator_compute_sparse_ops cellerator_compute_runtime cellerator_compute_preprocess)

add_library(cellerator_abi STATIC
    src/abi/abi.cu
)
target_include_directories(cellerator_abi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(cellerator_abi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_abi PUBLIC cellerator_sparse_ml)
target_link_libraries(cellerator_abi PRIVATE CUDA::cudart CUDA::cusparse CUDA::cublas)
set_target_properties(cellerator_abi PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_abi)

add_executable(sparseOpsBench
    bench/sparse_ops_bench.cu
)
target_include_directories(sparseOpsBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sparseOpsBench PRIVATE cellerator_compute_sparse_project cellerator_compute_sparse_ops cellerator_compute_runtime CUDA::cudart CUDA::cusparse)
set_target_properties(sparseOpsBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(sparseOpsBench)

if(CELLERATOR_BUILD_PREPROCESS_BENCHMARKS)
    add_executable(celleratorPreprocessScrnaBench
        bench/preprocess/scrna_preprocess_bench.cu
    )
    target_include_directories(celleratorPreprocessScrnaBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(celleratorPreprocessScrnaBench PRIVATE Cellerator::preprocess CUDA::cudart)
    set_target_properties(celleratorPreprocessScrnaBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    cellerator_enable_perf(celleratorPreprocessScrnaBench)

    add_executable(celleratorPreprocessNcclReduceBench
        bench/preprocess/nccl_gene_reduce_bench.cu
    )
    target_include_directories(celleratorPreprocessNcclReduceBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(celleratorPreprocessNcclReduceBench PRIVATE Cellerator::preprocess CUDA::cudart)
    set_target_properties(celleratorPreprocessNcclReduceBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    cellerator_enable_perf(celleratorPreprocessNcclReduceBench)

endif()

add_executable(exactSearchRuntimeTest
    tests/exact_search_runtime_test.cu
)
target_include_directories(exactSearchRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(exactSearchRuntimeTest PRIVATE cellerator_compute_exact_search CUDA::cudart)
set_target_properties(exactSearchRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(exactSearchRuntimeTest)

add_executable(sparseOpsRuntimeTest
    tests/sparse_ops_runtime_test.cu
)
target_include_directories(sparseOpsRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sparseOpsRuntimeTest PRIVATE cellerator_compute_sparse_project cellerator_compute_sparse_ops cellerator_compute_runtime CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(sparseOpsRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(sparseOpsRuntimeTest)

add_executable(coreSparseLayoutRuntimeTest
    tests/core_sparse_layout_runtime_test.cu
)
target_include_directories(coreSparseLayoutRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(coreSparseLayoutRuntimeTest PRIVATE cellerator_compute_matrix_convert Baseplane::seq CUDA::cudart CUDA::cusparse)
set_target_properties(coreSparseLayoutRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(coreSparseLayoutRuntimeTest)

add_executable(datasetRuntimeTest
    tests/dataset_runtime_test.cc
)
target_include_directories(datasetRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(datasetRuntimeTest PRIVATE cellerator_compute_dataset CUDA::cudart)
set_target_properties(datasetRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(datasetRuntimeTest)

add_executable(samplingRuntimeTest
    tests/sampling_runtime_test.cc
)
target_include_directories(samplingRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(samplingRuntimeTest PRIVATE cellerator_compute_sampling cellerator_compute_dataset CUDA::cudart)
set_target_properties(samplingRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(samplingRuntimeTest)

add_executable(samplingMaterializationRuntimeTest
    tests/sampling_materialization_runtime_test.cc
)
target_include_directories(samplingMaterializationRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(samplingMaterializationRuntimeTest PRIVATE cellerator_compute_sampling CUDA::cudart)
set_target_properties(samplingMaterializationRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(samplingMaterializationRuntimeTest)

add_executable(geneSupportBitsetRuntimeTest
    tests/gene_support_bitset_runtime_test.cu
)
target_include_directories(geneSupportBitsetRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(geneSupportBitsetRuntimeTest PRIVATE cellerator_compute_gene_support CUDA::cudart)
set_target_properties(geneSupportBitsetRuntimeTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(geneSupportBitsetRuntimeTest)

add_executable(geneCandidateDiscoveryRuntimeTest
    tests/gene_candidate_discovery_runtime_test.cu
)
target_include_directories(geneCandidateDiscoveryRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(geneCandidateDiscoveryRuntimeTest PRIVATE
    cellerator_compute_gene_candidates
    CUDA::cudart
)
set_target_properties(geneCandidateDiscoveryRuntimeTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(geneCandidateDiscoveryRuntimeTest)

add_executable(geneCandidateDiscoveryBench
    bench/gene_candidate_discovery_bench.cu
)
target_include_directories(geneCandidateDiscoveryBench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/bench
)
target_link_libraries(geneCandidateDiscoveryBench PRIVATE
    cellerator_compute_gene_candidates
    CUDA::cudart
)
set_target_properties(geneCandidateDiscoveryBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(geneCandidateDiscoveryBench)

add_executable(celleratorPreprocessRuntimeTest
    tests/preprocess/preprocess_runtime_test.cu
)
target_include_directories(celleratorPreprocessRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorPreprocessRuntimeTest PRIVATE Cellerator::preprocess CUDA::cudart)
set_target_properties(celleratorPreprocessRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorPreprocessRuntimeTest)

add_executable(celleratorPreprocessQCMetricsEquivalenceTest
    tests/preprocess/qc_metrics_equivalence_test.cu
)
target_include_directories(celleratorPreprocessQCMetricsEquivalenceTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorPreprocessQCMetricsEquivalenceTest PRIVATE Cellerator::preprocess CUDA::cudart)
set_target_properties(celleratorPreprocessQCMetricsEquivalenceTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorPreprocessQCMetricsEquivalenceTest)

add_executable(celleratorPreprocessQCMaskGroupsTest
    tests/preprocess/qc_mask_groups_test.cu
)
target_include_directories(celleratorPreprocessQCMaskGroupsTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorPreprocessQCMaskGroupsTest PRIVATE Cellerator::preprocess CUDA::cudart)
set_target_properties(celleratorPreprocessQCMaskGroupsTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorPreprocessQCMaskGroupsTest)

add_executable(celleratorPreprocessQCFleetTest
    tests/preprocess/qc_fleet_test.cu
)
target_include_directories(celleratorPreprocessQCFleetTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorPreprocessQCFleetTest PRIVATE Cellerator::preprocess CUDA::cudart)
set_target_properties(celleratorPreprocessQCFleetTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorPreprocessQCFleetTest)

add_executable(celleratorPreprocessCellShardSessionApiTest
    tests/preprocess/cellshard_session_api_test.cu
)
target_include_directories(celleratorPreprocessCellShardSessionApiTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorPreprocessCellShardSessionApiTest PRIVATE Cellerator::preprocess CUDA::cudart)
set_target_properties(celleratorPreprocessCellShardSessionApiTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorPreprocessCellShardSessionApiTest)

add_executable(celleratorPreprocessAdapterStagingTest
    tests/preprocess/adapter_staging_test.cc
)
target_include_directories(celleratorPreprocessAdapterStagingTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorPreprocessAdapterStagingTest PRIVATE Cellerator::preprocess)
set_target_properties(celleratorPreprocessAdapterStagingTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorPreprocessAdapterStagingTest)

add_executable(celleratorPreprocessDoublePreprocessRejectionTest
    tests/preprocess/double_preprocess_rejection_test.cc
)
target_include_directories(celleratorPreprocessDoublePreprocessRejectionTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorPreprocessDoublePreprocessRejectionTest PRIVATE Cellerator::preprocess)
set_target_properties(celleratorPreprocessDoublePreprocessRejectionTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorPreprocessDoublePreprocessRejectionTest)

add_executable(abiRuntimeTest
    tests/abi_runtime_test.cu
)
target_include_directories(abiRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(abiRuntimeTest PRIVATE cellerator_abi CUDA::cudart)
set_target_properties(abiRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(abiRuntimeTest)

add_executable(celleratorBiologicalAbiHostTest
    tests/biological_abi/biological_abi_host_test.cc
)
target_link_libraries(celleratorBiologicalAbiHostTest PRIVATE
    Cellerator::biological_abi
    Baseplane::seq
)
set_target_properties(celleratorBiologicalAbiHostTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

add_executable(celleratorBiologicalAbiCudaCompileTest
    tests/biological_abi/biological_abi_cuda_compile_test.cu
)
target_link_libraries(celleratorBiologicalAbiCudaCompileTest PRIVATE
    Cellerator::biological_abi
    CUDA::cudart
)
set_target_properties(celleratorBiologicalAbiCudaCompileTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

# CE-ARCH-40: Cellerator-owned preparation and execution over frozen Baseplane
# predicate contracts. Baseplane remains independently buildable and owns no
# Cellerator runtime or numerical policy.
add_library(cellerator_sequence_integration STATIC
    src/interop/baseplane/baseplane_strategy.cc
    src/interop/baseplane/baseplane_integration.cu
)
add_library(Cellerator::sequence_integration ALIAS
    cellerator_sequence_integration)
target_include_directories(cellerator_sequence_integration PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_sequence_integration PUBLIC
    Cellerator::operation_core
    Baseplane::seq
    CUDA::cudart
)
set_target_properties(cellerator_sequence_integration PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_executable(celleratorBaseplaneSequenceIntegrationTest
    tests/interop/baseplane/baseplane_integration_test.cu
)
target_link_libraries(celleratorBaseplaneSequenceIntegrationTest PRIVATE
    Cellerator::sequence_integration
)
set_target_properties(celleratorBaseplaneSequenceIntegrationTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(celleratorBaseplaneSequenceIntegrationTest)

add_library(cellerator_state_reduce_cuda STATIC
    examples/models/state_reduce.cu
)
target_include_directories(cellerator_state_reduce_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_state_reduce_cuda PRIVATE cellerator_compute_sparse_project cellerator_compute_sparse_ops cellerator_compute_runtime CellShard::runtime CUDA::cudart CUDA::cusparse CUDA::cublas)
set_target_properties(cellerator_state_reduce_cuda PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_state_reduce_cuda)

add_executable(stateReduceRuntimeTest
    tests/state_reduce_runtime_test.cu
)
target_include_directories(stateReduceRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stateReduceRuntimeTest PRIVATE cellerator_state_reduce_cuda cellerator_compute_sparse_project cellerator_compute_sparse_ops cellerator_compute_runtime CellShard::runtime CUDA::cudart CUDA::cusparse CUDA::cublas)
set_target_properties(stateReduceRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(stateReduceRuntimeTest)

add_library(cellerator_developmental_time_cuda STATIC
    examples/models/developmental_time_model.cu
)
target_include_directories(cellerator_developmental_time_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_developmental_time_cuda PRIVATE cellerator_compute_sparse_project cellerator_compute_sparse_ops cellerator_compute_runtime CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_developmental_time_cuda PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_developmental_time_cuda)

add_library(cellerator_developmental_time_trajectory_cuda STATIC
    examples/models/developmental_time_trajectory_model.cu
)
target_include_directories(cellerator_developmental_time_trajectory_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_developmental_time_trajectory_cuda PRIVATE cellerator_developmental_time_cuda cellerator_compute_sparse_project cellerator_compute_sparse_ops cellerator_compute_runtime CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_developmental_time_trajectory_cuda PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_developmental_time_trajectory_cuda)

add_executable(developmentalTimeCompileTest
    tests/developmental_time_compile_test.cc
)
target_include_directories(developmentalTimeCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(developmentalTimeCompileTest PRIVATE CellShard::runtime CUDA::cudart)
set_target_properties(developmentalTimeCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(developmentalTimeCompileTest)

add_executable(developmentalTimeRuntimeTest
    tests/developmental_time_runtime_test.cu
)
target_include_directories(developmentalTimeRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(developmentalTimeRuntimeTest PRIVATE cellerator_developmental_time_cuda cellerator_compute_sparse_project cellerator_compute_sparse_ops cellerator_compute_runtime CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(developmentalTimeRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(developmentalTimeRuntimeTest)

add_executable(developmentalTimeTrajectoryRuntimeTest
    tests/developmental_time_trajectory_runtime_test.cu
)
target_include_directories(developmentalTimeTrajectoryRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(developmentalTimeTrajectoryRuntimeTest PRIVATE cellerator_developmental_time_trajectory_cuda cellerator_developmental_time_cuda cellerator_compute_sparse_project cellerator_compute_sparse_ops cellerator_compute_runtime CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(developmentalTimeTrajectoryRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(developmentalTimeTrajectoryRuntimeTest)
