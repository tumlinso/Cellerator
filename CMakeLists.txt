cmake_minimum_required(VERSION 3.20)

# Choose the CUDA toolchain before enabling the CUDA language.
#
# On this host CMake can resolve `/usr/bin/nvcc` from the distro CUDA 12.0
# package even when a newer HPC SDK nvcc is earlier on PATH. That older nvcc
# also trips over the default GCC 13 host compiler on Ubuntu 24.04, producing
# the recurring `_Float32` / `_Float64` failures during compiler detection.
#
# Respect explicit user/tooling choices first:
# - `-DCMAKE_CUDA_COMPILER=...`
# - `-DCMAKE_CUDA_HOST_COMPILER=...`
# - `CUDACXX`
# - `CUDAHOSTCXX`
#
# Otherwise prefer the known-good local toolchain pair.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(DEFINED ENV{CUDACXX} AND NOT "$ENV{CUDACXX}" STREQUAL "")
        set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "CUDA compiler" FORCE)
    elseif(EXISTS "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc")
        set(CMAKE_CUDA_COMPILER
            "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc"
            CACHE FILEPATH "CUDA compiler" FORCE)
        message(STATUS "Defaulting CUDA compiler to the local HPC SDK nvcc")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    if(DEFINED ENV{CUDAHOSTCXX} AND NOT "$ENV{CUDAHOSTCXX}" STREQUAL "")
        set(CMAKE_CUDA_HOST_COMPILER "$ENV{CUDAHOSTCXX}" CACHE FILEPATH "CUDA host compiler" FORCE)
    elseif(EXISTS "/usr/bin/g++-12")
        set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12" CACHE FILEPATH "CUDA host compiler" FORCE)
        message(STATUS "Defaulting CUDA host compiler to /usr/bin/g++-12 for nvcc compatibility")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

project(Cellerator LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(PkgConfig REQUIRED)

option(CELLERATOR_ENABLE_TORCH_MODELS "Enable libtorch-backed model targets" ON)
option(CELLERATOR_ENABLE_CUDA_LINEINFO "Emit CUDA line mapping for Nsight profiling" OFF)
option(CELLERATOR_ENABLE_NCURSES_WORKBENCH "Enable the ncurses ingest/inspect workbench" ON)
option(CELLERATOR_ENABLE_PYTHON "Enable the optional Cellerator Python wrapper." OFF)
option(CELLERATOR_ENABLE_CELLSHARD_PYTHON "Enable the optional CellShard pybind/H5AD surface." OFF)
set(CELLERATOR_CUDA_MODE "generic" CACHE STRING
    "CUDA optimization mode: generic, native, or native-extreme")
set_property(CACHE CELLERATOR_CUDA_MODE PROPERTY STRINGS generic native native-extreme)
set(CELLERATOR_PREFERRED_LIBTORCH_ROOT "/usr/local" CACHE PATH
    "Preferred root for the source-built libtorch installation")
set(CUDABIOTYPES_BUILD_TESTS OFF CACHE BOOL "Disable nested cudaBioTypes tests in the root build" FORCE)
if(DEFINED CELLSHARD_ENABLE_CUDA AND NOT CELLSHARD_ENABLE_CUDA)
    message(FATAL_ERROR
        "The integrated Cellerator build is accelerator-only. "
        "Do not configure the root build with CELLSHARD_ENABLE_CUDA=OFF.")
endif()
set(CELLSHARD_ENABLE_CUDA ON CACHE BOOL
    "Cellerator hard-wires the integrated CellShard dependency to the CUDA runtime path." FORCE)
set(CELLSHARD_BUILD_EXPORT ON CACHE BOOL "Build CellShard export helpers in the root build." FORCE)
set(CELLSHARD_ENABLE_PYTHON ${CELLERATOR_ENABLE_CELLSHARD_PYTHON} CACHE BOOL
    "Build the optional CellShard Python bindings and H5AD export app." FORCE)
set(CELLSHARD_BUILD_APPS ${CELLERATOR_ENABLE_CELLSHARD_PYTHON} CACHE BOOL
    "Build optional CellShard applications from the root build." FORCE)
set(CELLSHARD_BUILD_TESTS OFF CACHE BOOL "Disable nested CellShard tests in the root build." FORCE)
set(CELLSHARD_ENABLE_CUDA_LINEINFO ${CELLERATOR_ENABLE_CUDA_LINEINFO} CACHE BOOL
    "Emit CUDA line mapping for Nsight profiling inside CellShard." FORCE)
set(CELLSHARD_CUDA_MODE ${CELLERATOR_CUDA_MODE} CACHE STRING
    "CellShard CUDA optimization mode: generic, native, or native-extreme." FORCE)

add_subdirectory(extern/cudaBioTypes EXCLUDE_FROM_ALL)
add_subdirectory(extern/CellShard EXCLUDE_FROM_ALL)

if(CELLERATOR_ENABLE_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    if(NOT TARGET pybind11::module)
        find_package(pybind11 CONFIG QUIET)
    endif()
    if(NOT TARGET pybind11::module)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
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
    if(NOT TARGET pybind11::module)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.6
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

if(CELLERATOR_ENABLE_TORCH_MODELS)
    if(DEFINED ENV{LIBTORCH_PATH} AND NOT "$ENV{LIBTORCH_PATH}" STREQUAL "")
        list(PREPEND CMAKE_PREFIX_PATH "$ENV{LIBTORCH_PATH}")
    elseif(EXISTS "${CELLERATOR_PREFERRED_LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
        if(NOT Torch_DIR OR Torch_DIR MATCHES ".*/site-packages/torch/share/cmake/Torch/?$")
            set(Torch_DIR
                "${CELLERATOR_PREFERRED_LIBTORCH_ROOT}/share/cmake/Torch"
                CACHE PATH "Torch CMake package directory" FORCE)
            message(STATUS "Defaulting Torch_DIR to source-built libtorch at ${Torch_DIR}")
        endif()
        if(EXISTS "${CELLERATOR_PREFERRED_LIBTORCH_ROOT}/share/cmake/Caffe2/Caffe2Config.cmake")
            if(NOT Caffe2_DIR OR Caffe2_DIR MATCHES ".*/site-packages/torch/share/cmake/Caffe2/?$")
                set(Caffe2_DIR
                    "${CELLERATOR_PREFERRED_LIBTORCH_ROOT}/share/cmake/Caffe2"
                    CACHE PATH "Caffe2 CMake package directory" FORCE)
                message(STATUS "Defaulting Caffe2_DIR to source-built libtorch at ${Caffe2_DIR}")
            endif()
        endif()
        foreach(cellerator_torch_cache_var
                IN ITEMS
                    TORCH_LIBRARY
                    C10_CUDA_LIBRARY
                    c10_LIBRARY
                    kineto_LIBRARY)
            if(DEFINED ${cellerator_torch_cache_var}
               AND "${${cellerator_torch_cache_var}}" MATCHES ".*/site-packages/torch/.*")
                unset(${cellerator_torch_cache_var} CACHE)
            endif()
        endforeach()
    endif()
endif()

if(CELLERATOR_ENABLE_TORCH_MODELS)
    find_package(Torch REQUIRED)
endif()

find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS
        /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/lib
)

if(CELLERATOR_ENABLE_NCURSES_WORKBENCH)
    pkg_check_modules(NCURSESW REQUIRED ncursesw)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT CELLERATOR_CUDA_MODE STREQUAL "generic"
   AND NOT CELLERATOR_CUDA_MODE STREQUAL "native"
   AND NOT CELLERATOR_CUDA_MODE STREQUAL "native-extreme")
    message(FATAL_ERROR "CELLERATOR_CUDA_MODE must be one of: generic, native, native-extreme")
endif()

if(CELLERATOR_CUDA_MODE STREQUAL "native-extreme")
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING
        "Native-extreme mode hard-locks the build to Volta sm_70." FORCE)
    set(cellerator_extreme_arch_count 0)
    foreach(cellerator_cuda_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        math(EXPR cellerator_extreme_arch_count "${cellerator_extreme_arch_count} + 1")
        string(STRIP "${cellerator_cuda_arch}" cellerator_cuda_arch_clean)
        if(NOT cellerator_cuda_arch_clean MATCHES "^(70|7\\.0)(-real|-virtual)?$")
            message(FATAL_ERROR
                "CELLERATOR_CUDA_MODE=native-extreme is a Volta-only path and currently requires "
                "CMAKE_CUDA_ARCHITECTURES to contain only sm_70 entries.")
        endif()
    endforeach()
    if(cellerator_extreme_arch_count EQUAL 0)
        message(FATAL_ERROR
            "CELLERATOR_CUDA_MODE=native-extreme requires at least one explicit sm_70 architecture entry.")
    endif()
endif()

set(CELLERATOR_CUDA_MODE_GENERIC 0)
set(CELLERATOR_CUDA_MODE_NATIVE 0)
set(CELLERATOR_CUDA_MODE_NATIVE_EXTREME 0)
set(CELLERATOR_CUDA_MODE_ENUM generic)
if(CELLERATOR_CUDA_MODE STREQUAL "generic")
    set(CELLERATOR_CUDA_MODE_GENERIC 1)
elseif(CELLERATOR_CUDA_MODE STREQUAL "native")
    set(CELLERATOR_CUDA_MODE_NATIVE 1)
    set(CELLERATOR_CUDA_MODE_ENUM native)
else()
    set(CELLERATOR_CUDA_MODE_NATIVE_EXTREME 1)
    set(CELLERATOR_CUDA_MODE_ENUM native_extreme)
endif()

message(STATUS "Cellerator CUDA mode: ${CELLERATOR_CUDA_MODE}")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/support/cuda_mode.hh.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/cellerator_cuda_mode.hh
    @ONLY
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(cellerator_semantics INTERFACE)
target_include_directories(cellerator_semantics INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_semantics INTERFACE cudaBioTypes::cudaBioTypes)
target_compile_features(cellerator_semantics INTERFACE cxx_std_23)

add_library(cellerator_cuda_mode INTERFACE)
target_include_directories(cellerator_cuda_mode INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_compile_definitions(cellerator_cuda_mode INTERFACE
    CELLERATOR_CUDA_MODE_GENERIC=${CELLERATOR_CUDA_MODE_GENERIC}
    CELLERATOR_CUDA_MODE_NATIVE=${CELLERATOR_CUDA_MODE_NATIVE}
    CELLERATOR_CUDA_MODE_NATIVE_EXTREME=${CELLERATOR_CUDA_MODE_NATIVE_EXTREME}
)
target_compile_features(cellerator_cuda_mode INTERFACE cxx_std_17)

function(cellerator_enable_perf target)
    set(cellerator_cuda_perf_flags
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xptxas=-O3,-dlcm=ca
    )
    if(CELLERATOR_ENABLE_CUDA_LINEINFO)
        list(APPEND cellerator_cuda_perf_flags -lineinfo)
    endif()

    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG -march=native -mtune=native>
        $<$<COMPILE_LANGUAGE:CUDA>:${cellerator_cuda_perf_flags}>
    )
    target_link_libraries(${target} PRIVATE cellerator_cuda_mode)
    set_target_properties(${target} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE YES
    )
endfunction()

if(CELLERATOR_ENABLE_TORCH_MODELS)
    add_library(cellerator_torch INTERFACE)
    target_link_libraries(cellerator_torch INTERFACE CUDA::cudart ${TORCH_LIBRARIES})
    target_compile_features(cellerator_torch INTERFACE cxx_std_17)
    if(DEFINED TORCH_INCLUDE_DIRS)
        target_include_directories(cellerator_torch SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})
    endif()
    if(TORCH_CXX_FLAGS)
        separate_arguments(CELLERATOR_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(cellerator_torch INTERFACE ${CELLERATOR_TORCH_CXX_FLAGS})
    endif()

    add_library(cellerator_torch_bindings INTERFACE)
    target_include_directories(cellerator_torch_bindings INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(cellerator_torch_bindings INTERFACE cellerator_torch)
    target_compile_features(cellerator_torch_bindings INTERFACE cxx_std_17)

    add_library(cellerator_compute_model_ops STATIC
        src/compute/model_ops/model_ops.cu
    )
    target_include_directories(cellerator_compute_model_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(cellerator_compute_model_ops PRIVATE cellerator_torch CUDA::cudart)
    set_target_properties(cellerator_compute_model_ops PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )
    cellerator_enable_perf(cellerator_compute_model_ops)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix/matrix_io.cu)
    add_executable(matrixTest
        tests/matrix_test.cc
        src/matrix/matrix_io.cu
    )
    set_source_files_properties(
        tests/matrix_test.cc
        src/matrix/matrix_io.cu
        PROPERTIES LANGUAGE CUDA
    )
    target_include_directories(matrixTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(matrixTest PRIVATE CUDA::cusparse)
    target_compile_features(matrixTest PRIVATE cxx_std_17)
    set_target_properties(matrixTest PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    cellerator_enable_perf(matrixTest)

    add_executable(convertEmbryoMtxShards
        src/legacy/monoliths/load_embryo_mtx_shards.cu
        src/matrix/matrix_io.cu
    )
    target_include_directories(convertEmbryoMtxShards PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(convertEmbryoMtxShards PRIVATE CUDA::cusparse)
    target_compile_features(convertEmbryoMtxShards PRIVATE cxx_std_17)
    set_target_properties(convertEmbryoMtxShards PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    cellerator_enable_perf(convertEmbryoMtxShards)
    target_compile_options(convertEmbryoMtxShards PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp,-march=native,-mtune=native>
    )
    target_link_options(convertEmbryoMtxShards PRIVATE -fopenmp)
else()
    message(WARNING "Skipping legacy matrix targets because src/matrix/matrix_io.cu is not present in this checkout.")
endif()

add_executable(scrnaPreprocessBench
    bench/scrna_preprocess_bench.cu
)
target_include_directories(scrnaPreprocessBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(scrnaPreprocessBench PRIVATE CellShard::runtime CUDA::cusparse)
if(NCCL_LIBRARY)
    target_link_libraries(scrnaPreprocessBench PRIVATE ${NCCL_LIBRARY})
endif()
set_target_properties(scrnaPreprocessBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(scrnaPreprocessBench)

add_executable(preprocessFormatCompareBench
    bench/preprocess_format_compare_bench.cu
)
target_include_directories(preprocessFormatCompareBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(preprocessFormatCompareBench PRIVATE CellShard::runtime CUDA::cusparse)
set_target_properties(preprocessFormatCompareBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(preprocessFormatCompareBench)

add_executable(blockedEllOptimizationBench
    bench/blocked_ell_optimization_bench.cu
)
target_include_directories(blockedEllOptimizationBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(blockedEllOptimizationBench PRIVATE CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(blockedEllOptimizationBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(blockedEllOptimizationBench)

add_executable(cellShardDatasetH5Test
    tests/cellshard_dataset_h5_test.cu
)
target_include_directories(cellShardDatasetH5Test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellShardDatasetH5Test PRIVATE CellShard::runtime)
set_target_properties(cellShardDatasetH5Test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellShardDatasetH5Test)

add_executable(cellShardFetchBench
    bench/cellshard_fetch_bench.cu
)
target_include_directories(cellShardFetchBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellShardFetchBench PRIVATE CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(cellShardFetchBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellShardFetchBench)

add_executable(quantizedMatrixTest
    tests/quantized_matrix_test.cu
)
target_include_directories(quantizedMatrixTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedMatrixTest PRIVATE CUDA::cudart)
set_target_properties(quantizedMatrixTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedMatrixTest)

add_library(datasetIngestCompileTest OBJECT
    tests/dataset_ingest_compile_test.cu
)
target_include_directories(datasetIngestCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(datasetIngestCompileTest PRIVATE ${HDF5_INCLUDE_DIRS})
set_target_properties(datasetIngestCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(datasetIngestCompileTest)

add_library(cellerator_workbench
    src/workbench/dataset_workbench.cc
    src/workbench/dataset_workbench_cuda.cu
)
target_include_directories(cellerator_workbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cellerator_workbench PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(cellerator_workbench PRIVATE CellShard::runtime CUDA::cudart CUDA::cusparse ${HDF5_LIBRARIES})
if(NCCL_LIBRARY)
    target_link_libraries(cellerator_workbench PRIVATE ${NCCL_LIBRARY})
endif()
set_target_properties(cellerator_workbench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_workbench)

add_executable(cellShardFirstFileFixtureTest
    tests/cellshard_first_file_fixture_test.cu
)
target_include_directories(cellShardFirstFileFixtureTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cellShardFirstFileFixtureTest PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(cellShardFirstFileFixtureTest PRIVATE cellerator_workbench CellShard::runtime)
set_target_properties(cellShardFirstFileFixtureTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellShardFirstFileFixtureTest)

if(CELLERATOR_ENABLE_PYTHON)
    pybind11_add_module(_cellerator
        python/module.cc
    )
    target_include_directories(_cellerator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(_cellerator PRIVATE
        cellerator_workbench
        CellShard::runtime
        CUDA::cudart
        CUDA::cusparse
        ${HDF5_LIBRARIES}
    )
    if(NCCL_LIBRARY)
        target_link_libraries(_cellerator PRIVATE ${NCCL_LIBRARY})
    endif()
    set_target_properties(_cellerator PROPERTIES
        OUTPUT_NAME _cellerator
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )

    add_custom_target(celleratorPythonCompileTest DEPENDS _cellerator)

    if(CELLERATOR_ENABLE_CELLSHARD_PYTHON AND TARGET _cellshard)
        set(cellerator_python_build_root ${CMAKE_CURRENT_BINARY_DIR}/python)
        set_target_properties(_cellerator PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${cellerator_python_build_root}/cellerator
        )
        set_target_properties(_cellshard PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${cellerator_python_build_root}/cellshard
        )
        add_custom_target(celleratorPythonRuntimeStage
            COMMAND ${CMAKE_COMMAND} -E make_directory ${cellerator_python_build_root}/cellerator
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/python/cellerator/__init__.py
                ${cellerator_python_build_root}/cellerator/__init__.py
            COMMAND ${CMAKE_COMMAND} -E make_directory ${cellerator_python_build_root}/cellshard
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/extern/CellShard/python/cellshard/__init__.py
                ${cellerator_python_build_root}/cellshard/__init__.py
            DEPENDS _cellerator _cellshard
        )
        add_custom_target(celleratorPythonRuntimeTest
            COMMAND ${CMAKE_COMMAND} -E env
                PYTHONPATH=${cellerator_python_build_root}
                ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cellerator_python_smoke_test.py
            DEPENDS celleratorPythonRuntimeStage
        )
    endif()
endif()

add_executable(datasetWorkbenchRuntimeTest
    tests/dataset_workbench_runtime_test.cc
)
set_source_files_properties(
    tests/dataset_workbench_runtime_test.cc
    PROPERTIES LANGUAGE CUDA
)
target_include_directories(datasetWorkbenchRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(datasetWorkbenchRuntimeTest PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(datasetWorkbenchRuntimeTest PRIVATE cellerator_workbench CUDA::cudart CUDA::cusparse ${HDF5_LIBRARIES})
if(NCCL_LIBRARY)
    target_link_libraries(datasetWorkbenchRuntimeTest PRIVATE ${NCCL_LIBRARY})
endif()
set_target_properties(datasetWorkbenchRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(datasetWorkbenchRuntimeTest)

if(CELLERATOR_ENABLE_NCURSES_WORKBENCH)
    add_executable(celleratorWorkbench
        src/workbench/dataset_workbench_main.cc
    )
    target_include_directories(celleratorWorkbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(celleratorWorkbench PRIVATE ${HDF5_INCLUDE_DIRS})
    target_include_directories(celleratorWorkbench PRIVATE ${NCURSESW_INCLUDE_DIRS})
    target_compile_options(celleratorWorkbench PRIVATE ${NCURSESW_CFLAGS_OTHER})
    target_link_directories(celleratorWorkbench PRIVATE ${NCURSESW_LIBRARY_DIRS})
    target_link_libraries(celleratorWorkbench PRIVATE cellerator_workbench ${NCURSESW_LIBRARIES})
    set_target_properties(celleratorWorkbench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
endif()

add_executable(quantizedMatrixBench
    bench/quantized_matrix_bench.cu
)
target_include_directories(quantizedMatrixBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedMatrixBench PRIVATE CUDA::cudart)
set_target_properties(quantizedMatrixBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedMatrixBench)

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

add_library(cellerator_compute_forward_neighbors STATIC
    src/compute/neighbors/forward_neighbors/forward_neighbors.cu
)
target_include_directories(cellerator_compute_forward_neighbors PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_forward_neighbors PRIVATE CUDA::cudart)
set_target_properties(cellerator_compute_forward_neighbors PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_forward_neighbors)

add_executable(forwardNeighborsBench
    bench/forward_neighbors_bench.cc
)
target_include_directories(forwardNeighborsBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(forwardNeighborsBench PRIVATE cellerator_compute_forward_neighbors CUDA::cudart)
set_target_properties(forwardNeighborsBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(forwardNeighborsBench)

add_library(cellerator_compute_autograd STATIC
    src/compute/autograd/runtime.cu
    src/compute/autograd/kernels/base_sparse.cu
    src/compute/autograd/kernels/dist_sparse.cu
)
target_include_directories(cellerator_compute_autograd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_autograd PRIVATE CUDA::cudart CUDA::cusparse)
if(NCCL_LIBRARY)
    target_link_libraries(cellerator_compute_autograd PRIVATE ${NCCL_LIBRARY})
endif()
set_target_properties(cellerator_compute_autograd PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_autograd)

add_executable(computeAutogradBench
    bench/compute_autograd_bench.cu
)
target_include_directories(computeAutogradBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(computeAutogradBench PRIVATE cellerator_compute_autograd CUDA::cudart CUDA::cusparse)
if(NCCL_LIBRARY)
    target_link_libraries(computeAutogradBench PRIVATE ${NCCL_LIBRARY})
endif()
set_target_properties(computeAutogradBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(computeAutogradBench)

add_executable(cellShardRealSpmmBench
    bench/cellshard_real_spmm_bench.cu
)
target_include_directories(cellShardRealSpmmBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellShardRealSpmmBench PRIVATE cellerator_compute_autograd CUDA::cudart CUDA::cusparse)
if(NCCL_LIBRARY)
    target_link_libraries(cellShardRealSpmmBench PRIVATE ${NCCL_LIBRARY})
endif()
set_target_properties(cellShardRealSpmmBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellShardRealSpmmBench)

add_executable(forwardNeighborsCompileTest
    tests/forward_neighbors_compile_test.cc
)
target_include_directories(forwardNeighborsCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(forwardNeighborsCompileTest PRIVATE cellerator_compute_forward_neighbors CUDA::cudart)
set_target_properties(forwardNeighborsCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(forwardNeighborsCompileTest)

add_executable(computeAutogradRuntimeTest
    tests/compute_autograd_runtime_test.cu
)
target_include_directories(computeAutogradRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(computeAutogradRuntimeTest PRIVATE cellerator_compute_autograd CUDA::cudart CUDA::cusparse)
if(NCCL_LIBRARY)
    target_link_libraries(computeAutogradRuntimeTest PRIVATE ${NCCL_LIBRARY})
endif()
set_target_properties(computeAutogradRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(computeAutogradRuntimeTest)

add_executable(cellShardBlockedEllTest
    tests/cellshard_blocked_ell_test.cu
)
target_include_directories(cellShardBlockedEllTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellShardBlockedEllTest PRIVATE CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(cellShardBlockedEllTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellShardBlockedEllTest)

add_executable(cellShardExportRuntimeTest
    extern/CellShard/tests/cellshard_export_runtime_test.cu
)
target_include_directories(cellShardExportRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellShardExportRuntimeTest PRIVATE CellShard::export CUDA::cudart)
set_target_properties(cellShardExportRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellShardExportRuntimeTest)

add_executable(cudaBioTypesSemanticsTest
    tests/cudabiotypes_semantics_test.cc
)
target_include_directories(cudaBioTypesSemanticsTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cudaBioTypesSemanticsTest PRIVATE cellerator_semantics CUDA::cudart)
set_target_properties(cudaBioTypesSemanticsTest PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(cudaBioTypesSemanticsTest)

if(CELLERATOR_ENABLE_TORCH_MODELS)
    add_library(cellerator_quantize_cuda STATIC
        src/models/quantize/quantize_cuda.cu
    )
    target_include_directories(cellerator_quantize_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(cellerator_quantize_cuda PRIVATE cellerator_torch cellerator_compute_autograd CUDA::cudart)
    set_target_properties(cellerator_quantize_cuda PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    cellerator_enable_perf(cellerator_quantize_cuda)

    add_library(cellerator_developmental_time_cuda STATIC
        src/models/developmental_time_cuda/dtc_model.cu
    )
    target_include_directories(cellerator_developmental_time_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(cellerator_developmental_time_cuda PRIVATE cellerator_torch cellerator_compute_autograd CUDA::cudart)
    set_target_properties(cellerator_developmental_time_cuda PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    cellerator_enable_perf(cellerator_developmental_time_cuda)

    add_executable(developmentalTimeCompileTest
        tests/developmental_time_compile_test.cc
    )
    target_include_directories(developmentalTimeCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(developmentalTimeCompileTest PRIVATE cellerator_torch)
    set_target_properties(developmentalTimeCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(developmentalTimeCompileTest)

    add_executable(developmentalTimeCudaRuntimeTest
        tests/developmental_time_cuda_runtime_test.cc
    )
    target_include_directories(developmentalTimeCudaRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(developmentalTimeCudaRuntimeTest PRIVATE cellerator_torch cellerator_developmental_time_cuda CUDA::cudart)
    set_target_properties(developmentalTimeCudaRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(developmentalTimeCudaRuntimeTest)

    add_executable(developmentalTimeABBench
        bench/developmental_time_ab_bench.cc
    )
    target_include_directories(developmentalTimeABBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(developmentalTimeABBench PRIVATE cellerator_torch cellerator_developmental_time_cuda CUDA::cudart)
    set_target_properties(developmentalTimeABBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(developmentalTimeABBench)

    add_executable(torchBindingsCompileTest
        tests/torch_bindings_compile_test.cc
    )
    target_include_directories(torchBindingsCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(torchBindingsCompileTest PRIVATE cellerator_torch_bindings)
    set_target_properties(torchBindingsCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(torchBindingsCompileTest)

    add_executable(denseReduceCompileTest
        tests/dense_reduce_compile_test.cc
    )
    target_include_directories(denseReduceCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(denseReduceCompileTest PRIVATE cellerator_torch cellerator_compute_model_ops)
    set_target_properties(denseReduceCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(denseReduceCompileTest)

    add_executable(quantizeModelTest
        tests/quantize_model_test.cc
    )
    target_include_directories(quantizeModelTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(quantizeModelTest PRIVATE cellerator_torch cellerator_compute_model_ops cellerator_compute_forward_neighbors cellerator_quantize_cuda CUDA::cudart)
    set_target_properties(quantizeModelTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(quantizeModelTest)

    add_executable(modelCustomOpsTest
        tests/model_custom_ops_test.cc
    )
    target_include_directories(modelCustomOpsTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(modelCustomOpsTest PRIVATE cellerator_torch cellerator_compute_model_ops)
    set_target_properties(modelCustomOpsTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(modelCustomOpsTest)
endif()
