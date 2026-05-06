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
    set(CELLERATOR_CMAKE_CUDA_ARCHITECTURES_WAS_DEFAULTED 1)
    set(CMAKE_CUDA_ARCHITECTURES 70)
else()
    set(CELLERATOR_CMAKE_CUDA_ARCHITECTURES_WAS_DEFAULTED 0)
endif()

project(Cellerator LANGUAGES CXX CUDA)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(PkgConfig QUIET)

option(CELLERATOR_ENABLE_TORCH_MODELS "Enable libtorch-backed model targets" ON)
option(CELLERATOR_ENABLE_CUDA_LINEINFO "Emit CUDA line mapping for Nsight profiling" OFF)
option(CELLERATOR_ENABLE_HARDWARE_PROBE "Probe host CPU vector features and visible CUDA device capability at configure time." ON)
option(CELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES "Use the probed CUDA SM as the default CMAKE_CUDA_ARCHITECTURES when the user did not provide one." ON)
option(CELLERATOR_ENABLE_HIGHWAY "Enable Google Highway portable SIMD sequence kernels" ON)
option(CELLERATOR_BUILD_PREPROCESS_WORKBENCH "Build the optional Cellerator preprocessing ncurses workbench." ON)
option(CELLERATOR_BUILD_PREPROCESS_BENCHMARKS "Build Cellerator preprocessing benchmarks." ON)
option(CELLERATOR_ENABLE_PYTHON "Build the optional Cellerator Python bindings." OFF)
set(CELLERATOR_CUDA_MODE "generic" CACHE STRING
    "CUDA optimization mode: generic, native, or native-extreme")
set_property(CACHE CELLERATOR_CUDA_MODE PROPERTY STRINGS generic native native-extreme)
set(CELLERATOR_REAL_STORAGE "f16" CACHE STRING
    "Default Cellerator real storage type for new, unspecified in-memory work: f16, bf16, fp8_e4m3, fp8_e5m2, f32, or f64")
set_property(CACHE CELLERATOR_REAL_STORAGE PROPERTY STRINGS f16 bf16 fp8_e4m3 fp8_e5m2 f32 f64)
set(CELLERATOR_REAL_COMPUTE "f32" CACHE STRING
    "Default Cellerator real compute type for new, unspecified arithmetic: f32 or f64")
set_property(CACHE CELLERATOR_REAL_COMPUTE PROPERTY STRINGS f32 f64)
set(CELLERATOR_REAL_ACCUM "f32" CACHE STRING
    "Default Cellerator real accumulation type for new, unspecified reductions: f32 or f64")
set_property(CACHE CELLERATOR_REAL_ACCUM PROPERTY STRINGS f32 f64)
set(CELLERATOR_DEFAULT_SPARSE_LAYOUT "blocked_ell" CACHE STRING
    "Default sparse layout for new, unspecified Cellerator execution data: blocked_ell, sliced_ell, or csr")
set_property(CACHE CELLERATOR_DEFAULT_SPARSE_LAYOUT PROPERTY STRINGS blocked_ell sliced_ell csr)
set(CELLERATOR_DEFAULT_TRAINING_PRECISION "f32" CACHE STRING
    "Default training precision policy for future model configs: f32, mixed_f16, mixed_bf16, or mixed_fp8")
set_property(CACHE CELLERATOR_DEFAULT_TRAINING_PRECISION PROPERTY STRINGS f32 mixed_f16 mixed_bf16 mixed_fp8)
set(CELLERATOR_DEFAULT_GRADIENT_CLIPPING "none" CACHE STRING
    "Default gradient clipping policy for future model configs: none, global_norm, or value")
set_property(CACHE CELLERATOR_DEFAULT_GRADIENT_CLIPPING PROPERTY STRINGS none global_norm value)
set(CELLERATOR_DEFAULT_GRADIENT_CLIP_NORM "1.0" CACHE STRING
    "Default global-norm gradient clipping threshold for future model configs")
set(CELLERATOR_DEFAULT_GRADIENT_CLIP_VALUE "1.0" CACHE STRING
    "Default value gradient clipping threshold for future model configs")
set(CELLERATOR_PREFERRED_LIBTORCH_ROOT "/usr/local" CACHE PATH
    "Preferred root for the source-built libtorch installation")
set(CELLERATOR_CELLSHARD_SOURCE_DIR "" CACHE PATH
    "Optional local CellShard source checkout. Defaults to ../CellShard before find_package(CellShard).")

if(CELLERATOR_ENABLE_HIGHWAY)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/highway/CMakeLists.txt")
        message(FATAL_ERROR
            "CELLERATOR_ENABLE_HIGHWAY=ON requires the vendored Highway submodule at external/highway. "
            "Run: git submodule update --init --recursive")
    endif()
    set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "Include Highway contrib/" FORCE)
    set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "Build Highway examples" FORCE)
    set(HWY_ENABLE_INSTALL OFF CACHE BOOL "Install Highway" FORCE)
    set(HWY_ENABLE_TESTS OFF CACHE BOOL "Enable Highway tests" FORCE)
    add_subdirectory(external/highway EXCLUDE_FROM_ALL)
endif()

function(cellerator_require_choice var)
    set(cellerator_allowed_values ${ARGN})
    set(cellerator_choice_ok 0)
    foreach(cellerator_allowed IN LISTS cellerator_allowed_values)
        if("${${var}}" STREQUAL "${cellerator_allowed}")
            set(cellerator_choice_ok 1)
        endif()
    endforeach()
    if(NOT cellerator_choice_ok)
        string(REPLACE ";" ", " cellerator_allowed_text "${cellerator_allowed_values}")
        message(FATAL_ERROR "${var} must be one of: ${cellerator_allowed_text}")
    endif()
endfunction()

function(cellerator_real_kind_code out_var real_kind)
    if("${real_kind}" STREQUAL "f16")
        set(${out_var} 3 PARENT_SCOPE)
    elseif("${real_kind}" STREQUAL "f32")
        set(${out_var} 4 PARENT_SCOPE)
    elseif("${real_kind}" STREQUAL "f64")
        set(${out_var} 5 PARENT_SCOPE)
    elseif("${real_kind}" STREQUAL "bf16")
        set(${out_var} 6 PARENT_SCOPE)
    elseif("${real_kind}" STREQUAL "fp8_e4m3")
        set(${out_var} 7 PARENT_SCOPE)
    elseif("${real_kind}" STREQUAL "fp8_e5m2")
        set(${out_var} 8 PARENT_SCOPE)
    else()
        message(FATAL_ERROR "unknown real kind: ${real_kind}")
    endif()
endfunction()

cellerator_require_choice(CELLERATOR_REAL_STORAGE f16 bf16 fp8_e4m3 fp8_e5m2 f32 f64)
cellerator_require_choice(CELLERATOR_REAL_COMPUTE f32 f64)
cellerator_require_choice(CELLERATOR_REAL_ACCUM f32 f64)
cellerator_require_choice(CELLERATOR_DEFAULT_SPARSE_LAYOUT blocked_ell sliced_ell csr)
cellerator_require_choice(CELLERATOR_DEFAULT_TRAINING_PRECISION f32 mixed_f16 mixed_bf16 mixed_fp8)
cellerator_require_choice(CELLERATOR_DEFAULT_GRADIENT_CLIPPING none global_norm value)

cellerator_real_kind_code(CELLERATOR_REAL_STORAGE_CODE "${CELLERATOR_REAL_STORAGE}")
cellerator_real_kind_code(CELLERATOR_REAL_COMPUTE_CODE "${CELLERATOR_REAL_COMPUTE}")
cellerator_real_kind_code(CELLERATOR_REAL_ACCUM_CODE "${CELLERATOR_REAL_ACCUM}")
set(CELLERATOR_REAL_STORAGE_ENUM "${CELLERATOR_REAL_STORAGE}")
set(CELLERATOR_REAL_COMPUTE_ENUM "${CELLERATOR_REAL_COMPUTE}")
set(CELLERATOR_REAL_ACCUM_ENUM "${CELLERATOR_REAL_ACCUM}")
set(CELLERATOR_DEFAULT_SPARSE_LAYOUT_ENUM "${CELLERATOR_DEFAULT_SPARSE_LAYOUT}")
set(CELLERATOR_DEFAULT_TRAINING_PRECISION_ENUM "${CELLERATOR_DEFAULT_TRAINING_PRECISION}")
set(CELLERATOR_DEFAULT_GRADIENT_CLIPPING_ENUM "${CELLERATOR_DEFAULT_GRADIENT_CLIPPING}")

message(STATUS
    "Cellerator default real: storage=${CELLERATOR_REAL_STORAGE}, "
    "compute=${CELLERATOR_REAL_COMPUTE}, accum=${CELLERATOR_REAL_ACCUM}")
message(STATUS "Cellerator default sparse layout: ${CELLERATOR_DEFAULT_SPARSE_LAYOUT}")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/detect_hardware.cmake)
cellerator_detect_host_vector_ops()
cellerator_detect_cuda_hardware()
if(CELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES
   AND CELLERATOR_CMAKE_CUDA_ARCHITECTURES_WAS_DEFAULTED
   AND CELLERATOR_DETECTED_CUDA_SM GREATER 0)
    set(CMAKE_CUDA_ARCHITECTURES "${CELLERATOR_DETECTED_CUDA_SM}" CACHE STRING
        "CUDA architectures detected from the first visible CUDA device." FORCE)
    message(STATUS "Cellerator CUDA architectures auto-detected as sm_${CELLERATOR_DETECTED_CUDA_SM}")
endif()

find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS
        /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/lib
)
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS
        /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/include
)

add_library(cellerator_dist INTERFACE)
add_library(Cellerator::dist ALIAS cellerator_dist)
target_include_directories(cellerator_dist INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_dist INTERFACE cxx_std_17)
target_link_libraries(cellerator_dist INTERFACE CUDA::cudart)
if(NCCL_LIBRARY)
    target_compile_definitions(cellerator_dist INTERFACE CELLERATOR_DIST_HAS_NCCL=1)
    target_link_libraries(cellerator_dist INTERFACE ${NCCL_LIBRARY})
    if(NCCL_INCLUDE_DIR)
        target_include_directories(cellerator_dist SYSTEM INTERFACE ${NCCL_INCLUDE_DIR})
    endif()
else()
    target_compile_definitions(cellerator_dist INTERFACE CELLERATOR_DIST_HAS_NCCL=0)
endif()

add_library(cellerator_core STATIC
    src/core/runtime/runtime.cu
)
add_library(Cellerator::core ALIAS cellerator_core)
target_include_directories(cellerator_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_core PUBLIC cxx_std_17)
target_link_libraries(cellerator_core PUBLIC CUDA::cudart)
target_link_libraries(cellerator_core PRIVATE CUDA::cublas CUDA::cusparse)
set_target_properties(cellerator_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_library(cellerator_compute_matrix_convert STATIC
    src/compute/matrix/convert/bucket.cu
    src/compute/matrix/convert/compressed.cu
)
add_library(Cellerator::compute_matrix_convert ALIAS cellerator_compute_matrix_convert)
target_include_directories(cellerator_compute_matrix_convert PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_matrix_convert PUBLIC Cellerator::core)
target_link_libraries(cellerator_compute_matrix_convert PRIVATE CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_compute_matrix_convert PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)

function(cellerator_add_cellshard_source cellshard_source_dir)
    if(NOT EXISTS "${cellshard_source_dir}/CMakeLists.txt")
        message(FATAL_ERROR "CELLERATOR_CELLSHARD_SOURCE_DIR does not point to a CellShard source tree: ${cellshard_source_dir}")
    endif()
    if(NOT EXISTS "${cellshard_source_dir}/include/CellShard/CellShard.hh")
        message(FATAL_ERROR
            "CellShard source tree must expose the package include layout at include/CellShard. "
            "Update the CellShard checkout or set CELLERATOR_CELLSHARD_SOURCE_DIR to a compatible source tree: "
            "${cellshard_source_dir}")
    endif()
    if(DEFINED CELLSHARD_ENABLE_CUDA AND NOT CELLSHARD_ENABLE_CUDA)
        message(FATAL_ERROR
            "Cellerator requires the CUDA CellShard runtime. "
            "Do not configure with CELLSHARD_ENABLE_CUDA=OFF.")
    endif()
    if(NOT DEFINED CELLSHARD_ENABLE_CUDA)
        set(CELLSHARD_ENABLE_CUDA ON CACHE BOOL
            "Enable the CUDA CellShard runtime required by Cellerator.")
    endif()
    if(NOT DEFINED CELLSHARD_ENABLE_CELLERATOR_QUANTIZED)
        set(CELLSHARD_ENABLE_CELLERATOR_QUANTIZED ON CACHE BOOL
            "Use the Cellerator-owned quantized format ABI inside CellShard.")
    endif()
    if(NOT DEFINED CELLSHARD_ENABLE_CUDA_LINEINFO)
        set(CELLSHARD_ENABLE_CUDA_LINEINFO ${CELLERATOR_ENABLE_CUDA_LINEINFO} CACHE BOOL
            "Emit CUDA line mapping for Nsight profiling inside CellShard.")
    endif()
    if(NOT DEFINED CELLSHARD_CUDA_MODE)
        set(CELLSHARD_CUDA_MODE ${CELLERATOR_CUDA_MODE} CACHE STRING
            "CellShard CUDA optimization mode: generic, native, or native-extreme.")
    endif()
    add_subdirectory("${cellshard_source_dir}" "${CMAKE_CURRENT_BINARY_DIR}/CellShard" EXCLUDE_FROM_ALL)
endfunction()

if(CELLERATOR_CELLSHARD_SOURCE_DIR)
    cellerator_add_cellshard_source("${CELLERATOR_CELLSHARD_SOURCE_DIR}")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../CellShard/CMakeLists.txt")
    cellerator_add_cellshard_source("${CMAKE_CURRENT_SOURCE_DIR}/../CellShard")
else()
    find_package(CellShard CONFIG REQUIRED)
endif()

if(NOT TARGET CellShard::runtime)
    message(FATAL_ERROR "Cellerator requires CellShard::runtime")
endif()
if(NOT TARGET CellShard::export)
    message(FATAL_ERROR "Cellerator compute dataset utilities require CellShard::export")
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

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/config.cuh.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/Cellerator/core/config.cuh
    @ONLY
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

cellerator_enable_perf(cellerator_core)
cellerator_enable_perf(cellerator_compute_matrix_convert)

set(CELLERATOR_SEQ_SOURCES
    src/seq/dna2.cu
)
list(APPEND CELLERATOR_SEQ_SOURCES src/seq/dna2_scalar.cpp)
if(CELLERATOR_ENABLE_HIGHWAY)
    list(APPEND CELLERATOR_SEQ_SOURCES src/seq/dna2_highway.cpp)
endif()

add_library(cellerator_seq STATIC
    ${CELLERATOR_SEQ_SOURCES}
)
add_library(Cellerator::seq ALIAS cellerator_seq)
target_include_directories(cellerator_seq PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_seq PUBLIC CUDA::cudart)
if(CELLERATOR_ENABLE_HIGHWAY)
    target_link_libraries(cellerator_seq PRIVATE hwy)
    target_compile_definitions(cellerator_seq PUBLIC CELLERATOR_ENABLE_HIGHWAY=1)
else()
    target_compile_definitions(cellerator_seq PUBLIC CELLERATOR_ENABLE_HIGHWAY=0)
endif()
set_target_properties(cellerator_seq PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellerator_seq)

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
    target_link_libraries(cellerator_torch_bindings INTERFACE cellerator_torch CellShard::runtime)
    target_compile_features(cellerator_torch_bindings INTERFACE cxx_std_17)

    add_library(cellerator_compute_model_ops STATIC
        src/compute/ml/model_ops/model_ops.cu
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

add_executable(quantizedMatrixTest
    tests/quantized_matrix_test.cu
)
target_include_directories(quantizedMatrixTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedMatrixTest PRIVATE CUDA::cudart)
set_target_properties(quantizedMatrixTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedMatrixTest)

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

add_library(cellerator_compute_runtime STATIC
    src/compute/runtime/runtime.cu
)
target_include_directories(cellerator_compute_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_runtime PUBLIC CellShard::runtime Cellerator::core Cellerator::dist CUDA::cublas)
target_link_libraries(cellerator_compute_runtime PRIVATE CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_compute_runtime PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_runtime)

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

add_library(cellerator_compute_sparse_ops STATIC
    src/compute/sparse/ops/kernels/base_sparse.cu
    src/compute/sparse/ops/kernels/dist_sparse.cu
)
target_include_directories(cellerator_compute_sparse_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_sparse_ops PUBLIC cellerator_compute_runtime)
target_link_libraries(cellerator_compute_sparse_ops PRIVATE CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_compute_sparse_ops PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_sparse_ops)

add_library(cellerator_compute_sparse_project STATIC
    src/compute/sparse/project/project.cu
)
target_include_directories(cellerator_compute_sparse_project PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_sparse_project PUBLIC cellerator_compute_sparse_ops cellerator_compute_runtime)
target_link_libraries(cellerator_compute_sparse_project PRIVATE CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_compute_sparse_project PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_sparse_project)

add_library(cellerator_compute_preprocess STATIC
    src/compute/preprocess/preprocess.cu
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
    src/preprocess/runtime.cc
)
add_library(Cellerator::preprocess ALIAS cellerator_preprocess)
target_include_directories(cellerator_preprocess PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(cellerator_preprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_preprocess PUBLIC Cellerator::compute_preprocess)
set_target_properties(cellerator_preprocess PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_preprocess)

if(CELLERATOR_BUILD_PREPROCESS_WORKBENCH AND PkgConfig_FOUND)
    pkg_check_modules(CELLERATOR_NCURSESW QUIET ncursesw)
    if(CELLERATOR_NCURSESW_FOUND)
        add_executable(celleratorPreprocessWorkbench
            src/preprocess/apps/workbench_main.cc
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
target_link_libraries(cellerator_sparse_ml INTERFACE cellerator_compute_dataset cellerator_compute_sparse_project cellerator_compute_sparse_ops cellerator_compute_runtime cellerator_compute_preprocess)

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

    add_executable(celleratorPreprocessFormatCompareBench
        bench/preprocess/preprocess_format_compare_bench.cu
    )
    target_include_directories(celleratorPreprocessFormatCompareBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(celleratorPreprocessFormatCompareBench PRIVATE Cellerator::preprocess CUDA::cudart)
    set_target_properties(celleratorPreprocessFormatCompareBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    cellerator_enable_perf(celleratorPreprocessFormatCompareBench)
endif()

add_executable(sequenceDna2Bench
    bench/seq/bench_dna2.cu
)
target_include_directories(sequenceDna2Bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sequenceDna2Bench PRIVATE cellerator_seq CUDA::cudart)
set_target_properties(sequenceDna2Bench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(sequenceDna2Bench)

add_executable(sequenceDna2CpuBench
    bench/seq/bench_dna2_cpu.cpp
)
target_include_directories(sequenceDna2CpuBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sequenceDna2CpuBench PRIVATE cellerator_seq)
set_target_properties(sequenceDna2CpuBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(sequenceDna2CpuBench)

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
target_link_libraries(coreSparseLayoutRuntimeTest PRIVATE cellerator_compute_matrix_convert CUDA::cudart CUDA::cusparse)
set_target_properties(coreSparseLayoutRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(coreSparseLayoutRuntimeTest)

add_executable(datasetRuntimeTest
    tests/dataset_runtime_test.cc
)
target_include_directories(datasetRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(datasetRuntimeTest PRIVATE cellerator_compute_dataset CUDA::cudart)
set_target_properties(datasetRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(datasetRuntimeTest)

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

add_executable(sequenceDna2Test
    tests/seq/test_dna2.cpp
)
target_include_directories(sequenceDna2Test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sequenceDna2Test PRIVATE cellerator_seq CUDA::cudart)
set_target_properties(sequenceDna2Test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(sequenceDna2Test)

add_executable(sequenceDna2CudaTest
    tests/seq/test_dna2_cuda.cu
)
target_include_directories(sequenceDna2CudaTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sequenceDna2CudaTest PRIVATE cellerator_seq CUDA::cudart)
set_target_properties(sequenceDna2CudaTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(sequenceDna2CudaTest)

add_executable(abiRuntimeTest
    tests/abi_runtime_test.cu
)
target_include_directories(abiRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(abiRuntimeTest PRIVATE cellerator_abi CUDA::cudart)
set_target_properties(abiRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(abiRuntimeTest)

add_library(cellerator_state_reduce_cuda STATIC
    src/models/state_reduce/state_reduce_cuda.cu
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
    src/models/developmental_time/dT_model.cu
)
target_include_directories(cellerator_developmental_time_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_developmental_time_cuda PRIVATE cellerator_compute_sparse_project cellerator_compute_sparse_ops cellerator_compute_runtime CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_developmental_time_cuda PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_developmental_time_cuda)

add_library(cellerator_developmental_time_trajectory_cuda STATIC
    src/models/developmental_time_trajectory/dTT_model.cu
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

if(CELLERATOR_ENABLE_TORCH_MODELS)
    add_library(cellerator_compute_quantize STATIC
        src/compute/ml/quantize/quantize_cuda.cu
    )
    target_include_directories(cellerator_compute_quantize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(cellerator_compute_quantize PRIVATE cellerator_torch cellerator_compute_sparse_ops cellerator_compute_runtime CUDA::cudart)
    set_target_properties(cellerator_compute_quantize PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    cellerator_enable_perf(cellerator_compute_quantize)

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
    target_link_libraries(denseReduceCompileTest PRIVATE cellerator_torch cellerator_compute_model_ops CellShard::runtime)
    set_target_properties(denseReduceCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(denseReduceCompileTest)

    add_executable(quantizePrimitiveTest
        tests/quantize_primitive_test.cc
    )
    target_include_directories(quantizePrimitiveTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(quantizePrimitiveTest PRIVATE cellerator_torch cellerator_compute_model_ops cellerator_compute_quantize CUDA::cudart)
    set_target_properties(quantizePrimitiveTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(quantizePrimitiveTest)

    add_executable(modelCustomOpsTest
        tests/model_custom_ops_test.cc
    )
    target_include_directories(modelCustomOpsTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(modelCustomOpsTest PRIVATE cellerator_torch cellerator_compute_model_ops CellShard::runtime)
    set_target_properties(modelCustomOpsTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
    cellerator_enable_perf(modelCustomOpsTest)
endif()
