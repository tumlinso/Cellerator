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

option(CELLERATOR_ENABLE_TORCH_MODELS "Enable the CelleraTorch component targets" ON)
option(CELLERATOR_ENABLE_CUDA_LINEINFO "Emit CUDA line mapping for Nsight profiling" OFF)
option(CELLERATOR_ENABLE_HARDWARE_PROBE "Probe host CPU vector features and visible CUDA device capability at configure time." ON)
option(CELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES "Use the probed CUDA SM as the default CMAKE_CUDA_ARCHITECTURES when the user did not provide one." ON)
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
    "Legacy compatibility fallback for unspecified sparse surfaces; the planner selects operation projections: blocked_ell, sliced_ell, or csr")
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
set(CELLERATOR_CELLSHARD_SOURCE_DIR "" CACHE PATH
    "Optional local CellShard source checkout. Defaults to ../CellShard before find_package(CellShard).")
set(BASEPLANE_SOURCE_DIR "" CACHE PATH
    "Optional local Baseplane source checkout. Defaults to ../Baseplane before find_package(Baseplane).")

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
message(STATUS
    "Cellerator compatibility sparse fallback: ${CELLERATOR_DEFAULT_SPARSE_LAYOUT} "
    "(operation projections remain planner-selected)")

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

# Minimal CPU/CUDA-compatible biological identity and operand ABI. This target
# intentionally has no runtime or CUDA-library dependency so Baseplane can
# consume the contract without joining Cellerator's numerical runtime.
add_library(cellerator_biological_abi INTERFACE)
add_library(Cellerator::biological_abi ALIAS cellerator_biological_abi)
target_include_directories(cellerator_biological_abi INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_biological_abi INTERFACE cxx_std_17)

add_library(cellerator_identity_registry STATIC
    src/execution/identity_registry.cc
)
add_library(Cellerator::identity_registry ALIAS cellerator_identity_registry)
target_include_directories(cellerator_identity_registry PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_identity_registry PUBLIC
    Cellerator::biological_abi
)
target_compile_features(cellerator_identity_registry PUBLIC cxx_std_17)

add_executable(celleratorIdentityRegistryTest
    tests/execution_identity/identity_registry_test.cc
)
target_link_libraries(celleratorIdentityRegistryTest PRIVATE
    Cellerator::identity_registry
)

add_library(cellerator_runtime STATIC
    src/runtime/runtime.cu
)
add_library(Cellerator::runtime ALIAS cellerator_runtime)
target_include_directories(cellerator_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_runtime PUBLIC cxx_std_17)
target_link_libraries(cellerator_runtime PUBLIC CUDA::cudart)
target_link_libraries(cellerator_runtime PRIVATE CUDA::cublas CUDA::cusparse)
set_target_properties(cellerator_runtime PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

# CE-ARCH-22: backend-neutral operation preparation and direct dispatch over
# the biological execution contracts and the single Cellerator runtime.
add_library(cellerator_operation_core STATIC
    src/compute/math/operation_core/operation_core.cc
)
add_library(Cellerator::operation_core ALIAS cellerator_operation_core)
target_include_directories(cellerator_operation_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_operation_core PUBLIC cxx_std_17)
target_link_libraries(cellerator_operation_core PUBLIC
    Cellerator::biological_abi
    Cellerator::runtime
)
set_target_properties(cellerator_operation_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

add_executable(celleratorOperationCoreTest
    tests/math_core/operation_core_test.cc
)
target_link_libraries(celleratorOperationCoreTest PRIVATE
    Cellerator::operation_core
)
set_target_properties(celleratorOperationCoreTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

# CE-ARCH-60: keep the recovered execution substrate and its concurrency
# contract in the normal build graph. This is the only supported math runtime;
# the former DeviceMathContext island has been retired.
add_executable(celleratorExecutionSessionTest
    tests/runtime/execution_session_test.cu
)
target_link_libraries(celleratorExecutionSessionTest PRIVATE
    Cellerator::runtime
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)
set_target_properties(celleratorExecutionSessionTest PROPERTIES
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
target_link_libraries(cellerator_compute_matrix_convert PUBLIC Cellerator::runtime)
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

function(cellerator_add_baseplane_source baseplane_source_dir)
    if(NOT EXISTS "${baseplane_source_dir}/CMakeLists.txt")
        message(FATAL_ERROR "BASEPLANE_SOURCE_DIR does not point to a Baseplane source tree: ${baseplane_source_dir}")
    endif()
    set(cellerator_baseplane_required_sequence_headers
        include/Baseplane/seq/dna2.cuh
        include/Baseplane/seq/dna2_validity.hh
        include/Baseplane/seq/predicate_plan.hh
        include/Baseplane/seq/sequence_event.hh
    )
    foreach(cellerator_baseplane_header IN LISTS
            cellerator_baseplane_required_sequence_headers)
        if(NOT EXISTS "${baseplane_source_dir}/${cellerator_baseplane_header}")
            message(FATAL_ERROR
                "Baseplane sequence predicate ABI incompatibility: required "
                "header ${cellerator_baseplane_header} is absent. Cellerator "
                "requires Baseplane sequence predicate ABI version 1. Update "
                "the Baseplane checkout or set BASEPLANE_SOURCE_DIR to a "
                "compatible source tree: ${baseplane_source_dir}")
        endif()
    endforeach()
    if(NOT DEFINED BASEPLANE_ENABLE_CUDA)
        set(BASEPLANE_ENABLE_CUDA ON CACHE BOOL
            "Enable the CUDA Baseplane sequence kernels preferred by Cellerator.")
    endif()
    if(NOT DEFINED BASEPLANE_ENABLE_CUDA_LINEINFO)
        set(BASEPLANE_ENABLE_CUDA_LINEINFO ${CELLERATOR_ENABLE_CUDA_LINEINFO} CACHE BOOL
            "Emit CUDA line mapping for Nsight profiling inside Baseplane.")
    endif()
    add_subdirectory("${baseplane_source_dir}" "${CMAKE_CURRENT_BINARY_DIR}/Baseplane" EXCLUDE_FROM_ALL)
endfunction()

if(BASEPLANE_SOURCE_DIR)
    cellerator_add_baseplane_source("${BASEPLANE_SOURCE_DIR}")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../Baseplane/CMakeLists.txt")
    cellerator_add_baseplane_source("${CMAKE_CURRENT_SOURCE_DIR}/../Baseplane")
else()
    find_package(Baseplane CONFIG REQUIRED)
endif()

if(NOT TARGET Baseplane::seq)
    message(FATAL_ERROR "Cellerator requires Baseplane::seq for sequence bit primitives")
endif()
get_target_property(cellerator_baseplane_sequence_predicate_abi
    Baseplane::seq BASEPLANE_SEQUENCE_PREDICATE_ABI_VERSION)
if(NOT cellerator_baseplane_sequence_predicate_abi
   OR NOT cellerator_baseplane_sequence_predicate_abi STREQUAL "1")
    message(FATAL_ERROR
        "Baseplane sequence predicate ABI incompatibility: Cellerator requires "
        "Baseplane::seq ABI version 1, but the target reports "
        "'${cellerator_baseplane_sequence_predicate_abi}'. Update the "
        "Baseplane checkout or package before configuring Cellerator.")
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config.cuh.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/Cellerator/config.cuh
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

cellerator_enable_perf(cellerator_runtime)
cellerator_enable_perf(cellerator_compute_matrix_convert)

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

add_library(cellerator_compute_runtime STATIC
    src/compute/runtime/runtime.cu
    src/compute/runtime/reduce.cu
)
target_include_directories(cellerator_compute_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_runtime PUBLIC CellShard::runtime Cellerator::runtime Cellerator::dist CUDA::cublas)
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
    src/compute/packing/gene_candidate_discovery.cc
    src/compute/packing/gene_candidate_minhash.cu
    src/compute/packing/gene_candidate_discovery_cuda.cu
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
    src/compute/preprocess/fleet.cu
    src/compute/preprocess/fused_workflows.cu
    src/compute/preprocess/gene_stats.cu
    src/compute/preprocess/normalize.cu
    src/compute/preprocess/preprocess.cu
    src/compute/preprocess/qc_metrics.cu
    src/compute/preprocess/workspace.cu
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
    src/preprocess/cellshard_session.cu
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

    add_executable(celleratorPreprocessFormatCompareBench
        bench/preprocess/preprocess_format_compare_bench.cu
    )
    target_include_directories(celleratorPreprocessFormatCompareBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(celleratorPreprocessFormatCompareBench PRIVATE Cellerator::preprocess CUDA::cudart)
    set_target_properties(celleratorPreprocessFormatCompareBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    cellerator_enable_perf(celleratorPreprocessFormatCompareBench)

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
    src/compute/sequence/baseplane_strategy.cc
    src/compute/sequence/baseplane_integration.cu
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
    tests/sequence/baseplane_integration_test.cu
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

add_subdirectory(components/CellPack)

add_library(cellpack_statistical_validation STATIC
    components/CellPack/src/statistical_validation.cc
)
add_library(CellPack::statistical_validation ALIAS cellpack_statistical_validation)
target_include_directories(cellpack_statistical_validation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
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
    components/CellPack/tests/statistical_validation_test.cc
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
    components/CellPack/src/record_statistical_validation.cc
)
add_library(CellPack::record_statistical_validation ALIAS
    cellpack_record_statistical_validation)
target_include_directories(cellpack_record_statistical_validation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
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
    components/CellPack/tests/record_statistical_validation_test.cc
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
    components/CellPack/src/tile_statistical_validation.cc
)
add_library(CellPack::tile_statistical_validation ALIAS
    cellpack_tile_statistical_validation)
target_include_directories(cellpack_tile_statistical_validation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
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
    components/CellPack/tests/tile_statistical_validation_test.cc
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
    components/CellPack/src/alternating_refinement.cc
)
add_library(CellPack::alternating_refinement ALIAS
    cellpack_alternating_refinement)
target_include_directories(cellpack_alternating_refinement PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
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
    components/CellPack/tests/alternating_refinement_test.cc
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
    components/CellPack/src/runtime_statistical_validation.cc
)
add_library(CellPack::runtime_statistical_validation ALIAS
    cellpack_runtime_statistical_validation)
target_include_directories(cellpack_runtime_statistical_validation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
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
    components/CellPack/tests/runtime_statistical_validation_test.cc
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
    components/CellPack/src/local_cell_ordering.cc
    components/CellPack/src/local_cell_ordering_cuda.cu
)
add_library(CellPack::local_cell_ordering ALIAS cellpack_local_cell_ordering)
target_include_directories(cellpack_local_cell_ordering PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
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
    components/CellPack/tests/local_cell_ordering_test.cu
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
    components/CellPack/bench/local_cell_ordering_bench.cu
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
    components/CellPack/src/apply_plan.cc
    components/CellPack/src/apply_plan_cuda.cu
)
add_library(CellPack::apply_plan ALIAS cellpack_apply_plan)
target_include_directories(cellpack_apply_plan PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
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
    components/CellPack/src/feature_weighted_row_reduction.cc
)
add_library(CellPack::feature_weighted_row_reduction ALIAS
    cellpack_feature_weighted_row_reduction)
target_include_directories(cellpack_feature_weighted_row_reduction PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
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
    components/CellPack/src/persistent_packing_payload.cc
)
add_library(CellPack::persistent_packing_payload ALIAS
    cellpack_persistent_packing_payload)
target_include_directories(cellpack_persistent_packing_payload PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
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
    components/CellPack/tests/persistent_packing_payload_test.cu
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
    components/CellPack/src/semantic_geometry.cc
)
add_library(CellPack::semantic_geometry ALIAS cellpack_semantic_geometry)
target_include_directories(cellpack_semantic_geometry PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
)
target_link_libraries(cellpack_semantic_geometry PUBLIC
    CellPack::cellpack
    Cellerator::biological_abi
)
target_compile_features(cellpack_semantic_geometry PUBLIC cxx_std_17)

add_executable(cellPackSemanticGeometryAdapterTest
    components/CellPack/tests/semantic_geometry_adapter_test.cc
)
target_link_libraries(cellPackSemanticGeometryAdapterTest PRIVATE
    CellPack::semantic_geometry
)
set_target_properties(cellPackSemanticGeometryAdapterTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

add_library(cellpack_execution_image_v2 STATIC
    components/CellPack/src/persistence/execution_image_v2.cc
    components/CellPack/src/persistence/execution_image_v2_cpk1.cc
)
add_library(CellPack::execution_image_v2 ALIAS cellpack_execution_image_v2)
target_include_directories(cellpack_execution_image_v2 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
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
    CellShard::headers
)
target_compile_features(cellerator_opaque_execution_artifact PUBLIC cxx_std_17)

add_executable(cellPackExecutionImageV2Test
    components/CellPack/tests/persistence/execution_image_v2_test.cc
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
    components/CellPack/tests/persistence/execution_image_v2_device_test.cu
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

# CE-ARCH-71: operation-core activation of the preserved CP-BP v1 row-masked
# N=1 direct consumer. This adapter adds no kernel or persistence format.
add_library(cellerator_row_masked_n1_candidate STATIC
    src/compute/math/operation_core/row_masked_n1_candidate.cu
)
add_library(Cellerator::row_masked_n1_candidate ALIAS
    cellerator_row_masked_n1_candidate)
target_include_directories(cellerator_row_masked_n1_candidate PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_row_masked_n1_candidate PUBLIC
    Cellerator::operation_core
    CellPack::feature_weighted_row_reduction_cuda
    CellPack::persistent_packing_payload
)
set_target_properties(cellerator_row_masked_n1_candidate PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

# CE-ARCH-72: operation-core activation of the existing conventional CSR N=1
# execution path. Projection construction remains explicit and caller-owned.
add_library(cellerator_csr_fallback_candidate STATIC
    src/compute/math/operation_core/csr_fallback_candidate.cu
)
add_library(Cellerator::csr_fallback_candidate ALIAS
    cellerator_csr_fallback_candidate)
target_include_directories(cellerator_csr_fallback_candidate PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(cellerator_csr_fallback_candidate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(cellerator_csr_fallback_candidate PUBLIC
    Cellerator::operation_core
    CellPack::apply_plan
    CellPack::persistent_packing_payload
    CUDA::cudart
)
set_target_properties(cellerator_csr_fallback_candidate PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

# CE-ARCH-75: a second projection of the frozen CP-BP semantic structure.
# Construction and value packing are explicit host preparation; prepared
# execution consumes the native feature-major payload directly for N=1..16.
add_library(cellerator_feature_major_projection STATIC
    src/compute/math/physical_feature_major.cc
)
add_library(Cellerator::feature_major_projection ALIAS
    cellerator_feature_major_projection)
target_include_directories(cellerator_feature_major_projection PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_feature_major_projection PUBLIC
    CellPack::persistent_packing_payload
    Cellerator::biological_abi
)
target_compile_features(cellerator_feature_major_projection PUBLIC cxx_std_17)

add_library(cellerator_transpose_projection STATIC
    src/compute/math/physical_transpose.cc
)
add_library(Cellerator::transpose_projection ALIAS
    cellerator_transpose_projection)
target_include_directories(cellerator_transpose_projection PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_transpose_projection PUBLIC
    Cellerator::feature_major_projection
    Cellerator::biological_abi
)
target_compile_features(cellerator_transpose_projection PUBLIC cxx_std_17)

add_library(cellerator_feature_major_small_n_candidate STATIC
    src/compute/math/operation_core/feature_major_small_n_candidate.cu
)
add_library(Cellerator::feature_major_small_n_candidate ALIAS
    cellerator_feature_major_small_n_candidate)
target_include_directories(cellerator_feature_major_small_n_candidate PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_feature_major_small_n_candidate PUBLIC
    Cellerator::operation_core
    Cellerator::feature_major_projection
    CUDA::cudart
)
set_target_properties(cellerator_feature_major_small_n_candidate PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_library(cellerator_transpose_backward_candidate STATIC
    src/compute/math/operation_core/transpose_backward_candidate.cu
)

add_library(cellerator_native_training_slice STATIC
    src/compute/math/native_training_slice.cu
)
add_library(Cellerator::native_training_slice ALIAS
    cellerator_native_training_slice)
target_include_directories(cellerator_native_training_slice PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_native_training_slice PUBLIC
    Cellerator::feature_major_projection
    Cellerator::transpose_projection
    CUDA::cudart
)
set_target_properties(cellerator_native_training_slice PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
add_library(Cellerator::transpose_backward_candidate ALIAS
    cellerator_transpose_backward_candidate)
target_include_directories(cellerator_transpose_backward_candidate PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_transpose_backward_candidate PUBLIC
    Cellerator::operation_core
    Cellerator::transpose_projection
    CUDA::cudart
)
set_target_properties(cellerator_transpose_backward_candidate PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_library(cellerator_planner STATIC
    src/planner/end_to_end_planner.cc
    src/distributed/hierarchy.cc
    components/CellPack/src/optimization/objective_v2.cc
    src/planner/objective_v2_calibration.cc
)
add_library(Cellerator::planner ALIAS cellerator_planner)
target_include_directories(cellerator_planner PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_planner PUBLIC
    Cellerator::operation_core
    CellPack::semantic_geometry
    CellPack::alternating_refinement
)
target_compile_features(cellerator_planner PUBLIC cxx_std_17)

add_library(cellerator_candidate_measurement STATIC
    src/planner/candidate_measurement.cu
)
add_library(Cellerator::candidate_measurement ALIAS
    cellerator_candidate_measurement)
target_include_directories(cellerator_candidate_measurement PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_candidate_measurement PUBLIC
    Cellerator::planner
    Cellerator::operation_core
    CUDA::cudart
)
set_target_properties(cellerator_candidate_measurement PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_executable(celleratorPlannerV1Test
    tests/planner/end_to_end_planner_test.cc
)
add_executable(celleratorConnectedOperationPlannerTest
    tests/planner/connected_operation_planner_test.cc
)
add_executable(celleratorObjectiveV2CalibrationTest
    tests/planner/objective_v2_calibration_test.cc
)
add_executable(celleratorHierarchyPartitionBoundaryTest
    tests/distributed/hierarchy_partition_boundary_test.cc
)

add_executable(celleratorCandidateMeasurementTest
    tests/planner/candidate_measurement_test.cu
)
target_link_libraries(celleratorCandidateMeasurementTest PRIVATE
    Cellerator::candidate_measurement
    Cellerator::row_masked_n1_candidate
    Cellerator::csr_fallback_candidate
)
set_target_properties(celleratorCandidateMeasurementTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorPlannerV1Test PRIVATE Cellerator::planner)
target_link_libraries(celleratorConnectedOperationPlannerTest PRIVATE
    Cellerator::planner
)
target_link_libraries(celleratorObjectiveV2CalibrationTest PRIVATE
    Cellerator::planner
)
target_link_libraries(celleratorHierarchyPartitionBoundaryTest PRIVATE
    Cellerator::planner
)
set_target_properties(celleratorPlannerV1Test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
set_target_properties(celleratorConnectedOperationPlannerTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
set_target_properties(celleratorObjectiveV2CalibrationTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
set_target_properties(celleratorHierarchyPartitionBoundaryTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

add_executable(celleratorRowMaskedN1CandidateTest
    tests/math_core/row_masked_n1_candidate_test.cu
)

add_executable(celleratorCsrFallbackCandidateTest
    tests/math_core/csr_fallback_candidate_test.cu
)
add_executable(celleratorFeatureMajorSmallNCandidateTest
    tests/math_core/feature_major_small_n_candidate_test.cu
)
add_executable(celleratorTransposeBackwardCandidateTest
    tests/math_core/transpose_backward_candidate_test.cu
)
add_executable(celleratorNativeTrainingSliceTest
    tests/math_core/native_training_slice_test.cu
)
add_executable(celleratorCeArch76CandidateBench
    bench/math/feature_major_candidate_compare_bench.cu
)
add_executable(celleratorValueGenerationReuseTest
    tests/math_core/value_generation_reuse_test.cu
)
target_link_libraries(celleratorValueGenerationReuseTest PRIVATE
    CUDA::cudart
)
set_target_properties(celleratorValueGenerationReuseTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorCsrFallbackCandidateTest PRIVATE
    Cellerator::csr_fallback_candidate
    Cellerator::row_masked_n1_candidate
    Cellerator::planner
)
set_target_properties(celleratorCsrFallbackCandidateTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorFeatureMajorSmallNCandidateTest PRIVATE
    Cellerator::feature_major_small_n_candidate
    Cellerator::row_masked_n1_candidate
    Cellerator::csr_fallback_candidate
    Cellerator::planner
    CellPack::execution_image_v2
    cellerator_math_v1_evidence
)
set_target_properties(celleratorFeatureMajorSmallNCandidateTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorTransposeBackwardCandidateTest PRIVATE
    Cellerator::transpose_backward_candidate
    Cellerator::feature_major_small_n_candidate
    CellPack::execution_image_v2
    cellerator_math_v1_evidence
)
set_target_properties(celleratorTransposeBackwardCandidateTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorNativeTrainingSliceTest PRIVATE
    Cellerator::native_training_slice
    Cellerator::feature_major_projection
    Cellerator::transpose_projection
    CUDA::cudart
)
set_target_properties(celleratorNativeTrainingSliceTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorCeArch76CandidateBench PRIVATE
    Cellerator::feature_major_small_n_candidate
    Cellerator::row_masked_n1_candidate
    Cellerator::csr_fallback_candidate
    cellerator_math_v1_evidence
)
set_target_properties(celleratorCeArch76CandidateBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorRowMaskedN1CandidateTest PRIVATE
    Cellerator::row_masked_n1_candidate
    Cellerator::planner
)
set_target_properties(celleratorRowMaskedN1CandidateTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

# The surviving CP-Math v1 pieces are validation and compatibility evidence,
# not a runtime or planner. They remain explicit, EXCLUDE_FROM_ALL targets so
# old referees and projection adapters stay reproducible without entering the
# supported operation/session graph.
add_library(cellerator_math_v1_evidence STATIC EXCLUDE_FROM_ALL
    src/compute/math/operation.cc
    src/compute/math/operation_signature.cc
    src/compute/math/physical_csr.cc
    src/compute/math/physical_bell.cc
    src/compute/math/physical_bell_candidates.cc
    src/compute/math/native_tile_view.cc
    src/compute/math/referee.cc
)
target_include_directories(cellerator_math_v1_evidence PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
)
target_link_libraries(cellerator_math_v1_evidence PUBLIC
    CellPack::persistent_packing_payload
    CUDA::cudart
)
target_compile_features(cellerator_math_v1_evidence PUBLIC cxx_std_17)

add_library(cellerator_packed_dense_operand STATIC
    src/compute/math/packed_dense_operand.cu
)
add_library(Cellerator::packed_dense_operand ALIAS
    cellerator_packed_dense_operand)
target_include_directories(cellerator_packed_dense_operand PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
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
    components/CellPack/tests/feature_weighted_row_reduction_test.cc
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
    components/CellPack/src/hardware_cost_model.cc
)
add_library(CellPack::hardware_cost_model ALIAS cellpack_hardware_cost_model)
target_include_directories(cellpack_hardware_cost_model PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/components/CellPack/include>
)
target_link_libraries(cellpack_hardware_cost_model PUBLIC CellPack::cellpack)
set_target_properties(cellpack_hardware_cost_model PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_hardware_cost_model)

add_executable(cellPackHardwareCostModelTest
    components/CellPack/tests/hardware_cost_model_test.cc
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
    components/CellPack/tests/apply_plan_test.cu
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
    components/CellPack/tests/inferred_packing_pipeline_test.cc
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
    components/CellPack/bench/cellpack_apply_plan_bench.cu
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
