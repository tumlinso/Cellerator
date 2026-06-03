cmake_minimum_required(VERSION 3.24)

# Cellerator is now the umbrella CellStack workspace. Choose the CUDA toolchain
# before enabling CUDA so child components inherit the same compiler pair.
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

find_package(CUDAToolkit REQUIRED)

option(CELLERATOR_ENABLE_CUDA_LINEINFO "Emit CUDA line mapping for Nsight profiling." OFF)
option(CELLERATOR_ENABLE_HARDWARE_PROBE "Probe host CPU vector features and visible CUDA device capability at configure time." ON)
option(CELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES "Use the probed CUDA SM as the default CMAKE_CUDA_ARCHITECTURES when the user did not provide one." ON)
option(CELLERATOR_ENABLE_ARCHIVE "Expose the archived pre-umbrella source tree as source-only targets." OFF)
set(CELLERATOR_CUDA_MODE "generic" CACHE STRING "CUDA optimization mode: generic, native, or native-extreme")
set_property(CACHE CELLERATOR_CUDA_MODE PROPERTY STRINGS generic native native-extreme)
set(CELLERATOR_REAL_STORAGE "f16" CACHE STRING "Default real storage type for unspecified in-memory work.")
set_property(CACHE CELLERATOR_REAL_STORAGE PROPERTY STRINGS f16 bf16 fp8_e4m3 fp8_e5m2 f32 f64)
set(CELLERATOR_REAL_COMPUTE "f32" CACHE STRING "Default real compute type for unspecified arithmetic.")
set_property(CACHE CELLERATOR_REAL_COMPUTE PROPERTY STRINGS f32 f64)
set(CELLERATOR_REAL_ACCUM "f32" CACHE STRING "Default real accumulation type for unspecified reductions.")
set_property(CACHE CELLERATOR_REAL_ACCUM PROPERTY STRINGS f32 f64)
set(CELLERATOR_DEFAULT_SPARSE_LAYOUT "blocked_ell" CACHE STRING "Default sparse execution layout.")
set_property(CACHE CELLERATOR_DEFAULT_SPARSE_LAYOUT PROPERTY STRINGS blocked_ell sliced_ell csr)
set(CELLERATOR_DEFAULT_TRAINING_PRECISION "f32" CACHE STRING "Default future training precision policy.")
set_property(CACHE CELLERATOR_DEFAULT_TRAINING_PRECISION PROPERTY STRINGS f32 mixed_f16 mixed_bf16 mixed_fp8)
set(CELLERATOR_DEFAULT_GRADIENT_CLIPPING "none" CACHE STRING "Default future gradient clipping policy.")
set_property(CACHE CELLERATOR_DEFAULT_GRADIENT_CLIPPING PROPERTY STRINGS none global_norm value)
set(CELLERATOR_DEFAULT_GRADIENT_CLIP_NORM "1.0" CACHE STRING "Default global-norm clipping threshold.")
set(CELLERATOR_DEFAULT_GRADIENT_CLIP_VALUE "1.0" CACHE STRING "Default value clipping threshold.")

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
cellerator_require_choice(CELLERATOR_CUDA_MODE generic native native-extreme)

cellerator_real_kind_code(CELLERATOR_REAL_STORAGE_CODE "${CELLERATOR_REAL_STORAGE}")
cellerator_real_kind_code(CELLERATOR_REAL_COMPUTE_CODE "${CELLERATOR_REAL_COMPUTE}")
cellerator_real_kind_code(CELLERATOR_REAL_ACCUM_CODE "${CELLERATOR_REAL_ACCUM}")
set(CELLERATOR_REAL_STORAGE_ENUM "${CELLERATOR_REAL_STORAGE}")
set(CELLERATOR_REAL_COMPUTE_ENUM "${CELLERATOR_REAL_COMPUTE}")
set(CELLERATOR_REAL_ACCUM_ENUM "${CELLERATOR_REAL_ACCUM}")
set(CELLERATOR_DEFAULT_SPARSE_LAYOUT_ENUM "${CELLERATOR_DEFAULT_SPARSE_LAYOUT}")
set(CELLERATOR_DEFAULT_TRAINING_PRECISION_ENUM "${CELLERATOR_DEFAULT_TRAINING_PRECISION}")
set(CELLERATOR_DEFAULT_GRADIENT_CLIPPING_ENUM "${CELLERATOR_DEFAULT_GRADIENT_CLIPPING}")

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

if(CELLERATOR_CUDA_MODE STREQUAL "native-extreme")
    foreach(cellerator_cuda_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        string(STRIP "${cellerator_cuda_arch}" cellerator_cuda_arch_clean)
        if(NOT cellerator_cuda_arch_clean MATCHES "^(70|7\\.0)(-real|-virtual)?$")
            message(FATAL_ERROR "CELLERATOR_CUDA_MODE=native-extreme requires only sm_70 architectures.")
        endif()
    endforeach()
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/foundation/support/cuda_mode.hh.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/cellerator_cuda_mode.hh
    @ONLY
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/foundation/core/config.cuh.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/Cellerator/core/config.cuh
    @ONLY
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/lib
)
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/include
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
    src/foundation/core/runtime/runtime.cu
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
    src/foundation/compute/matrix/convert/bucket.cu
    src/foundation/compute/matrix/convert/compressed.cu
)
add_library(Cellerator::compute_matrix_convert ALIAS cellerator_compute_matrix_convert)
target_include_directories(cellerator_compute_matrix_convert PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_compute_matrix_convert PUBLIC cxx_std_17)
target_link_libraries(cellerator_compute_matrix_convert PUBLIC Cellerator::core)
target_link_libraries(cellerator_compute_matrix_convert PRIVATE CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_compute_matrix_convert PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_library(cellerator_cuda_mode INTERFACE)
target_include_directories(cellerator_cuda_mode INTERFACE
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
    set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE YES)
endfunction()

cellerator_enable_perf(cellerator_core)
cellerator_enable_perf(cellerator_compute_matrix_convert)

set(BASEPLANE_ENABLE_CUDA ON CACHE BOOL "Enable CUDA sequence kernels in umbrella builds.")
set(BASEPLANE_ENABLE_CUDA_LINEINFO ${CELLERATOR_ENABLE_CUDA_LINEINFO} CACHE BOOL
    "Emit CUDA line mapping inside Baseplane.")
add_subdirectory(components/Baseplane)

set(CELLSHARD_ENABLE_CUDA ON CACHE BOOL "Enable the CUDA CellShard runtime required by Cellerator.")
set(CELLSHARD_ENABLE_CUDA_LINEINFO ${CELLERATOR_ENABLE_CUDA_LINEINFO} CACHE BOOL
    "Emit CUDA line mapping inside CellShard.")
set(CELLSHARD_CUDA_MODE ${CELLERATOR_CUDA_MODE} CACHE STRING
    "CellShard CUDA optimization mode inherited from the Cellerator umbrella.")
set(CELLSHARD_ENABLE_CELLERATOR_QUANTIZED ON CACHE BOOL
    "Use Cellerator-owned quantized format ABI inside CellShard.")
set(CELLSHARD_CELLERATOR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH
    "Cellerator umbrella source tree used for foundation headers." FORCE)
add_subdirectory(components/CellShard)

foreach(cellerator_component_target
        IN ITEMS
            baseplane_seq
            cellshard_headers
            cellshard_inspect
            cellshard_runtime
            cellshard_ingest
            cellshard_export
            cellshard_h5ad_python)
    if(TARGET ${cellerator_component_target})
        add_library(Cellerator::${cellerator_component_target} ALIAS ${cellerator_component_target})
    endif()
endforeach()

add_library(cellerator_umbrella INTERFACE)
add_library(Cellerator::cellerator ALIAS cellerator_umbrella)
target_link_libraries(cellerator_umbrella INTERFACE
    Cellerator::core
    Cellerator::dist
    Cellerator::compute_matrix_convert
)
if(TARGET Baseplane::seq)
    target_link_libraries(cellerator_umbrella INTERFACE Baseplane::seq)
endif()
if(TARGET CellShard::headers)
    target_link_libraries(cellerator_umbrella INTERFACE CellShard::headers)
endif()
if(TARGET CellShard::runtime)
    target_link_libraries(cellerator_umbrella INTERFACE CellShard::runtime)
endif()

add_executable(celleratorFoundationRuntimeTest
    tests/foundation_runtime_test.cu
)
target_include_directories(celleratorFoundationRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorFoundationRuntimeTest PRIVATE
    Cellerator::compute_matrix_convert
    Baseplane::seq
    CUDA::cudart
    CUDA::cusparse
)
set_target_properties(celleratorFoundationRuntimeTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(celleratorFoundationRuntimeTest)

if(CELLERATOR_ENABLE_ARCHIVE)
    add_custom_target(celleratorArchiveSources
        SOURCES
            Archive/current-cellerator/README.md
            Archive/current-cellerator/CMakeLists.txt
            Archive/current-cellerator/pyproject.toml
    )
    message(STATUS "Archived pre-umbrella Cellerator sources are available under Archive/current-cellerator")
endif()
