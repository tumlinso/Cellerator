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

# Host-only module proof infrastructure is independent of native CUDA targets.
add_subdirectory(modules)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig QUIET)

# Native Cellerator is the default build. CelleraTorch remains an explicit
# compatibility adapter and is the only path that may discover Torch.
option(CELLERATOR_ENABLE_TORCH_MODELS "Enable the CelleraTorch compatibility component targets" OFF)
option(CELLERATOR_BUILD_TESTS "Build Cellerator test executables" ON)
option(CELLERATOR_ENABLE_CUDA_LINEINFO "Emit CUDA line mapping for Nsight profiling" OFF)
option(CELLERATOR_ENABLE_HARDWARE_PROBE "Probe host CPU vector features and visible CUDA device capability at configure time." ON)
option(CELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES "Use the probed CUDA SM as the default CMAKE_CUDA_ARCHITECTURES when the user did not provide one." ON)
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

# CE-EXOP provider inclusion and tuning are explicit build inputs.  The
# provider policy module deliberately does not inspect a visible device or
# mutate CMAKE_CUDA_ARCHITECTURES; binary content remains reproducible from
# the cache configuration alone.
include(cmake/providers/CelleratorProviderPolicy.cmake)

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

# Canonical subsystem target ownership. These files are included here because
# the runtime and format ABI must exist before the sibling CellShard project
# is added below.
include(src/runtime/CMakeLists.txt)
include(src/execution/CMakeLists.txt)
include(src/compute/foundation_targets.cmake)
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
    set(CELLSHARD_CELLERATOR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH
        "Cellerator source checkout used by CellShard compatibility conversion." FORCE)
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
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xptxas=-O3
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
cellerator_enable_perf(cellerator_runtime_fleet)

include(src/CMakeLists.txt)
# The legacy compute manifest declares the provider source inventories.  Split
# them into provider-owned targets after all of their shared dependencies have
# been declared, then emit the cold compiled-provider manifest.
include(cmake/providers/CelleratorProviderTargets.cmake)
cellerator_finalize_cuda_provider_targets()
include(src/geometry/owned_targets.cmake)
include(src/planner/CMakeLists.txt)

# CE-EXOP core-contract fan-in.  Keep the portfolio and profiling surfaces
# explicit: the portfolio is planner input rather than canonical selection,
# and profiling remains a cold, optional observation path.
target_sources(cellerator_semantic_geometry_v1 PRIVATE
    src/geometry/compiler/v2/exact_evaluator.cc
    src/geometry/compiler/v2/solution.cc
    src/geometry/compiler/v2/strategy_registry.cc
    src/geometry/compiler/v2/work_window.cc
    src/geometry/compiler/v2/workload_profile.cc
    src/geometry/persistence/chunk/chunk_manifest_v1.cc
    src/geometry/validation/scalable_validation_v2.cc
)
target_sources(cellerator_architecture_provider PRIVATE
    src/compute/architecture/target_cover/contract.cc
    src/compute/architecture/target_cover/strategy_registry.cc
)
target_sources(cellerator_planner PRIVATE
    src/planner/end_to_end_planner/connected_economics_v1.cc
    src/planner/end_to_end_planner/resource_cost_v1.cc
    src/planner/portfolio/candidate_workspace_v1.cc
    src/planner/portfolio/pareto_portfolio_v1.cc
    src/planner/portfolio/validation_v1.cc
)

add_library(cellerator_profiling_v1 STATIC
    src/profiling/mechanism_manifest.cc
    src/profiling/partition_export.cc
    src/profiling/resource_receipt.cc
    src/profiling/static_markers.cc
)
add_library(Cellerator::profiling_v1 ALIAS cellerator_profiling_v1)
target_include_directories(cellerator_profiling_v1 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_profiling_v1 PUBLIC cxx_std_17)

if(CELLERATOR_BUILD_TESTS)
    add_executable(cellerator_operation_core_v2_test
        src/compute/operation/operation_core_v2/operation_core_v2_test.cc)
    target_link_libraries(cellerator_operation_core_v2_test PRIVATE
        Cellerator::operation_core)
    target_compile_features(cellerator_operation_core_v2_test PRIVATE cxx_std_17)

    add_executable(cellerator_geometry_acquisition_v2_test
        src/execution/geometry_acquisition_v2/geometry_acquisition_v2_test.cc)
    target_link_libraries(cellerator_geometry_acquisition_v2_test PRIVATE
        Cellerator::geometry_acquisition)
    target_compile_features(cellerator_geometry_acquisition_v2_test PRIVATE cxx_std_17)

    add_executable(cellerator_geometry_v2_test
        src/geometry/compiler/v2/geometry_v2_test.cc)
    target_link_libraries(cellerator_geometry_v2_test PRIVATE
        Cellerator::architecture_provider
        Cellerator::semantic_geometry_v1)
    target_compile_features(cellerator_geometry_v2_test PRIVATE cxx_std_17)

    add_executable(cellerator_planner_portfolio_v1_test
        src/planner/portfolio/tests/planner_portfolio_v1_test.cc)
    target_link_libraries(cellerator_planner_portfolio_v1_test PRIVATE
        Cellerator::planner)
    target_compile_features(cellerator_planner_portfolio_v1_test PRIVATE cxx_std_17)
endif()

include(bench/CMakeLists.txt)
if(CELLERATOR_BUILD_TESTS)
    include(tests/CMakeLists.txt)

    # CE-EXOP final acceptance entry points. These targets keep correctness,
    # profiler-fixture validation, and CUDA compile validation explicit. The
    # CUDA target compiles sm_70 binaries only; device execution remains under
    # Project Control resource authority.
    add_custom_target(cellerator_ce_exop_host_acceptance
        COMMAND ${CMAKE_COMMAND} -E env
            CXX=${CMAKE_CXX_COMPILER}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/ce_exop/host/run_host_acceptance.sh
            ${CMAKE_CURRENT_SOURCE_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        USES_TERMINAL)

    add_custom_target(cellerator_ce_exop_profiler_acceptance
        COMMAND ${CMAKE_COMMAND} -E env
            CXX=${CMAKE_CXX_COMPILER}
            CUDA_ROOT=${CUDAToolkit_ROOT}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/profiling/ce_exop/run_fixture_validation.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        USES_TERMINAL)

    add_custom_target(cellerator_ce_exop_cuda_compile_acceptance
        COMMAND ${CMAKE_COMMAND} -E env
            PATH=${CUDAToolkit_BIN_DIR}:$ENV{PATH}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/ce_exop/cuda/run_cuda_validation.sh
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/ce-exop-cuda-validation
            --build-only
            all
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        USES_TERMINAL)
endif()

add_subdirectory(compat)
include(compat/cp_math_v1/CMakeLists.txt)
