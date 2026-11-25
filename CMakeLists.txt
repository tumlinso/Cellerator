cmake_minimum_required(VERSION 3.28)

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

project(Cellerator LANGUAGES CXX)

include(cmake/CelleratorBuildContract.cmake)

option(CELLERATOR_ENABLE_MODULES "Build the primary and selected component C++20 modules" OFF)

# Selectable installed host/SM70 substrate components use actual source owners.
# Branch before legacy compiler/native/Torch exports to avoid namespace collisions.
option(CELLERATOR_BUILD_SUBSTRATE_SDK "Build the selected substrate SDK" OFF)
if(CELLERATOR_BUILD_SUBSTRATE_SDK)
    include(cmake/substrate/Substrate.cmake)
    return()
endif()

option(CELLERATOR_ENABLE_VELOCITY "Build native RNA velocity methods" OFF)
option(CELLERATOR_ENABLE_PREPROCESS "Build native single-cell preprocessing methods" OFF)
option(CELLERATOR_ENABLE_ANNDATA "Build the optional AnnData++ component" OFF)
option(CELLERATOR_ENABLE_RDS "Build the optional indexed RDS reader" OFF)

# Native session and execution targets are declared early. They also back the
# optional shared Torch producer, so PIC must be enabled before target creation.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CELLERATOR_ENABLE_CUDA "AUTO" CACHE STRING
    "Build CUDA numerical support: AUTO, ON, or OFF")
set_property(CACHE CELLERATOR_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)

string(TOUPPER "${CELLERATOR_ENABLE_CUDA}" CELLERATOR_ENABLE_CUDA)
if(NOT CELLERATOR_ENABLE_CUDA MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "CELLERATOR_ENABLE_CUDA must be AUTO, ON, or OFF")
endif()
if(NOT CELLERATOR_ENABLE_CUDA STREQUAL "OFF")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(CELLERATOR_HAS_CUDA TRUE)
    elseif(CELLERATOR_ENABLE_CUDA STREQUAL "ON")
        message(FATAL_ERROR "CELLERATOR_ENABLE_CUDA=ON requires a usable CUDA compiler")
    else()
        set(CELLERATOR_ENABLE_CUDA OFF)
        set(CELLERATOR_HAS_CUDA FALSE)
    endif()
else()
    set(CELLERATOR_HAS_CUDA FALSE)
endif()
find_package(PkgConfig QUIET)

# Python and Torch are optional integrations. Native Cellerator builds never
# discover either dependency unless its corresponding option is enabled.
option(CELLERATOR_ENABLE_PYTHON "Build the optional cellerator Python extension" OFF)
option(CELLERATOR_ENABLE_TORCH "Build the optional Cellerator Torch adapters" OFF)
option(CELLERATOR_ENABLE_TOKENIZATION
    "Build the optional ATen-backed tokenization facility" OFF)
option(CELLERATOR_ENABLE_NN
    "Build the optional native C++ neural-network foundation" OFF)
option(CELLERATOR_ENABLE_NN_CUDA
    "Build optional CUDA neural-network derivative adapters" OFF)
if(CELLERATOR_ENABLE_MODULES)
    include(GNUInstallDirs)
    include(cmake/CelleratorModuleInterfaces.cmake)
    cellerator_require_module_scanner()
    install(FILES "${PROJECT_SOURCE_DIR}/cmake/CelleratorModuleInterfaces.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator" COMPONENT NativeData)
    if(CELLERATOR_ENABLE_NN)
        install(FILES "${PROJECT_SOURCE_DIR}/cmake/CelleratorModuleInterfaces.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator" COMPONENT NN)
    endif()
endif()
option(CELLERATOR_BUILD_TESTS "Include Cellerator checks, examples, and benchmarks in the default build" ON)
set(CELLERATOR_CHECK_EXCLUDE_FROM_ALL)
if(NOT CELLERATOR_BUILD_TESTS)
    set(CELLERATOR_CHECK_EXCLUDE_FROM_ALL EXCLUDE_FROM_ALL)
endif()

# These files declare only checks, examples, and benchmarks in the root
# directory. Keep their named targets available when building just libraries.
function(cellerator_include_checks path)
    if(NOT CELLERATOR_BUILD_TESTS)
        get_property(existing DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    endif()
    include("${PROJECT_SOURCE_DIR}/${path}")
    if(NOT CELLERATOR_BUILD_TESTS)
        get_property(checks DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
        if(existing)
            list(REMOVE_ITEM checks ${existing})
        endif()
        if(checks)
            set_target_properties(${checks} PROPERTIES EXCLUDE_FROM_ALL TRUE)
        endif()
    endif()
endfunction()

option(CELLERATOR_BUILD_NATIVE_DATA_TESTS
    "Build focused native data and cellular geometry tests" OFF)
option(CELLERATOR_BUILD_NATIVE_DATA_BENCHMARKS
    "Build the optional Torch/CUDA native-data comparison benchmark" OFF)
if(CELLERATOR_BUILD_NATIVE_DATA_BENCHMARKS
   AND CELLERATOR_ENABLE_CUDA STREQUAL "OFF")
    message(FATAL_ERROR
        "CELLERATOR_BUILD_NATIVE_DATA_BENCHMARKS requires CUDA; configure with CELLERATOR_ENABLE_CUDA=AUTO or ON")
endif()
option(CELLERATOR_ENABLE_CUDA_LINEINFO "Emit CUDA line mapping for Nsight profiling" OFF)
option(CELLERATOR_ENABLE_HARDWARE_PROBE "Probe host CPU vector features and visible CUDA device capability at configure time." ON)
option(CELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES "Use the probed CUDA SM as the default CMAKE_CUDA_ARCHITECTURES when the user did not provide one." ON)

include(cmake/CelleratorHost.cmake)
add_subdirectory(components/sequence)
add_subdirectory(src/math/reference)
add_subdirectory(src/ops/numeric/binary16)
option(CELLERATOR_BUILD_SEMANTIC_SPINE_V1 "Build the bounded Semantic Spine v1 conformance suite" OFF)
option(CELLERATOR_BUILD_RELATION_UPDATE_SPINE_V1 "Build the internal relation update spine and conformance suite" OFF)
if(CELLERATOR_BUILD_RELATION_UPDATE_SPINE_V1)
    set(CELLERATOR_BUILD_SEMANTIC_SPINE_V1 ON)
endif()
if(CELLERATOR_BUILD_SEMANTIC_SPINE_V1)
    include(cmake/SemanticSpineV1.cmake)
endif()
if(CELLERATOR_BUILD_RELATION_UPDATE_SPINE_V1)
    if(CELLERATOR_BUILD_TESTS)
        enable_testing()
    endif()
    cellerator_include_checks(tests/relation_update_spine_v1/CMakeLists.txt)
endif()
add_subdirectory(src/execution/program)
include(cmake/Product2.cmake)
# Product2 is host-capable and must be available to the Python module before
# the CUDA-off early return. Mechanism bindings are attached after their CUDA
# owner targets have been declared below.
if(CELLERATOR_ENABLE_PYTHON)
    include(cmake/CelleratorPython.cmake)
endif()
# ATen numerical operations and tokenization have a separate optional
# dependency boundary from the CUDA mechanism Torch adapters below.
option(CELLERATOR_ENABLE_ANNOTATION "Build the native reference annotation component" ${CELLERATOR_ENABLE_TOKENIZATION})
if(CELLERATOR_ENABLE_ANNOTATION OR CELLERATOR_ENABLE_TOKENIZATION)
    add_subdirectory(components/annotation)
endif()
include(cmake/CelleratorTokenization.cmake)
include(cmake/CelleratorSparseSupport.cmake)
# Native data and geometry owners supply the direct host operation targets.
include(cmake/CelleratorNativeData.cmake)
if(CELLERATOR_BUILD_TESTS)
    enable_testing()
endif()
add_subdirectory(tests/math_reference ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL})
add_subdirectory(tests/sequence ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL})
add_subdirectory(bench/sequence ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL})
if(CELLERATOR_ENABLE_CUDA STREQUAL "OFF")
    cellerator_append_native_data_package_config()
endif()
add_subdirectory(tests/framework/native ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL})
if(CELLERATOR_BUILD_NATIVE_DATA_TESTS)
    enable_testing()
endif()
cellerator_add_native_data_tests()
include(cmake/CelleratorConditional.cmake)
include(cmake/CelleratorGGML.cmake)
# Native data CUDA realizations are layered over the host owners below.
include(cmake/CelleratorFrameworkCUDA.cmake)
# With framework CUDA disabled, the root CUDA configuration still exposes the
# independent native-data CUDA targets. The module is include-guarded because
# FrameworkCUDA also includes it when that optional SDK is enabled.
include(cmake/CelleratorNativeDataCUDA.cmake)
add_subdirectory(src/execution/streaming)
include(cmake/CelleratorTorchSparse.cmake)
include(cmake/CelleratorTorchDense.cmake)
include(cmake/CelleratorNN.cmake)

macro(cellerator_append_component_validation)
    cellerator_append_optional_component_config()
    if(DEFINED CELLERATOR_HOST_CONFIG AND EXISTS "${CELLERATOR_HOST_CONFIG}")
        file(APPEND "${CELLERATOR_HOST_CONFIG}"
            "foreach(_cellerator_component IN LISTS Cellerator_FIND_COMPONENTS)\n"
            "  if(Cellerator_FIND_REQUIRED_\${_cellerator_component} AND NOT Cellerator_\${_cellerator_component}_FOUND)\n"
            "    set(Cellerator_FOUND FALSE)\n"
            "    set(Cellerator_NOT_FOUND_MESSAGE \"Required Cellerator component '\${_cellerator_component}' is unavailable\")\n"
            "  endif()\n"
            "endforeach()\n")
    endif()
endmacro()

# Native arithmetic families are host-capable and independent of adapters.
add_subdirectory(src/ops/numeric)
add_subdirectory(src/ops/matrix)
add_subdirectory(src/ops/process)
add_subdirectory(src/structure)
if(CELLERATOR_ENABLE_ANNDATA)
    add_subdirectory(components/anndata)
endif()
if(CELLERATOR_ENABLE_RDS)
    add_subdirectory(components/rds)
endif()
if(CELLERATOR_ENABLE_VELOCITY OR CELLERATOR_ENABLE_PREPROCESS)
    include(cmake/CelleratorModuleInterfaces.cmake)
    cellerator_require_module_scanner()
    option(CELLERATOR_ENABLE_METHOD_SANITIZERS "Instrument native methods and their build-tree consumers" OFF)
    option(CELLERATOR_BUILD_METHOD_QUALIFICATION "Build native method qualification drivers" OFF)
endif()
if(CELLERATOR_ENABLE_VELOCITY)
    add_subdirectory(components/velocity)
endif()
if(CELLERATOR_ENABLE_PREPROCESS)
    add_subdirectory(components/preprocess)
endif()
if(CELLERATOR_BUILD_TESTS AND CELLERATOR_ENABLE_PREPROCESS AND CELLERATOR_ENABLE_VELOCITY)
    add_subdirectory(tests/single_cell)
endif()
# Build the linked NF1 numerical foundation and its host reference consumer.
# CUDA relation/value tests are enabled only with the semantic spine owner.
option(CELLERATOR_BUILD_NATIVE_FOUNDATION_TESTS "Build NF1 numerical foundation tests" OFF)
if(CELLERATOR_ENABLE_CUDA STREQUAL "OFF")
    if(CELLERATOR_ENABLE_TORCH)
        message(FATAL_ERROR
            "CELLERATOR_ENABLE_TORCH requires the CUDA indexed-mechanism owner; "
            "configure with CELLERATOR_ENABLE_CUDA=AUTO or ON.")
    endif()
    if(COMMAND cellerator_append_native_data_cuda_package_config)
        cellerator_append_native_data_cuda_package_config()
    endif()
    if(COMMAND cellerator_append_framework_cuda_package_config)
        cellerator_append_framework_cuda_package_config()
    endif()
    if(TARGET Cellerator::torch_sparse)
        cellerator_append_torch_sparse_package_config()
    endif()
    if(TARGET Cellerator::torch_dense)
        cellerator_append_torch_dense_package_config()
    endif()
    cellerator_append_component_validation()
    return()
endif()
# Provider inclusion and tuning are explicit build inputs. The
# provider policy module deliberately does not inspect a visible device or
# mutate CMAKE_CUDA_ARCHITECTURES; binary content remains reproducible from
# the cache configuration alone.
include(cmake/providers/CelleratorProviderPolicy.cmake)

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
set_target_properties(cellerator_sequence cellerator_sequence_cuda_test
    cellerator_sequence_cuda_bench PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

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

# Canonical computational subsystem owners.
include(src/runtime/CMakeLists.txt)
include(src/compute/decomposition/CMakeLists.txt)
include(src/execution/CMakeLists.txt)
include(src/compute/foundation_targets.cmake)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

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
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    set_target_properties(${target} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE YES
    )
endfunction()

cellerator_enable_perf(cellerator_runtime)
cellerator_enable_perf(cellerator_compute_matrix_convert)
cellerator_enable_perf(cellerator_runtime_fleet)
cellerator_enable_perf(cellerator_csr_multiply)

include(src/CMakeLists.txt)
add_subdirectory(bench/execution/streaming)
add_subdirectory(components/neighbors)
# General numerical operations used by both native C++ and optional framework
# adapters remain in Cellerator even when the Torch integration is disabled.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/compute/operation/model_ops/CMakeLists.txt")
    add_subdirectory(src/compute/operation/model_ops)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/bindings/model_ops/model_ops_dimension_contract_test.cc")
        add_executable(celleratorModelOpsDimensionContractTest ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
            tests/bindings/model_ops/model_ops_dimension_contract_test.cc)
        target_link_libraries(celleratorModelOpsDimensionContractTest PRIVATE
            Cellerator::model_ops)
        if(CELLERATOR_BUILD_TESTS)
            add_test(NAME cellerator_model_ops_dimension_contract
                COMMAND celleratorModelOpsDimensionContractTest)
        endif()
    endif()
endif()
# The legacy compute manifest declares the provider source inventories.  Split
# them into provider-owned targets after all of their shared dependencies have
# been declared, then emit the cold compiled-provider manifest.
include(cmake/providers/CelleratorProviderTargets.cmake)
cellerator_finalize_cuda_provider_targets()
include(src/geometry/owned_targets.cmake)
include(tests/execution/fixtures.cmake)

# Complete geometry and provider registrations before building their consumers.
target_sources(cellerator_semantic_geometry_v1 PRIVATE
    src/geometry/planning/v2/exact_evaluator.cc
    src/geometry/planning/v2/solution.cc
    src/geometry/planning/v2/strategy_registry.cc
    src/geometry/planning/v2/work_window.cc
    src/geometry/planning/v2/workload_profile.cc
    src/geometry/persistence/chunk/chunk_manifest_v1.cc
    src/geometry/validation/scalable_validation_v2.cc
)
target_sources(cellerator_architecture_provider PRIVATE
    src/compute/architecture/target_cover/contract.cc
    src/compute/architecture/target_cover/strategy_registry.cc
)
add_library(cellerator_profiling_v1 STATIC
    src/profiling/mechanism_manifest.cc
    src/profiling/preparation/execution_export_v2.cc
    src/profiling/partition_export.cc
    src/profiling/resource_receipt.cc
    src/profiling/static_markers.cc
)
add_library(Cellerator::profiling_v1 ALIAS cellerator_profiling_v1)
target_include_directories(cellerator_profiling_v1 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_profiling_v1 PUBLIC cxx_std_20)

if(TARGET cellerator_dependency_execution_cuda)
    target_link_libraries(cellerator_dependency_execution_cuda PUBLIC
        Cellerator::execution_preparation Cellerator::gated_product)
endif()

block()
    add_executable(cellerator_operation_core_v2_test
        ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
        src/compute/operation/operation_core_v2/operation_core_v2_test.cc)
    target_link_libraries(cellerator_operation_core_v2_test PRIVATE
        Cellerator::operation_core)
    target_compile_features(cellerator_operation_core_v2_test PRIVATE cxx_std_20)

    add_executable(cellerator_geometry_artifacts_test
        ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
        tests/execution/geometry_artifacts_test.cc)
    target_link_libraries(cellerator_geometry_artifacts_test PRIVATE
        Cellerator::execution_geometry_acquisition)
    target_compile_features(cellerator_geometry_artifacts_test PRIVATE cxx_std_20)

    add_executable(cellerator_geometry_v2_test
        ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
        src/geometry/planning/v2/geometry_v2_test.cc)
    target_link_libraries(cellerator_geometry_v2_test PRIVATE
        Cellerator::architecture_provider
        Cellerator::semantic_geometry_v1)
    target_compile_features(cellerator_geometry_v2_test PRIVATE cxx_std_20)

    add_executable(cellerator_planner_portfolio_v1_test
        ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
        tests/execution/planning/candidate_set_test.cc)
    target_link_libraries(cellerator_planner_portfolio_v1_test PRIVATE
        Cellerator::execution_planning)
    target_compile_features(cellerator_planner_portfolio_v1_test PRIVATE cxx_std_20)

    # JBC standalone ABI gate. This target intentionally consumes only
    # Cellerator public headers and the Cellerator-owned profiling library;
    # it must not gain a CellShard source, include, or link dependency.
    add_executable(cellerator_execution_contract_standalone_abi_gate
        ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
        tests/jbc/verification/standalone_abi_gate_v1_test.cc)
    target_link_libraries(cellerator_execution_contract_standalone_abi_gate PRIVATE
        Cellerator::profiling_v1)
    target_compile_features(cellerator_execution_contract_standalone_abi_gate PRIVATE cxx_std_20)

    file(GLOB_RECURSE CELLERATOR_EXECUTION_PREPARATION_TEST_SOURCES CONFIGURE_DEPENDS
        tests/jbc/atom_plane/*.cc
        tests/jbc/cross_operation/*.cc
        tests/jbc/decomposition/*.cc
        tests/jbc/external_cost/*.cc
        tests/jbc/fragment/*.cc
        tests/jbc/interfaces/*.cc
        tests/jbc/multi_extent/*.cc
        tests/jbc/resumption/*.cc
        tests/jbc/verification/atom_fragment_verifier_v1_test.cc
        tests/jbc/verification/numerical_verifier_v1_test.cc
        tests/jbc/verification/profiler_manifest_v1_test.cc
    )
    add_custom_target(cellerator_execution_preparation_tests)
    foreach(cellerator_execution_contract_test_source IN LISTS CELLERATOR_EXECUTION_PREPARATION_TEST_SOURCES)
        file(RELATIVE_PATH cellerator_execution_contract_test_relative
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/jbc"
            "${cellerator_execution_contract_test_source}")
        string(REPLACE "/" "_" cellerator_execution_contract_test_name
            "${cellerator_execution_contract_test_relative}")
        string(REGEX REPLACE "\\.cc$" "" cellerator_execution_contract_test_name
            "${cellerator_execution_contract_test_name}")
        set(cellerator_execution_contract_test_target
            "cellerator_execution_contract_${cellerator_execution_contract_test_name}")
        add_executable(${cellerator_execution_contract_test_target}
            ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
            "${cellerator_execution_contract_test_source}")
        target_link_libraries(${cellerator_execution_contract_test_target} PRIVATE
            Cellerator::execution_preparation)
        target_compile_features(${cellerator_execution_contract_test_target} PRIVATE cxx_std_20)
        add_dependencies(cellerator_execution_preparation_tests ${cellerator_execution_contract_test_target})
    endforeach()

    add_executable(cellerator_execution_contract_multi_extent_assembly_bench
        ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
        bench/jbc/multi_extent/assembly_vs_direct_v1.cc)
    target_link_libraries(cellerator_execution_contract_multi_extent_assembly_bench PRIVATE
        Cellerator::execution_preparation)
    target_compile_features(cellerator_execution_contract_multi_extent_assembly_bench PRIVATE
        cxx_std_20)
endblock()

cellerator_include_checks(bench/CMakeLists.txt)
cellerator_include_checks(tests/CMakeLists.txt)
block()
    # CE-EXOP final acceptance entry points. These targets keep correctness,
    # profiler-fixture validation, and CUDA compile validation explicit. The
    # CUDA target compiles sm_70 binaries only; device execution remains under
    # Project Control resource authority.
    add_custom_target(cellerator_operation_host_acceptance
        COMMAND ${CMAKE_COMMAND} -E env
            CXX=${CMAKE_CXX_COMPILER}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/operation_execution/host/run_host_acceptance.sh
            ${CMAKE_CURRENT_SOURCE_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        USES_TERMINAL)

    add_custom_target(cellerator_operation_profiler_acceptance
        COMMAND ${CMAKE_COMMAND} -E env
            CXX=${CMAKE_CXX_COMPILER}
            CUDA_ROOT=${CUDAToolkit_ROOT}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/profiling/operations/run_fixture_validation.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        USES_TERMINAL)

    add_custom_target(cellerator_operation_cuda_compile_acceptance
        COMMAND ${CMAKE_COMMAND} -E env
            PATH=${CUDAToolkit_BIN_DIR}:$ENV{PATH}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/operation_execution/cuda/run_cuda_validation.sh
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/ce-exop-cuda-validation
            --build-only
            all
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        USES_TERMINAL)
endblock()

include(cmake/NativeFoundation.cmake)

# NativeData exports depend on host owners exported by NativeFoundation in a
# CUDA build. Append these package includes only after both export files have
# been added so installed imported targets exist before their consumers.
cellerator_append_native_data_package_config()
if(COMMAND cellerator_append_native_data_cuda_package_config)
    cellerator_append_native_data_cuda_package_config()
endif()
if(COMMAND cellerator_append_framework_cuda_package_config)
    cellerator_append_framework_cuda_package_config()
endif()
if(TARGET Cellerator::torch_sparse)
    cellerator_append_torch_sparse_package_config()
endif()
if(TARGET Cellerator::torch_dense)
    cellerator_append_torch_dense_package_config()
endif()

if(CELLERATOR_ENABLE_SPARSE_SUPPORT)
    add_subdirectory(tests/sparse_support ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL})
endif()
if(CELLERATOR_ENABLE_SPARSE_SUPPORT AND EXISTS "${PROJECT_SOURCE_DIR}/bench/sparse_support/CMakeLists.txt")
    add_subdirectory(bench/sparse_support ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL})
endif()

add_executable(celleratorCudaStreamDeviceTest ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    tests/bindings/native/cuda_stream_device_test.cc)
target_include_directories(celleratorCudaStreamDeviceTest PRIVATE
    "${PROJECT_SOURCE_DIR}/src/runtime")
target_link_libraries(celleratorCudaStreamDeviceTest PRIVATE
    CUDA::cudart CUDA::cuda_driver)
target_compile_features(celleratorCudaStreamDeviceTest PRIVATE cxx_std_20)
if(CELLERATOR_BUILD_TESTS)
    add_test(NAME cellerator_cuda_stream_device
        COMMAND celleratorCudaStreamDeviceTest)
    set_tests_properties(cellerator_cuda_stream_device PROPERTIES SKIP_RETURN_CODE 77)
endif()

if(CELLERATOR_ENABLE_PYTHON)
    cellerator_configure_python_mechanism_bindings()
endif()

if(CELLERATOR_ENABLE_TORCH)
    add_subdirectory(src/interop/torch)
endif()

cellerator_append_component_validation()

# Keep the native factorized-interaction component independently installable.
# Its narrow package path loads only the shared structural base, CUDA memory,
# and factorized executor exports.  The conventional package path retains the
# complete existing Cellerator config, which is needed for every other request.
if(TARGET cellerator_factorized_interaction_cuda
   AND DEFINED CELLERATOR_HOST_CONFIG
   AND EXISTS "${CELLERATOR_HOST_CONFIG}")
    set(_cellerator_full_config
        "${CMAKE_CURRENT_BINARY_DIR}/CelleratorFullConfig.cmake")
    configure_file("${CELLERATOR_HOST_CONFIG}" "${_cellerator_full_config}" COPYONLY)
    file(WRITE "${CELLERATOR_HOST_CONFIG}"
        "set(_cellerator_factorized_only_request FALSE)\n"
        "list(LENGTH Cellerator_FIND_COMPONENTS _cellerator_requested_component_count)\n"
        "if(_cellerator_requested_component_count EQUAL 1 AND Cellerator_FIND_COMPONENTS STREQUAL \"factorized_interaction_cuda\")\n"
        "  set(_cellerator_factorized_only_request TRUE)\n"
        "endif()\n"
        "if(_cellerator_factorized_only_request OR NOT EXISTS \"\${CMAKE_CURRENT_LIST_DIR}/CelleratorFullConfig.cmake\")\n"
        "  if(NOT \"factorized_interaction_cuda\" IN_LIST Cellerator_FIND_COMPONENTS)\n"
        "    set(Cellerator_FOUND FALSE)\n"
        "    set(Cellerator_NOT_FOUND_MESSAGE \"This component-only installation provides only factorized_interaction_cuda\")\n"
        "    return()\n"
        "  endif()\n"
        "  include(CMakeFindDependencyMacro)\n"
        "  find_dependency(CUDAToolkit REQUIRED)\n"
        "  include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorFactorizedInteractionBaseTargets.cmake\")\n"
        "  include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorFactorizedInteractionMemoryTargets.cmake\")\n"
        "  include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorFactorizedInteractionTargets.cmake\")\n"
        "  set(Cellerator_factorized_interaction_cuda_FOUND TRUE)\n"
        "  foreach(_cellerator_requested_component IN LISTS Cellerator_FIND_COMPONENTS)\n"
        "    if(NOT _cellerator_requested_component STREQUAL \"factorized_interaction_cuda\")\n"
        "      set(Cellerator_\${_cellerator_requested_component}_FOUND FALSE)\n"
        "    endif()\n"
        "  endforeach()\n"
        "  foreach(_cellerator_requested_component IN LISTS Cellerator_FIND_COMPONENTS)\n"
        "    if(Cellerator_FIND_REQUIRED_\${_cellerator_requested_component} AND NOT Cellerator_\${_cellerator_requested_component}_FOUND)\n"
        "      set(Cellerator_FOUND FALSE)\n"
        "      set(Cellerator_NOT_FOUND_MESSAGE \"Required Cellerator component '\${_cellerator_requested_component}' is unavailable\")\n"
        "    endif()\n"
        "  endforeach()\n"
        "elseif(EXISTS \"\${CMAKE_CURRENT_LIST_DIR}/CelleratorFullConfig.cmake\")\n"
        "  include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorFullConfig.cmake\")\n"
        "else()\n"
        "  set(Cellerator_FOUND FALSE)\n"
        "  set(Cellerator_NOT_FOUND_MESSAGE \"This component-only installation provides only factorized_interaction_cuda\")\n"
        "endif()\n")
    install(FILES "${_cellerator_full_config}"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator")
    install(FILES "${CELLERATOR_HOST_CONFIG}"
                  "${CMAKE_CURRENT_BINARY_DIR}/CelleratorConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator"
        COMPONENT FactorizedInteractionCUDA)
endif()
