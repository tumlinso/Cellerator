# Native biological data contracts have a host-only dependency boundary.
# Their CUDA realizations are optional and live in the separate
# CelleratorNativeDataCUDA.cmake owner.
include_guard(GLOBAL)
include(GNUInstallDirs)

function(cellerator_configure_native_data_host_target target export_name)
    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME "${export_name}")
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
endfunction()

function(cellerator_add_native_data_memory_cuda_owner)
    if(TARGET cellerator_memory_cuda)
        return()
    endif()
    add_library(cellerator_memory_cuda STATIC
        "${PROJECT_SOURCE_DIR}/src/memory/cuda/resources.cu")
    add_library(Cellerator::memory_cuda ALIAS cellerator_memory_cuda)
    set_target_properties(cellerator_memory_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME memory_cuda
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_features(cellerator_memory_cuda PUBLIC cxx_std_20)
    target_include_directories(cellerator_memory_cuda PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_link_libraries(cellerator_memory_cuda PUBLIC CUDA::cudart)
    # Keep the one canonical memory target in a small export set.  Both the
    # full SDK and the factorized-interaction component import this file, so
    # their target-export sets never define memory_cuda twice.
    install(TARGETS cellerator_memory_cuda
        EXPORT CelleratorFactorizedInteractionMemoryTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT FactorizedInteractionCUDA)
    install(EXPORT CelleratorFactorizedInteractionMemoryTargets
        NAMESPACE Cellerator::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator
        COMPONENT FactorizedInteractionCUDA)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/Cellerator/memory/cuda"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/memory"
        COMPONENT NativeDataCUDA)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/Cellerator/memory/cuda"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/memory"
        COMPONENT FactorizedInteractionCUDA)
    if(DEFINED CELLERATOR_HOST_CONFIG AND EXISTS "${CELLERATOR_HOST_CONFIG}")
        file(APPEND "${CELLERATOR_HOST_CONFIG}"
            "include(CMakeFindDependencyMacro)\n"
            "find_dependency(CUDAToolkit)\n"
            "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorFactorizedInteractionMemoryTargets.cmake\")\n")
    endif()
endfunction()

add_library(cellerator_dependency_geometry STATIC
    "${PROJECT_SOURCE_DIR}/src/geometry/dependency/validation.cc"
    "${PROJECT_SOURCE_DIR}/src/geometry/dependency/cartesian_access.cc"
    "${PROJECT_SOURCE_DIR}/src/geometry/dependency/explicit_pattern.cc"
    "${PROJECT_SOURCE_DIR}/src/geometry/dependency/explicit_incidence.cc"
    "${PROJECT_SOURCE_DIR}/src/geometry/dependency/patterns.cc"
    "${PROJECT_SOURCE_DIR}/src/geometry/dependency/pattern_owner.cc"
    "${PROJECT_SOURCE_DIR}/src/ops/mechanism/access.cc")
target_compile_features(cellerator_dependency_geometry PUBLIC cxx_std_20)
cellerator_configure_native_data_host_target(
    cellerator_dependency_geometry dependency_geometry)
add_library(Cellerator::dependency_geometry ALIAS cellerator_dependency_geometry)
if(NOT TARGET Cellerator::support OR NOT TARGET Cellerator::indexing)
    message(FATAL_ERROR
        "Native dependency geometry requires CELLERATOR_ENABLE_SPARSE_SUPPORT=ON")
endif()
target_link_libraries(cellerator_dependency_geometry PUBLIC
    Cellerator::support Cellerator::indexing)

add_library(cellerator_gated_product STATIC
    "${PROJECT_SOURCE_DIR}/src/ops/mechanism/gated_product_host.cc"
    "${PROJECT_SOURCE_DIR}/src/ops/mechanism/gated_product_binding.cc")
add_library(Cellerator::gated_product ALIAS cellerator_gated_product)
target_compile_features(cellerator_gated_product PUBLIC cxx_std_20)
cellerator_configure_native_data_host_target(cellerator_gated_product gated_product)
target_link_libraries(cellerator_gated_product PUBLIC
    Cellerator::dependency_geometry)

add_library(cellerator_block_masked STATIC
    "${PROJECT_SOURCE_DIR}/src/tensor/block_masked/structure.cc")
add_library(Cellerator::block_masked ALIAS cellerator_block_masked)
target_compile_features(cellerator_block_masked PUBLIC cxx_std_20)
cellerator_configure_native_data_host_target(cellerator_block_masked block_masked)

add_library(cellerator_state_codes INTERFACE)
add_library(Cellerator::state_codes ALIAS cellerator_state_codes)
set_target_properties(cellerator_state_codes PROPERTIES EXPORT_NAME state_codes)
target_include_directories(cellerator_state_codes INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(cellerator_state_codes INTERFACE cxx_std_20)

add_library(cellerator_tensor INTERFACE)
add_library(Cellerator::tensor ALIAS cellerator_tensor)
set_target_properties(cellerator_tensor PROPERTIES EXPORT_NAME tensor)
target_include_directories(cellerator_tensor INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(cellerator_tensor INTERFACE cxx_std_20)
target_link_libraries(cellerator_tensor INTERFACE
    Cellerator::block_masked Cellerator::state_codes)
if(CELLERATOR_ENABLE_MODULES)
    include("${PROJECT_SOURCE_DIR}/src/tensor/tensor_module.cmake")
endif()

add_library(cellerator_dependency_execution STATIC
    "${PROJECT_SOURCE_DIR}/src/execution/dependency/distributed.cc"
    "${PROJECT_SOURCE_DIR}/src/execution/dependency/workset.cc"
    "${PROJECT_SOURCE_DIR}/src/execution/dependency/incidence.cc"
    "${PROJECT_SOURCE_DIR}/src/execution/dependency/read_index.cc"
    "${PROJECT_SOURCE_DIR}/src/execution/dependency/condition_index.cc")
add_library(Cellerator::dependency_execution ALIAS cellerator_dependency_execution)
target_compile_features(cellerator_dependency_execution PUBLIC cxx_std_20)
cellerator_configure_native_data_host_target(
    cellerator_dependency_execution dependency_execution)
target_link_libraries(cellerator_dependency_execution PUBLIC
    Cellerator::dependency_geometry Cellerator::layout Cellerator::gating)

set(_cellerator_host_operation_targets)
foreach(_operation IN ITEMS reaction exchange accessibility inheritance event_memory)
    add_library(cellerator_${_operation} INTERFACE)
    add_library(Cellerator::${_operation} ALIAS cellerator_${_operation})
    set_target_properties(cellerator_${_operation} PROPERTIES EXPORT_NAME ${_operation})
    target_include_directories(cellerator_${_operation} INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_compile_features(cellerator_${_operation} INTERFACE cxx_std_20)
    list(APPEND _cellerator_host_operation_targets cellerator_${_operation})
endforeach()
target_link_libraries(cellerator_exchange INTERFACE Cellerator::layout)
target_link_libraries(cellerator_event_memory INTERFACE Cellerator::layout)

add_library(cellerator_native_data_api INTERFACE)
add_library(Cellerator::native_data ALIAS cellerator_native_data_api)
set_target_properties(cellerator_native_data_api PROPERTIES EXPORT_NAME native_data)
target_include_directories(cellerator_native_data_api INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(cellerator_native_data_api INTERFACE cxx_std_20)
target_link_libraries(cellerator_native_data_api INTERFACE
    Cellerator::dependency_geometry
    Cellerator::gated_product
    Cellerator::tensor
    Cellerator::dependency_execution)

if(CELLERATOR_ENABLE_PYTHON)
    pybind11_add_module(cellerator_modeling_python MODULE
        "${PROJECT_SOURCE_DIR}/python/bindings/modeling/module.cc")
    target_link_libraries(cellerator_modeling_python PRIVATE
        Cellerator::dependency_execution Python::Module)
    set_target_properties(cellerator_modeling_python PROPERTIES OUTPUT_NAME _modeling_native)
    install(TARGETS cellerator_modeling_python
        LIBRARY DESTINATION cellerator COMPONENT Python
        RUNTIME DESTINATION cellerator COMPONENT Python)
    if(TARGET cellerator_python_extensions)
        add_dependencies(cellerator_python_extensions cellerator_modeling_python)
    endif()
endif()

# Structural preparation is a host operation and remains testable without a
# CUDA toolchain. The checked ExplicitPattern bridge separately exercises its
# geometry-owner boundary.
option(CELLERATOR_BUILD_FACTORIZED_INTERACTION_TESTS
    "Build host structural tests for the factorized interaction layout" OFF)
if(CELLERATOR_BUILD_FACTORIZED_INTERACTION_TESTS)
    enable_testing()
    add_executable(cellerator_factorized_interaction_preparation_test ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
        "${PROJECT_SOURCE_DIR}/tests/factorized_interactions_preparation/preparation_test.cc")
    target_link_libraries(cellerator_factorized_interaction_preparation_test PRIVATE
        Cellerator::layout)
    target_compile_features(cellerator_factorized_interaction_preparation_test PRIVATE cxx_std_20)
    add_test(NAME factorized_interaction_preparation
        COMMAND cellerator_factorized_interaction_preparation_test)
    set_tests_properties(factorized_interaction_preparation PROPERTIES
        LABELS "factorized_interaction;host_only" TIMEOUT 60)

    add_executable(cellerator_factorized_interaction_pattern_test ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
        "${PROJECT_SOURCE_DIR}/tests/factorized_interactions_preparation/pattern_binding_test.cc")
    target_link_libraries(cellerator_factorized_interaction_pattern_test PRIVATE
        Cellerator::layout Cellerator::dependency_geometry)
    target_compile_features(cellerator_factorized_interaction_pattern_test PRIVATE cxx_std_20)
    add_test(NAME factorized_interaction_pattern_binding
        COMMAND cellerator_factorized_interaction_pattern_test)
    set_tests_properties(factorized_interaction_pattern_binding PROPERTIES
        LABELS "factorized_interaction;host_only" TIMEOUT 60)
endif()

install(TARGETS
        cellerator_dependency_geometry
        cellerator_gated_product
        cellerator_block_masked
        cellerator_dependency_execution
    EXPORT CelleratorNativeDataTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cellerator_state_codes cellerator_tensor ${_cellerator_host_operation_targets}
    EXPORT CelleratorNativeDataTargets)
install(TARGETS cellerator_native_data_api EXPORT CelleratorNativeDataTargets)
install(EXPORT CelleratorNativeDataTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)

# The full host SDK already installs the complete public include tree. These
# component-scoped entries also let a consumer request just NativeData.
foreach(_header_subtree IN ITEMS
        structure
        geometry/dependency
        layout)
    get_filename_component(_header_parent "${_header_subtree}" DIRECTORY)
    if(NOT _header_parent OR _header_parent STREQUAL ".")
        set(_header_destination "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator")
    else()
        set(_header_destination
            "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/${_header_parent}")
    endif()
    install(DIRECTORY
        "${PROJECT_SOURCE_DIR}/include/Cellerator/${_header_subtree}"
        DESTINATION "${_header_destination}"
        COMPONENT NativeData)
endforeach()
foreach(_header IN ITEMS
        common/compiler_annotations.hh
        ops/mechanism/access.hh
        ops/mechanism/gated_product.hh
        ops/mechanism/gated_product_binding.hh
        ops/mechanism/gated_product_host.hh
        ops/mechanism/omission.hh
        layout/cell_packing_geometry.hh
        execution/cell_packing_binding.hh
        tensor/block_masked/layout.hh
        tensor/block_masked/view.hh
        tensor/block_masked/row_view.hh
        tensor/block_masked/structure.hh
        tensor/base/compressed.hh
        tensor/base/coordinate.hh
        tensor/feature_major_tiles.hh
        tensor/state_codes/view.hh
        execution/dependency/distributed.hh
        execution/dependency/condition_index.hh
        execution/dependency/feature_binding.hh
        execution/dependency/support_lifecycle.hh
        execution/dependency/workset_view.hh
        execution/dependency/incidence.hh
        execution/dependency/incremental.hh
        execution/dependency/read_index.hh
        execution/dependency/workset.hh
        execution/dependency/value_field.hh
        containers/value_shape.hh
        containers/posting_index.hh
        memory/domain.hh
        ops/reaction/network.hh
        ops/process/exchange.hh
        ops/process/inheritance.hh
        ops/condition/accessibility.hh
        ops/condition/program.hh
        ops/sequence/event_memory.hh
        interop/formats/affine_blocked_ell.hh
        interop/formats/affine_blocked_ell_view.hh
        tensor/quantized/csr.hh
        tensor/quantized/blocked_ell.hh)
    get_filename_component(_header_parent "${_header}" DIRECTORY)
    install(FILES "${PROJECT_SOURCE_DIR}/include/Cellerator/${_header}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/${_header_parent}"
        COMPONENT NativeData)
endforeach()
function(cellerator_append_native_data_package_config)
    if(DEFINED CELLERATOR_HOST_CONFIG AND EXISTS "${CELLERATOR_HOST_CONFIG}")
        file(APPEND "${CELLERATOR_HOST_CONFIG}"
            "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorNativeDataTargets.cmake\")\n"
            "set(Cellerator_native_data_FOUND TRUE)\n")
        if(CELLERATOR_ENABLE_MODULES)
            file(APPEND "${CELLERATOR_HOST_CONFIG}"
                "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorTensorInterfaceTargets.cmake\")\n"
                "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorModuleInterfaces.cmake\")\n"
                "cellerator_attach_module(Cellerator::tensor Cellerator::tensor_interface_binary)\n")
        endif()
    endif()
endfunction()
install(FILES "${PROJECT_SOURCE_DIR}/include/Cellerator/ops/incidence.hh"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/ops"
    COMPONENT NativeData)

function(cellerator_add_native_data_tests)
    if(NOT CELLERATOR_BUILD_NATIVE_DATA_TESTS)
        return()
    endif()
    add_subdirectory("${PROJECT_SOURCE_DIR}/tests/native_data"
        "${PROJECT_BINARY_DIR}/tests/native_data" ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL})
    if(TARGET cellerator_native_data_tests)
        add_dependencies(cellerator_native_data cellerator_native_data_tests)
    endif()
endfunction()

# This explicit build entry point compiles only the native data owners and
# their focused host tests, without pulling the legacy acceptance suite.
add_custom_target(cellerator_native_data
    DEPENDS
        cellerator_dependency_geometry
        cellerator_gated_product
        cellerator_block_masked
        cellerator_dependency_execution)
