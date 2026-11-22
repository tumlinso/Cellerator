# Independent opt-in CUDA numerical runtime. Does not require Torch, the legacy
# compiler runtime, or Python. May also be included by a focused CMake harness.
include(GNUInstallDirs)
option(CELLERATOR_ENABLE_FRAMEWORK_CUDA "Build framework CUDA resources and kernels" OFF)
option(CELLERATOR_ENABLE_FRAMEWORK_CUDA_PYTHON "Build optional framework CUDA Python binding" OFF)
option(CELLERATOR_BUILD_FRAMEWORK_CUDA_SMOKE "Build (do not automatically run) CUDA smoke executables" OFF)
if(NOT CELLERATOR_ENABLE_FRAMEWORK_CUDA)
    if(CELLERATOR_ENABLE_FRAMEWORK_CUDA_PYTHON)
        message(FATAL_ERROR "FRAMEWORK_CUDA_PYTHON requires CELLERATOR_ENABLE_FRAMEWORK_CUDA")
    endif()
    return()
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
if(NOT COMMAND cellerator_add_native_data_memory_cuda_owner)
    message(FATAL_ERROR
        "CelleratorNativeData.cmake must be included before FrameworkCUDA")
endif()
cellerator_add_native_data_memory_cuda_owner()
include("${CMAKE_CURRENT_LIST_DIR}/CelleratorNativeDataCUDA.cmake")
get_filename_component(_cellerator_framework_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

function(cellerator_add_framework_cuda_provider name source header)
    set(target "cellerator_${name}")
    add_library(${target} STATIC "${_cellerator_framework_root}/src/${source}")
    add_library(Cellerator::${name} ALIAS ${target})
    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON EXPORT_NAME ${name}
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_features(${target} PUBLIC cxx_std_20)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${_cellerator_framework_root}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_link_libraries(${target} PUBLIC CUDA::cudart ${ARGN})
    install(TARGETS ${target} EXPORT CelleratorFrameworkCUDATargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    get_filename_component(header_directory "${header}" DIRECTORY)
    install(FILES "${_cellerator_framework_root}/include/Cellerator/${header}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/${header_directory}")
endfunction()

cellerator_add_framework_cuda_provider(dispatch_cuda
    ops/routing/dispatch.cu ops/routing/dispatch_cuda.hh)
cellerator_add_framework_cuda_provider(conditional_dispatch_cuda
    execution/conditional/dispatch.cu execution/conditional/dispatch_cuda.hh
    Cellerator::dispatch_cuda Cellerator::memory_cuda)
cellerator_add_framework_cuda_provider(event_memory_cuda
    ops/sequence/event_memory.cu ops/sequence/event_memory_cuda.hh)
cellerator_add_framework_cuda_provider(dependency_frontier_cuda
    execution/dependency/frontier.cu execution/dependency/frontier_cuda.hh)
cellerator_add_framework_cuda_provider(attention_cuda
    ops/attention/attention.cu ops/attention/sparse_cuda.hh)
cellerator_add_framework_cuda_provider(grouped_linear_cuda
    ops/matrix/grouped_linear.cu ops/matrix/grouped_linear_cuda.hh
    Cellerator::memory_cuda CUDA::cublas)
cellerator_add_framework_cuda_provider(packed_embedding_cuda
    ops/sequence/packed_embedding.cu ops/sequence/packed_embedding_cuda.hh
    Cellerator::memory_cuda Cellerator::state_codes)
cellerator_add_framework_cuda_provider(reaction_cuda
    ops/reaction/reactions.cu ops/reaction/apply_cuda.hh Cellerator::memory_cuda)
cellerator_add_framework_cuda_provider(regulatory_gates_cuda
    ops/condition/regulatory_gates.cu ops/condition/regulatory_gates_cuda.hh)

add_library(cellerator_framework_cuda_api INTERFACE)
add_library(Cellerator::framework_cuda ALIAS cellerator_framework_cuda_api)
set_target_properties(cellerator_framework_cuda_api PROPERTIES EXPORT_NAME framework_cuda)
target_link_libraries(cellerator_framework_cuda_api INTERFACE
    Cellerator::conditional_dispatch_cuda
    Cellerator::event_memory_cuda
    Cellerator::dependency_frontier_cuda
    Cellerator::attention_cuda
    Cellerator::grouped_linear_cuda
    Cellerator::packed_embedding_cuda
    Cellerator::reaction_cuda
    Cellerator::regulatory_gates_cuda
    Cellerator::memory_cuda
    Cellerator::state_codes_cuda)
add_custom_target(cellerator_framework_cuda
    DEPENDS
        cellerator_conditional_dispatch_cuda
        cellerator_event_memory_cuda
        cellerator_dependency_frontier_cuda
        cellerator_attention_cuda
        cellerator_grouped_linear_cuda
        cellerator_packed_embedding_cuda
        cellerator_reaction_cuda
        cellerator_regulatory_gates_cuda
        cellerator_state_codes_cuda)

if(CELLERATOR_ENABLE_FRAMEWORK_CUDA_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(cellerator_modeling_cuda_python MODULE
        "${_cellerator_framework_root}/python/bindings/modeling_cuda/module.cc")
    target_link_libraries(cellerator_modeling_cuda_python PRIVATE
        Cellerator::conditional_dispatch_cuda Cellerator::grouped_linear_cuda
        Cellerator::state_codes_cuda Python::Module)
    set_target_properties(cellerator_modeling_cuda_python PROPERTIES OUTPUT_NAME _modeling_cuda)
    install(TARGETS cellerator_modeling_cuda_python
        LIBRARY DESTINATION cellerator COMPONENT Python
        RUNTIME DESTINATION cellerator COMPONENT Python)
    if(TARGET cellerator_python_extensions)
        add_dependencies(cellerator_python_extensions cellerator_modeling_cuda_python)
    endif()
endif()
if(CELLERATOR_BUILD_FRAMEWORK_CUDA_SMOKE)
    function(cellerator_add_framework_cuda_smoke case)
        set(target "cellerator_framework_cuda_${case}_smoke")
        add_executable(${target} ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
            "${_cellerator_framework_root}/tests/framework/cuda/${case}_smoke.cu")
        target_link_libraries(${target} PRIVATE ${ARGN})
        set_target_properties(${target} PROPERTIES CUDA_STANDARD 20)
        # Smoke assertions must execute in Release builds as well.
        target_compile_options(${target} PRIVATE -UNDEBUG)
    endfunction()
    cellerator_add_framework_cuda_smoke(runtime
        Cellerator::conditional_dispatch_cuda Cellerator::state_codes_cuda)
    cellerator_add_framework_cuda_smoke(dispatch Cellerator::conditional_dispatch_cuda)
    cellerator_add_framework_cuda_smoke(quantization Cellerator::state_codes_cuda)
    cellerator_add_framework_cuda_smoke(memory
        Cellerator::event_memory_cuda Cellerator::memory_cuda)
    cellerator_add_framework_cuda_smoke(frontier Cellerator::dependency_frontier_cuda)
    cellerator_add_framework_cuda_smoke(attention Cellerator::attention_cuda)
    cellerator_add_framework_cuda_smoke(grouped_linear Cellerator::grouped_linear_cuda)
    cellerator_add_framework_cuda_smoke(packed_embedding
        Cellerator::packed_embedding_cuda Cellerator::state_codes_cuda)
    cellerator_add_framework_cuda_smoke(reactions Cellerator::reaction_cuda)
    cellerator_add_framework_cuda_smoke(regulatory_gates Cellerator::regulatory_gates_cuda)
    # GPU execution requires scheduler admission. Deliberately no add_test().
endif()
install(FILES "${_cellerator_framework_root}/include/Cellerator/execution/conditional/dispatch.hh"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/execution/conditional)
install(TARGETS cellerator_framework_cuda_api EXPORT CelleratorFrameworkCUDATargets)
install(EXPORT CelleratorFrameworkCUDATargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
# The revision-aware transfer bridge depends on the small host conditional
# owner, so keep it in a separate target/export from the numerical CUDA library.
set(_cellerator_parameter_transfer_default OFF)
if(TARGET Cellerator::conditional)
    set(_cellerator_parameter_transfer_default ON)
endif()
option(CELLERATOR_ENABLE_FRAMEWORK_PARAMETER_TRANSFER
    "Build exact-revision host parameter transfer bridge"
    ${_cellerator_parameter_transfer_default})
if(CELLERATOR_ENABLE_FRAMEWORK_PARAMETER_TRANSFER)
    if(NOT TARGET Cellerator::conditional)
        message(FATAL_ERROR "Framework parameter transfer requires Cellerator::conditional; enable CELLERATOR_ENABLE_CONDITIONAL first")
    endif()
    add_library(cellerator_framework_parameter_transfer STATIC
        "${_cellerator_framework_root}/src/execution/conditional/parameter_transfer.cu")
    add_library(Cellerator::framework_parameter_transfer ALIAS cellerator_framework_parameter_transfer)
    set_target_properties(cellerator_framework_parameter_transfer PROPERTIES
        POSITION_INDEPENDENT_CODE ON EXPORT_NAME framework_parameter_transfer
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_features(cellerator_framework_parameter_transfer PUBLIC cxx_std_20)
    target_link_libraries(cellerator_framework_parameter_transfer PUBLIC
        Cellerator::memory_cuda Cellerator::conditional)
    if(CELLERATOR_BUILD_FRAMEWORK_CUDA_SMOKE)
        cellerator_add_framework_cuda_smoke(parameter_transfer
            Cellerator::framework_parameter_transfer)
    endif()
    install(FILES "${_cellerator_framework_root}/include/Cellerator/execution/conditional/parameter_transfer_cuda.hh"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/execution/conditional)
    install(TARGETS cellerator_framework_parameter_transfer EXPORT CelleratorFrameworkParameterTransferTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(EXPORT CelleratorFrameworkParameterTransferTargets NAMESPACE Cellerator::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
endif()

# Append the exports after NativeDataCUDA and its host dependencies. The
# operation targets need memory_cuda and state_codes_cuda. The separate transfer
# export also needs conditional.
function(cellerator_append_framework_cuda_package_config)
    if(NOT DEFINED CELLERATOR_HOST_CONFIG OR NOT EXISTS "${CELLERATOR_HOST_CONFIG}")
        return()
    endif()
    if(TARGET cellerator_framework_cuda_api)
        file(APPEND "${CELLERATOR_HOST_CONFIG}"
            "include(CMakeFindDependencyMacro)\n"
            "find_dependency(CUDAToolkit)\n"
            "if(NOT TARGET Cellerator::framework_cuda)\n"
            "  include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorFrameworkCUDATargets.cmake\")\n"
            "endif()\n"
            "set(Cellerator_framework_cuda_FOUND TRUE)\n")
    endif()
    if(TARGET cellerator_framework_parameter_transfer)
        file(APPEND "${CELLERATOR_HOST_CONFIG}"
            "if(NOT TARGET Cellerator::conditional)\n"
            "  include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorConditionalTargets.cmake\")\n"
            "endif()\n"
            "if(NOT TARGET Cellerator::framework_parameter_transfer)\n"
            "  include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorFrameworkParameterTransferTargets.cmake\")\n"
            "endif()\n"
            "set(Cellerator_framework_parameter_transfer_FOUND TRUE)\n")
    endif()
endfunction()
