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
get_filename_component(_cellerator_framework_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_library(cellerator_framework_cuda STATIC
    "${_cellerator_framework_root}/src/modeling/cuda/runtime.cu"
    "${_cellerator_framework_root}/src/modeling/cuda/plans.cu"
    "${_cellerator_framework_root}/src/modeling/cuda/dispatch.cu"
    "${_cellerator_framework_root}/src/modeling/cuda/quantization.cu"
    "${_cellerator_framework_root}/src/modeling/cuda/memory.cu"
    "${_cellerator_framework_root}/src/modeling/cuda/frontier.cu"
    "${_cellerator_framework_root}/src/modeling/cuda/attention.cu"
    "${_cellerator_framework_root}/src/modeling/cuda/grouped_linear.cu"
    "${_cellerator_framework_root}/src/modeling/cuda/packed_embedding.cu"
    "${_cellerator_framework_root}/src/modeling/cuda/reactions.cu"
    "${_cellerator_framework_root}/src/modeling/cuda/regulatory_gates.cu")
add_library(Cellerator::framework_cuda ALIAS cellerator_framework_cuda)
set_target_properties(cellerator_framework_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON EXPORT_NAME framework_cuda
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_features(cellerator_framework_cuda PUBLIC cxx_std_17)
target_include_directories(cellerator_framework_cuda PUBLIC
    $<BUILD_INTERFACE:${_cellerator_framework_root}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_framework_cuda PUBLIC CUDA::cudart CUDA::cublas)
if(CELLERATOR_ENABLE_FRAMEWORK_CUDA_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(cellerator_modeling_cuda_python MODULE
        "${_cellerator_framework_root}/bindings/modeling_cuda/module.cc")
    target_link_libraries(cellerator_modeling_cuda_python PRIVATE Cellerator::framework_cuda Python::Module)
    set_target_properties(cellerator_modeling_cuda_python PROPERTIES OUTPUT_NAME _modeling_cuda)
    install(TARGETS cellerator_modeling_cuda_python
        LIBRARY DESTINATION cellerator COMPONENT Python
        RUNTIME DESTINATION cellerator COMPONENT Python)
    if(TARGET cellerator_python_extensions)
        add_dependencies(cellerator_python_extensions cellerator_modeling_cuda_python)
    endif()
endif()
if(CELLERATOR_BUILD_FRAMEWORK_CUDA_SMOKE)
    foreach(_case runtime dispatch quantization memory frontier attention grouped_linear packed_embedding reactions regulatory_gates)
        add_executable(cellerator_framework_cuda_${_case}_smoke
            "${_cellerator_framework_root}/tests/framework/cuda/${_case}_smoke.cu")
        target_link_libraries(cellerator_framework_cuda_${_case}_smoke PRIVATE Cellerator::framework_cuda)
        set_target_properties(cellerator_framework_cuda_${_case}_smoke PROPERTIES CUDA_STANDARD 17)
        # Smoke assertions must execute in Release builds as well.
        target_compile_options(cellerator_framework_cuda_${_case}_smoke PRIVATE -UNDEBUG)
    endforeach()
    # GPU execution requires scheduler admission. Deliberately no add_test().
endif()
install(DIRECTORY "${_cellerator_framework_root}/include/Cellerator/modeling/cuda"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/modeling
    PATTERN "parameter_transfer.hh" EXCLUDE)
install(FILES "${_cellerator_framework_root}/include/Cellerator/execution/conditional/dispatch.hh"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/execution/conditional)
install(TARGETS cellerator_framework_cuda EXPORT CelleratorFrameworkCUDATargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT CelleratorFrameworkCUDATargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
if(DEFINED CELLERATOR_HOST_CONFIG AND EXISTS "${CELLERATOR_HOST_CONFIG}")
    file(APPEND "${CELLERATOR_HOST_CONFIG}"
        "include(CMakeFindDependencyMacro)\n"
        "find_dependency(CUDAToolkit)\n"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorFrameworkCUDATargets.cmake\")\n"
        "set(Cellerator_framework_cuda_FOUND TRUE)\n")
endif()

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
        "${_cellerator_framework_root}/src/modeling/cuda/parameter_transfer.cu")
    add_library(Cellerator::framework_parameter_transfer ALIAS cellerator_framework_parameter_transfer)
    set_target_properties(cellerator_framework_parameter_transfer PROPERTIES
        POSITION_INDEPENDENT_CODE ON EXPORT_NAME framework_parameter_transfer
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_features(cellerator_framework_parameter_transfer PUBLIC cxx_std_17)
    target_link_libraries(cellerator_framework_parameter_transfer PUBLIC
        Cellerator::framework_cuda Cellerator::conditional)
    if(CELLERATOR_BUILD_FRAMEWORK_CUDA_SMOKE)
        add_executable(cellerator_framework_cuda_parameter_transfer_smoke
            "${_cellerator_framework_root}/tests/framework/cuda/parameter_transfer_smoke.cu")
        target_link_libraries(cellerator_framework_cuda_parameter_transfer_smoke PRIVATE
            Cellerator::framework_parameter_transfer)
        set_target_properties(cellerator_framework_cuda_parameter_transfer_smoke PROPERTIES CUDA_STANDARD 17)
        target_compile_options(cellerator_framework_cuda_parameter_transfer_smoke PRIVATE -UNDEBUG)
    endif()
    install(FILES "${_cellerator_framework_root}/include/Cellerator/modeling/cuda/parameter_transfer.hh"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/modeling/cuda)
    install(TARGETS cellerator_framework_parameter_transfer EXPORT CelleratorFrameworkParameterTransferTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(EXPORT CelleratorFrameworkParameterTransferTargets NAMESPACE Cellerator::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
    if(DEFINED CELLERATOR_HOST_CONFIG AND EXISTS "${CELLERATOR_HOST_CONFIG}")
        file(APPEND "${CELLERATOR_HOST_CONFIG}"
            "if(NOT TARGET Cellerator::conditional)\n"
            "  include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorConditionalTargets.cmake\")\n"
            "endif()\n"
            "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorFrameworkParameterTransferTargets.cmake\")\n"
            "set(Cellerator_framework_parameter_transfer_FOUND TRUE)\n")
    endif()
endif()
