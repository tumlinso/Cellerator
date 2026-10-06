include(GNUInstallDirs)
include(CMakeFindDependencyMacro)

if(NOT CELLERATOR_ENABLE_TORCH)
    return()
endif()
if(CELLERATOR_ENABLE_CUDA STREQUAL "OFF" OR NOT CMAKE_CUDA_COMPILER_LOADED)
    message(FATAL_ERROR "CELLERATOR_ENABLE_TORCH requires the CUDA backend")
endif()

set(CELLERATOR_LIBTORCH_ROOT "/usr/local" CACHE PATH
    "Optional local libtorch installation prefix")
if(DEFINED ENV{LIBTORCH_PATH} AND NOT "$ENV{LIBTORCH_PATH}" STREQUAL "")
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{LIBTORCH_PATH}")
    set(Torch_DIR "$ENV{LIBTORCH_PATH}/share/cmake/Torch" CACHE PATH
        "Explicit libtorch package" FORCE)
elseif(EXISTS "${CELLERATOR_LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
    list(PREPEND CMAKE_PREFIX_PATH "${CELLERATOR_LIBTORCH_ROOT}")
endif()

# When building the Python extra, use the Torch installation belonging to the
# selected interpreter. Standalone C++ users can continue to use Torch_DIR or
# CMAKE_PREFIX_PATH without triggering Python discovery.
if(CELLERATOR_ENABLE_PYTHON)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import torch; print(torch.utils.cmake_prefix_path)"
        RESULT_VARIABLE cellerator_torch_python_status
        OUTPUT_VARIABLE cellerator_torch_python_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(cellerator_torch_python_status EQUAL 0 AND cellerator_torch_python_prefix)
        list(PREPEND CMAKE_PREFIX_PATH "${cellerator_torch_python_prefix}")
    endif()
endif()
# Torch's Caffe2 package may use TORCH_CUDA_ARCH_LIST instead of CMake's
# CUDA_ARCHITECTURES while configuring downstream CUDA code. Keep its choice
# aligned with the explicitly selected native architectures when possible;
# this avoids probing GPUs at configure time and avoids unrelated fallback
# architecture lists. An explicit environment setting remains authoritative.
if((NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST}
    OR "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
   AND (NOT DEFINED TORCH_CUDA_ARCH_LIST
        OR "${TORCH_CUDA_ARCH_LIST}" STREQUAL "")
   AND CMAKE_CUDA_ARCHITECTURES
   AND NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(all|native)$")
    set(_cellerator_torch_arch_entries)
    foreach(_cellerator_cuda_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        set(_cellerator_cuda_arch_suffix "")
        set(_cellerator_cuda_arch_digits "${_cellerator_cuda_arch}")
        if(_cellerator_cuda_arch_digits MATCHES "(-real|-virtual)$")
            set(_cellerator_cuda_arch_suffix "${CMAKE_MATCH_1}")
            string(REGEX REPLACE "(-real|-virtual)$" ""
                _cellerator_cuda_arch_digits "${_cellerator_cuda_arch_digits}")
        endif()
        if(_cellerator_cuda_arch_digits MATCHES "^([0-9])([0-9])$")
            list(APPEND _cellerator_torch_arch_entries
                "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}${_cellerator_cuda_arch_suffix}")
        else()
            set(_cellerator_torch_arch_entries)
            break()
        endif()
    endforeach()
    if(_cellerator_torch_arch_entries)
        string(REPLACE ";" " " _cellerator_torch_arch_list
            "${_cellerator_torch_arch_entries}")
        set(TORCH_CUDA_ARCH_LIST "${_cellerator_torch_arch_list}" CACHE STRING
            "CUDA architectures passed to the optional libtorch adapter")
        if(NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST}
           OR "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
            set(ENV{TORCH_CUDA_ARCH_LIST} "${_cellerator_torch_arch_list}")
        endif()
    endif()
endif()
find_package(Torch REQUIRED)

add_library(cellerator_torch_runtime INTERFACE)
add_library(Cellerator::torch_runtime ALIAS cellerator_torch_runtime)
set_target_properties(cellerator_torch_runtime PROPERTIES EXPORT_NAME torch_runtime)
target_link_libraries(cellerator_torch_runtime INTERFACE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse)
target_compile_features(cellerator_torch_runtime INTERFACE cxx_std_17)
if(TORCH_INCLUDE_DIRS)
    target_include_directories(cellerator_torch_runtime SYSTEM INTERFACE
        ${TORCH_INCLUDE_DIRS})
endif()
if(TORCH_CXX_FLAGS)
    separate_arguments(CELLERATOR_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(cellerator_torch_runtime INTERFACE
        ${CELLERATOR_TORCH_CXX_FLAGS})
endif()

add_library(cellerator_torch_native_views STATIC
    src/native_views.cc)
add_library(Cellerator::torch_native_views ALIAS cellerator_torch_native_views)
target_include_directories(cellerator_torch_native_views PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_torch_native_views PUBLIC
    Cellerator::torch_runtime
    Cellerator::runtime)
target_compile_features(cellerator_torch_native_views PUBLIC cxx_std_17)

add_library(cellerator_torch_program_ops STATIC
    src/program_ops.cu)
add_library(Cellerator::torch_program_ops ALIAS cellerator_torch_program_ops)
target_include_directories(cellerator_torch_program_ops PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_torch_program_ops PUBLIC
    Cellerator::torch_runtime
    Cellerator::executable_program)
set_target_properties(cellerator_torch_program_ops PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE ON)

add_library(cellerator_torch_autograd_ops STATIC
    src/autograd_ops.cu)
add_library(Cellerator::torch_autograd_ops ALIAS cellerator_torch_autograd_ops)
target_include_directories(cellerator_torch_autograd_ops PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_torch_autograd_ops PUBLIC
    Cellerator::torch_runtime
    Cellerator::training_program)
set_target_properties(cellerator_torch_autograd_ops PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE ON)

add_library(cellerator_torch_mechanism STATIC
    src/mechanism.cc)
add_library(Cellerator::torch_mechanism ALIAS cellerator_torch_mechanism)
target_include_directories(cellerator_torch_mechanism PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_torch_mechanism PUBLIC
    Cellerator::torch_runtime
    Cellerator::indexed_mechanism)
target_compile_features(cellerator_torch_mechanism PUBLIC cxx_std_20)
set_target_properties(cellerator_torch_mechanism PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

if(TARGET Cellerator::model_ops)
    add_library(cellerator_torch_model_ops STATIC
        src/model_ops.cu)
    add_library(Cellerator::torch_model_ops ALIAS cellerator_torch_model_ops)
    target_include_directories(cellerator_torch_model_ops PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_link_libraries(cellerator_torch_model_ops PUBLIC
        Cellerator::torch_runtime
        Cellerator::model_ops)
    set_target_properties(cellerator_torch_model_ops PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
        POSITION_INDEPENDENT_CODE ON)
    cellerator_enable_perf(cellerator_torch_model_ops)
endif()

add_library(cellerator_torch INTERFACE)
add_library(Cellerator::torch ALIAS cellerator_torch)
set_target_properties(cellerator_torch PROPERTIES EXPORT_NAME torch)
target_include_directories(cellerator_torch INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_torch INTERFACE
    Cellerator::torch_native_views
    Cellerator::torch_program_ops
    Cellerator::torch_autograd_ops
    Cellerator::torch_mechanism)
if(TARGET Cellerator::torch_model_ops)
    target_link_libraries(cellerator_torch INTERFACE Cellerator::torch_model_ops)
endif()
target_compile_features(cellerator_torch INTERFACE cxx_std_20)

if(CELLERATOR_ENABLE_PYTHON)
    pybind11_add_module(cellerator_python_torch MODULE
        src/python_module.cc)
    set_target_properties(cellerator_python_torch PROPERTIES
        OUTPUT_NAME _torch
        CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
        INSTALL_RPATH "$ORIGIN/../.libs")
    target_include_directories(cellerator_python_torch PRIVATE
        ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(cellerator_python_torch PRIVATE
        Cellerator::torch
        Cellerator::indexed_mechanism)
    install(TARGETS cellerator_python_torch
        LIBRARY DESTINATION cellerator/torch COMPONENT Python
        RUNTIME DESTINATION cellerator/torch COMPONENT Python)
    if(TARGET cellerator_python_extensions)
        add_dependencies(cellerator_python_extensions cellerator_python_torch)
    endif()
endif()

if(CELLERATOR_BUILD_TESTS)
    function(cellerator_add_torch_adapter_test target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE Cellerator::torch)
        set_target_properties(${target} PROPERTIES
            CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
            CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    endfunction()

    cellerator_add_torch_adapter_test(cellerator_torch_native_views_test
        ${PROJECT_SOURCE_DIR}/tests/bindings/torch/native_views_test.cc)
    cellerator_add_torch_adapter_test(cellerator_torch_program_ops_test
        ${PROJECT_SOURCE_DIR}/tests/bindings/torch/program_ops_test.cc)
    cellerator_add_torch_adapter_test(cellerator_torch_autograd_ops_test
        ${PROJECT_SOURCE_DIR}/tests/bindings/torch/autograd_ops_test.cc)
    add_executable(celleratorTorchQuantitativeSmokeTest
        ${PROJECT_SOURCE_DIR}/tests/bindings/torch/quantitative_smoke_test.cc
        ${PROJECT_SOURCE_DIR}/bench/ce_live/runtime_fixture/quantitative_fixture.cc)
    target_include_directories(celleratorTorchQuantitativeSmokeTest PRIVATE
        ${PROJECT_SOURCE_DIR})
    target_link_libraries(celleratorTorchQuantitativeSmokeTest PRIVATE
        Cellerator::torch
        Cellerator::identity_registry)
    set_target_properties(celleratorTorchQuantitativeSmokeTest PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    if(TARGET Cellerator::torch_model_ops AND
       EXISTS "${PROJECT_SOURCE_DIR}/tests/bindings/model_ops/model_custom_ops_test.cc")
        add_executable(celleratorTorchModelCustomOpsTest
            ${PROJECT_SOURCE_DIR}/tests/bindings/model_ops/model_custom_ops_test.cc)
        target_link_libraries(celleratorTorchModelCustomOpsTest PRIVATE
            Cellerator::torch_model_ops)
        set_target_properties(celleratorTorchModelCustomOpsTest PROPERTIES
            CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
            CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    endif()
    add_custom_target(cellerator_torch_adapter_compile_tests
        DEPENDS cellerator_torch_native_views_test
                cellerator_torch_program_ops_test
                cellerator_torch_autograd_ops_test
                celleratorTorchQuantitativeSmokeTest)
    if(TARGET celleratorTorchModelCustomOpsTest)
        add_dependencies(cellerator_torch_adapter_compile_tests
            celleratorTorchModelCustomOpsTest)
    endif()
endif()

if(CELLERATOR_BUILD_TORCH_EXAMPLES)
    if(NOT CELLERATOR_ENABLE_CELLSHARD OR NOT TARGET CellShard::runtime)
        message(FATAL_ERROR
            "CELLERATOR_BUILD_TORCH_EXAMPLES requires CELLERATOR_ENABLE_CELLSHARD")
    endif()
    add_subdirectory(${PROJECT_SOURCE_DIR}/examples/torch
        ${PROJECT_BINARY_DIR}/examples/torch)
endif()

set_property(TARGET cellerator_torch_runtime PROPERTY EXPORT_NAME torch_runtime)
set_property(TARGET cellerator_torch_native_views PROPERTY EXPORT_NAME torch_native_views)
set_property(TARGET cellerator_torch_program_ops PROPERTY EXPORT_NAME torch_program_ops)
set_property(TARGET cellerator_torch_autograd_ops PROPERTY EXPORT_NAME torch_autograd_ops)
set_property(TARGET cellerator_torch_mechanism PROPERTY EXPORT_NAME torch_mechanism)
set(_cellerator_torch_targets
    cellerator_torch_runtime
    cellerator_torch_native_views
    cellerator_torch_program_ops
    cellerator_torch_autograd_ops
    cellerator_torch_mechanism
    cellerator_torch)
if(TARGET cellerator_torch_model_ops)
    set_property(TARGET cellerator_torch_model_ops PROPERTY EXPORT_NAME torch_model_ops)
    list(APPEND _cellerator_torch_targets cellerator_torch_model_ops)
endif()
install(TARGETS ${_cellerator_torch_targets}
    EXPORT CelleratorTorchTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Torch
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Torch)
install(EXPORT CelleratorTorchTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator COMPONENT Torch)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/Cellerator/bindings/torch
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/bindings COMPONENT Torch)

file(APPEND "${CELLERATOR_HOST_CONFIG}"
    "if(\"torch\" IN_LIST Cellerator_FIND_COMPONENTS)\n"
    "  include(CMakeFindDependencyMacro)\n"
    "  find_dependency(Torch REQUIRED)\n"
    "  include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorTorchTargets.cmake\")\n"
    "  set(Cellerator_torch_FOUND TRUE)\n"
    "else()\n  set(Cellerator_torch_FOUND FALSE)\nendif()\n"
    "foreach(_cellerator_component IN LISTS Cellerator_FIND_COMPONENTS)\n"
    "  if(Cellerator_FIND_REQUIRED_${_cellerator_component} AND NOT Cellerator_${_cellerator_component}_FOUND)\n"
    "    set(Cellerator_FOUND FALSE)\n"
    "    set(Cellerator_NOT_FOUND_MESSAGE \"Required Cellerator component '${_cellerator_component}' is unavailable\")\n"
    "  endif()\n"
    "endforeach()\n")
