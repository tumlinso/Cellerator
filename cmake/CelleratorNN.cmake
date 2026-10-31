# Optional native C++ neural-network foundation. This is a separate Torch
# integration from the CUDA-only legacy torch_runtime target.
include(GNUInstallDirs)

if(CELLERATOR_ENABLE_NN_CUDA AND NOT CELLERATOR_ENABLE_NN)
    message(FATAL_ERROR "CELLERATOR_ENABLE_NN_CUDA requires CELLERATOR_ENABLE_NN")
endif()

if(NOT CELLERATOR_ENABLE_NN)
    return()
endif()

include("${PROJECT_SOURCE_DIR}/cmake/CelleratorTorchDependency.cmake")

# JSON is used only by the native manifest/checkpoint implementation. Copy its
# include path privately instead of exporting a header-only package target to
# downstream NN users.
find_package(nlohmann_json CONFIG QUIET)
if(TARGET nlohmann_json::nlohmann_json)
    get_target_property(cellerator_nn_json_includes
        nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
else()
    find_path(cellerator_nn_json_include_dir
        NAMES nlohmann/json.hpp
        HINTS ${CMAKE_PREFIX_PATH} "$ENV{CONDA_PREFIX}" "${CELLERATOR_LIBTORCH_ROOT}"
        PATH_SUFFIXES include
        REQUIRED)
    set(cellerator_nn_json_includes "${cellerator_nn_json_include_dir}")
endif()

find_package(OpenSSL REQUIRED COMPONENTS Crypto)

set(cellerator_nn_sources)
file(GLOB cellerator_nn_discovered_sources CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/nn/*.cc")
foreach(cellerator_nn_source IN LISTS cellerator_nn_discovered_sources)
    if(NOT cellerator_nn_source MATCHES "/ops_cuda\\.cc$")
        list(APPEND cellerator_nn_sources "${cellerator_nn_source}")
    endif()
endforeach()
if(NOT cellerator_nn_sources)
    message(FATAL_ERROR "CELLERATOR_ENABLE_NN=ON found no native NN sources")
endif()

# The CUDA adapter target depends only on ATen/Torch and the existing generic
# framework CUDA runtime. It intentionally has no link edge back to Cellerator::nn;
# the NN target links it when enabled, avoiding a static-library cycle.
set(cellerator_nn_export_targets)
if(CELLERATOR_ENABLE_NN_CUDA)
    if(NOT TARGET Cellerator::framework_cuda)
        message(FATAL_ERROR
            "CELLERATOR_ENABLE_NN_CUDA requires CELLERATOR_ENABLE_FRAMEWORK_CUDA=ON")
    endif()
    if(NOT CMAKE_CUDA_COMPILER_LOADED)
        message(FATAL_ERROR
            "CELLERATOR_ENABLE_NN_CUDA requires a loaded CUDA language toolchain")
    endif()
    if(NOT TARGET torch_cuda OR NOT TARGET c10_cuda)
        message(FATAL_ERROR
            "CELLERATOR_ENABLE_NN_CUDA requires a CUDA-enabled Torch package")
    endif()
    add_library(cellerator_nn_cuda STATIC
        "${PROJECT_SOURCE_DIR}/src/nn/ops_cuda.cc")
    add_library(Cellerator::nn_cuda ALIAS cellerator_nn_cuda)
    set_target_properties(cellerator_nn_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME nn_cuda)
    target_include_directories(cellerator_nn_cuda PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_compile_features(cellerator_nn_cuda PUBLIC cxx_std_20)
    target_compile_definitions(cellerator_nn_cuda PUBLIC CELLERATOR_NN_HAS_CUDA=1)
    target_link_libraries(cellerator_nn_cuda PUBLIC
        ${CELLERATOR_TORCH_LINK_TARGETS}
        torch_cuda
        c10_cuda
        Cellerator::framework_cuda)
    if(CELLERATOR_TORCH_CXX_OPTIONS)
        target_compile_options(cellerator_nn_cuda PUBLIC
            ${CELLERATOR_TORCH_CXX_OPTIONS})
    endif()
    list(APPEND cellerator_nn_export_targets cellerator_nn_cuda)
endif()

add_library(cellerator_nn STATIC ${cellerator_nn_sources})
add_library(Cellerator::nn ALIAS cellerator_nn)
set_target_properties(cellerator_nn PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME nn)
target_include_directories(cellerator_nn PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${cellerator_nn_json_includes})
target_compile_features(cellerator_nn PUBLIC cxx_std_20)
target_link_libraries(cellerator_nn PUBLIC ${CELLERATOR_TORCH_LINK_TARGETS})
target_link_libraries(cellerator_nn PRIVATE OpenSSL::Crypto)
if(CELLERATOR_TORCH_CXX_OPTIONS)
    target_compile_options(cellerator_nn PUBLIC ${CELLERATOR_TORCH_CXX_OPTIONS})
endif()
if(CELLERATOR_ENABLE_NN_CUDA)
    target_link_libraries(cellerator_nn PUBLIC Cellerator::nn_cuda)
endif()
list(APPEND cellerator_nn_export_targets cellerator_nn)

if(CELLERATOR_ENABLE_PYTHON)
    if(NOT TARGET cellerator_python_extensions)
        message(FATAL_ERROR
            "The NN Python binding requires the existing CELLERATOR_ENABLE_PYTHON setup")
    endif()
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/bindings/nn/module.cc")
        message(FATAL_ERROR "bindings/nn/module.cc is required for the NN Python binding")
    endif()
    pybind11_add_module(cellerator_nn_python MODULE
        "${PROJECT_SOURCE_DIR}/bindings/nn/module.cc")
    set_target_properties(cellerator_nn_python PROPERTIES
        OUTPUT_NAME _nn_native
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        BUILD_RPATH "${CELLERATOR_TORCH_LIB_DIR}"
        INSTALL_RPATH "$ORIGIN/.libs;$ORIGIN/../torch/lib")
    target_link_libraries(cellerator_nn_python PRIVATE
        Cellerator::nn
        Python::Module)
    install(TARGETS cellerator_nn_python
        LIBRARY DESTINATION cellerator COMPONENT Python
        RUNTIME DESTINATION cellerator COMPONENT Python)
    add_dependencies(cellerator_python_extensions cellerator_nn_python)
endif()

option(CELLERATOR_BUILD_NN_TESTS
    "Build standalone native NN CPU tests (CUDA tests compile only when NN CUDA is enabled)"
    ${CELLERATOR_BUILD_TESTS})
if(CELLERATOR_BUILD_TESTS AND CELLERATOR_BUILD_NN_TESTS)
    enable_testing()
    file(GLOB cellerator_nn_test_sources CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/tests/nn/native/*.cc")
    foreach(cellerator_nn_test_source IN LISTS cellerator_nn_test_sources)
        get_filename_component(cellerator_nn_test_name "${cellerator_nn_test_source}" NAME_WE)
        if(cellerator_nn_test_name STREQUAL "ops_cuda" AND NOT CELLERATOR_ENABLE_NN_CUDA)
            continue()
        endif()
        if(cellerator_nn_test_name STREQUAL "biology_gene_vocabulary_adapter"
           AND NOT TARGET Cellerator::tokenization)
            message(STATUS
                "Skipping NN/tokenization adapter test: enable CELLERATOR_ENABLE_TOKENIZATION")
            continue()
        endif()
        set(cellerator_nn_test_target "native_nn_${cellerator_nn_test_name}_test")
        add_executable(${cellerator_nn_test_target} "${cellerator_nn_test_source}")
        target_link_libraries(${cellerator_nn_test_target} PRIVATE Cellerator::nn)
        target_compile_features(${cellerator_nn_test_target} PRIVATE cxx_std_20)
        if(cellerator_nn_test_name STREQUAL "biology_gene_vocabulary_adapter")
            target_link_libraries(${cellerator_nn_test_target} PRIVATE Cellerator::tokenization)
        endif()
        add_test(NAME ${cellerator_nn_test_target} COMMAND ${cellerator_nn_test_target})
        if(cellerator_nn_test_name STREQUAL "ops_cuda")
            # Compile/link coverage is available here. Running CUDA tests is
            # still an explicit scheduler-admitted operation.
            set_tests_properties(${cellerator_nn_test_target} PROPERTIES DISABLED TRUE LABELS cuda)
        endif()
    endforeach()
endif()

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/Cellerator/nn"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator"
    COMPONENT NN)
install(FILES
    "${PROJECT_SOURCE_DIR}/include/Cellerator/biology/coordinates.hh"
    "${PROJECT_SOURCE_DIR}/include/Cellerator/biology/feature_binding.hh"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/biology"
    COMPONENT NN)
install(FILES "${PROJECT_SOURCE_DIR}/include/Cellerator/nn.hh"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator"
    COMPONENT NN)
install(TARGETS ${cellerator_nn_export_targets}
    EXPORT CelleratorNNTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT NN
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT NN)
install(EXPORT CelleratorNNTargets NAMESPACE Cellerator::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator"
    COMPONENT NN)

if(DEFINED CELLERATOR_HOST_CONFIG AND EXISTS "${CELLERATOR_HOST_CONFIG}")
    set(cellerator_nn_cuda_config "")
    if(CELLERATOR_ENABLE_NN_CUDA)
        set(cellerator_nn_cuda_config
            "  find_dependency(CUDAToolkit REQUIRED)\n"
            "  if(NOT TARGET Cellerator::framework_cuda)\n"
            "    include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorFrameworkCUDATargets.cmake\")\n"
            "  endif()\n")
    endif()
    file(APPEND "${CELLERATOR_HOST_CONFIG}"
        "if(\"nn\" IN_LIST Cellerator_FIND_COMPONENTS OR \"nn_cuda\" IN_LIST Cellerator_FIND_COMPONENTS)\n"
        "  include(CMakeFindDependencyMacro)\n"
        "  find_dependency(Torch REQUIRED)\n"
        "  find_dependency(OpenSSL REQUIRED COMPONENTS Crypto)\n"
        "${cellerator_nn_cuda_config}"
        "  if(NOT TARGET Cellerator::nn)\n"
        "    include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorNNTargets.cmake\")\n"
        "  endif()\n"
        "  set(Cellerator_nn_FOUND TRUE)\n"
        "endif()\n"
        "if(\"nn_cuda\" IN_LIST Cellerator_FIND_COMPONENTS)\n"
        "  set(Cellerator_nn_cuda_FOUND TRUE)\n"
        "endif()\n")
endif()
