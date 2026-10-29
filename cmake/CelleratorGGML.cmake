# Optional CPU quantization bridge. The rest of Cellerator does not depend on
# ggml; the pinned upstream source is fetched only when explicitly enabled.
include(GNUInstallDirs)
option(CELLERATOR_ENABLE_GGML "Build the optional ggml quantization bridge" OFF)
get_filename_component(cellerator_ggml_source_root
    "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

if(CELLERATOR_ENABLE_GGML)
    include(FetchContent)

    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    set(GGML_HIP OFF CACHE BOOL "" FORCE)
    set(GGML_SYCL OFF CACHE BOOL "" FORCE)
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
    set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
    set(GGML_MUSA OFF CACHE BOOL "" FORCE)
    set(GGML_WEBGPU OFF CACHE BOOL "" FORCE)
    set(GGML_ZENDNN OFF CACHE BOOL "" FORCE)
    set(GGML_OPENVINO OFF CACHE BOOL "" FORCE)
    set(GGML_VIRTGPU OFF CACHE BOOL "" FORCE)
    set(GGML_RPC OFF CACHE BOOL "" FORCE)
    set(GGML_ET OFF CACHE BOOL "" FORCE)
    set(GGML_HEXAGON OFF CACHE BOOL "" FORCE)
    set(GGML_BLAS OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(ggml
        GIT_REPOSITORY https://github.com/ggml-org/ggml.git
        GIT_TAG facdc44a67da17df4433adcc468b424f48e9ded2
        GIT_SHALLOW FALSE
    )
    # ggml defaults to shared libraries on Linux. Keep its implementation
    # static so the Python extension does not depend on a separately installed
    # libggml-base at import time, then restore Cellerator's caller setting.
    if(DEFINED BUILD_SHARED_LIBS)
        set(cellerator_ggml_previous_build_shared_libs "${BUILD_SHARED_LIBS}")
        set(cellerator_ggml_restore_build_shared_libs TRUE)
    else()
        set(cellerator_ggml_restore_build_shared_libs FALSE)
    endif()
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_MakeAvailable(ggml)
    FetchContent_GetProperties(ggml SOURCE_DIR cellerator_ggml_upstream_source)
    if(cellerator_ggml_restore_build_shared_libs)
        set(BUILD_SHARED_LIBS "${cellerator_ggml_previous_build_shared_libs}")
    else()
        unset(BUILD_SHARED_LIBS)
    endif()
    set_target_properties(ggml-base PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(cellerator_ggml_quantization STATIC
        "${cellerator_ggml_source_root}/src/interop/ggml/quantization.cc")
    add_library(Cellerator::ggml_quantization ALIAS cellerator_ggml_quantization)
    target_compile_features(cellerator_ggml_quantization PUBLIC cxx_std_20)
    target_include_directories(cellerator_ggml_quantization PUBLIC
        $<BUILD_INTERFACE:${cellerator_ggml_source_root}/include>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(cellerator_ggml_quantization PUBLIC ggml-base)
    set_target_properties(cellerator_ggml_quantization PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME ggml_quantization)
    set_target_properties(ggml-base PROPERTIES EXPORT_NAME ggml-base)

    # Export the implementation dependency with the adapter. The public
    # Cellerator header deliberately does not expose ggml types, but static
    # consumers still need ggml-base and its headers at link time.
    install(TARGETS ggml-base cellerator_ggml_quantization
        EXPORT CelleratorGGMLTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(FILES "${cellerator_ggml_upstream_source}/LICENSE"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/ggml
        COMPONENT Python)
    install(FILES "${cellerator_ggml_source_root}/include/Cellerator/interop/ggml/quantization.hh"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/interop/ggml)
    install(EXPORT CelleratorGGMLTargets NAMESPACE Cellerator::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
    if(DEFINED CELLERATOR_HOST_CONFIG AND EXISTS "${CELLERATOR_HOST_CONFIG}")
        file(APPEND "${CELLERATOR_HOST_CONFIG}"
            "include(CMakeFindDependencyMacro)\n"
            "find_dependency(Threads)\n"
            "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorGGMLTargets.cmake\")\n"
            "set(Cellerator_ggml_FOUND TRUE)\n")
    endif()

    if(CELLERATOR_ENABLE_PYTHON)
        pybind11_add_module(cellerator_python_ggml
            "${cellerator_ggml_source_root}/bindings/ggml/module.cc")
        set_target_properties(cellerator_python_ggml PROPERTIES OUTPUT_NAME _ggml_native)
        target_compile_features(cellerator_python_ggml PRIVATE cxx_std_20)
        target_link_libraries(cellerator_python_ggml PRIVATE
            Cellerator::ggml_quantization Python::Module)
        install(TARGETS cellerator_python_ggml
            LIBRARY DESTINATION cellerator COMPONENT Python
            RUNTIME DESTINATION cellerator COMPONENT Python)
        if(TARGET cellerator_python_extensions)
            add_dependencies(cellerator_python_extensions cellerator_python_ggml)
        endif()
    endif()
endif()
