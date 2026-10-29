include(GNUInstallDirs)

option(CELLERATOR_ENABLE_SPARSE_SUPPORT
    "Build reusable support, indexing and conditional gating primitives" ON)
if(NOT CELLERATOR_ENABLE_SPARSE_SUPPORT)
    return()
endif()

add_subdirectory(src/compute/support)
if(CELLERATOR_BUILD_TESTS)
    enable_testing()
endif()

# CUDA builds export the linked closure through NativeFoundation.cmake.
# Host-only consumers retain the same Torch-free support target.
if(CELLERATOR_ENABLE_CUDA STREQUAL "OFF")
    set_target_properties(cellerator_sparse_support PROPERTIES EXPORT_NAME sparse_support)
    install(TARGETS cellerator_sparse_support EXPORT CelleratorSparseSupportTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(EXPORT CelleratorSparseSupportTargets NAMESPACE Cellerator::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
    file(APPEND "${CELLERATOR_HOST_CONFIG}"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorSparseSupportTargets.cmake\")\n"
        "set(Cellerator_sparse_support_FOUND TRUE)\n")
    if(CELLERATOR_BUILD_TESTS)
        add_subdirectory(tests/sparse_support)
    endif()
    if(EXISTS "${PROJECT_SOURCE_DIR}/bench/sparse_support/CMakeLists.txt")
        add_subdirectory(bench/sparse_support)
    endif()
endif()
install(DIRECTORY
    "${PROJECT_SOURCE_DIR}/include/Cellerator/compute/support"
    "${PROJECT_SOURCE_DIR}/include/Cellerator/compute/indexing"
    "${PROJECT_SOURCE_DIR}/include/Cellerator/compute/gating"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/compute")
