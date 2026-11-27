include(GNUInstallDirs)

add_subdirectory(src/layout)

option(CELLERATOR_ENABLE_SPARSE_SUPPORT
    "Build reusable structure, support, indexing, and gating primitives" ON)
if(NOT CELLERATOR_ENABLE_SPARSE_SUPPORT)
    return()
endif()

add_subdirectory(src/geometry/dependency)
add_subdirectory(src/structure/indexing)
add_subdirectory(src/ops/condition)
if(TARGET cellerator_layout)
    target_link_libraries(cellerator_layout PUBLIC Cellerator::support)
endif()

# Sparse-support package target for the native support, indexing, gating, and
# layout owners.
add_library(cellerator_sparse_support INTERFACE)
add_library(Cellerator::sparse_support ALIAS cellerator_sparse_support)
target_link_libraries(cellerator_sparse_support INTERFACE
    Cellerator::support Cellerator::indexing Cellerator::gating Cellerator::layout)
target_include_directories(cellerator_sparse_support INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(CELLERATOR_BUILD_TESTS)
    enable_testing()
endif()

# CUDA builds export the linked closure through NativeFoundation.cmake.
# Host-only consumers retain the same Torch-free support facade.
if(CELLERATOR_ENABLE_CUDA STREQUAL "OFF")
    set_target_properties(cellerator_support cellerator_layout PROPERTIES EXPORT_NAME support)
    set_target_properties(cellerator_layout PROPERTIES EXPORT_NAME layout)
    set_target_properties(cellerator_indexing PROPERTIES EXPORT_NAME indexing)
    set_target_properties(cellerator_gating PROPERTIES EXPORT_NAME gating)
    set_target_properties(cellerator_sparse_support PROPERTIES EXPORT_NAME sparse_support)
    install(TARGETS cellerator_support cellerator_indexing cellerator_gating
        cellerator_layout cellerator_sparse_support EXPORT CelleratorSparseSupportTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(EXPORT CelleratorSparseSupportTargets NAMESPACE Cellerator::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
    file(APPEND "${CELLERATOR_HOST_CONFIG}"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorSparseSupportTargets.cmake\")\n"
        "set(Cellerator_sparse_support_FOUND TRUE)\n")
    add_subdirectory(tests/sparse_support ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL})
    if(EXISTS "${PROJECT_SOURCE_DIR}/bench/sparse_support/CMakeLists.txt")
        add_subdirectory(bench/sparse_support ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL})
    endif()
endif()
