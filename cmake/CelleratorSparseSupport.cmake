include(GNUInstallDirs)

add_subdirectory(src/layout)

option(CELLERATOR_ENABLE_SPARSE_SUPPORT
    "Build reusable structure, support, indexing, and gating primitives" ON)
if(NOT CELLERATOR_ENABLE_SPARSE_SUPPORT)
    return()
endif()

add_subdirectory(src/support)
add_subdirectory(src/indexing)
add_subdirectory(src/gating)
if(TARGET cellerator_layout)
    target_link_libraries(cellerator_layout PUBLIC Cellerator::support)
endif()

# Compatibility umbrella for existing native consumers. New code links the
# owner matching its responsibility directly.
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
