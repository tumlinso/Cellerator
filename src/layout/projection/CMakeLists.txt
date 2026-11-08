include(GNUInstallDirs)
# Borrowed CP-BP projection construction has no runtime or reference dependency.
add_library(cellerator_physical_projection STATIC
    csr.cc ../native_tile_view.cc bell.cc bell_candidates.cc)
add_library(Cellerator::physical_projection ALIAS cellerator_physical_projection)
target_include_directories(cellerator_physical_projection PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(cellerator_physical_projection PUBLIC cxx_std_20)
target_compile_definitions(cellerator_physical_projection PUBLIC
    CELLERATOR_ENABLE_CUDA=$<BOOL:${CELLERATOR_HAS_CUDA}>)
set_target_properties(cellerator_physical_projection PROPERTIES
    EXPORT_NAME physical_projection POSITION_INDEPENDENT_CODE ON)
set(_projection_exports cellerator_physical_projection)
if(CELLERATOR_HAS_CUDA)
    target_link_libraries(cellerator_physical_projection PUBLIC CUDA::cudart)
    add_library(cellerator_packed_dense_operand STATIC packed_dense_operand.cu)
    add_library(Cellerator::packed_dense_operand ALIAS cellerator_packed_dense_operand)
    target_link_libraries(cellerator_packed_dense_operand PUBLIC
        Cellerator::physical_projection CUDA::cudart)
    set_target_properties(cellerator_packed_dense_operand PROPERTIES
        EXPORT_NAME packed_dense_operand CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
    list(APPEND _projection_exports cellerator_packed_dense_operand)
endif()
install(TARGETS ${_projection_exports} EXPORT CelleratorProjectionTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT CelleratorProjectionTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
if(DEFINED CELLERATOR_HOST_CONFIG)
    file(APPEND "${CELLERATOR_HOST_CONFIG}"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorProjectionTargets.cmake\")\n"
        "set(Cellerator_physical_projection_FOUND TRUE)\n"
        "if(TARGET Cellerator::packed_dense_operand)\n"
        "  set(Cellerator_packed_dense_operand_FOUND TRUE)\nendif()\n")
endif()
unset(_projection_exports)
