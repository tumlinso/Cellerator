include(GNUInstallDirs)
add_library(cellerator_spmm_reference STATIC
    operation.cc operation_signature.cc referee.cc)
add_library(Cellerator::spmm_reference ALIAS cellerator_spmm_reference)
target_include_directories(cellerator_spmm_reference PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(cellerator_spmm_reference PUBLIC cxx_std_20)
target_compile_definitions(cellerator_spmm_reference PUBLIC
    CELLERATOR_ENABLE_CUDA=$<BOOL:${CELLERATOR_HAS_CUDA}>)
set_target_properties(cellerator_spmm_reference PROPERTIES
    EXPORT_NAME spmm_reference POSITION_INDEPENDENT_CODE ON)
if(CELLERATOR_HAS_CUDA)
    target_link_libraries(cellerator_spmm_reference PUBLIC CUDA::cudart)
    if(DEFINED CELLERATOR_HOST_CONFIG)
        file(APPEND "${CELLERATOR_HOST_CONFIG}"
            "include(CMakeFindDependencyMacro)\nfind_dependency(CUDAToolkit)\n")
    endif()
endif()
install(TARGETS cellerator_spmm_reference EXPORT CelleratorReferenceTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT CelleratorReferenceTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
if(DEFINED CELLERATOR_HOST_CONFIG)
    file(APPEND "${CELLERATOR_HOST_CONFIG}"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorReferenceTargets.cmake\")\n"
        "set(Cellerator_spmm_reference_FOUND TRUE)\n")
endif()
