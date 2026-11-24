if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/anndatapp/CMakeLists.txt")
    message(FATAL_ERROR "Initialize extern/anndatapp or configure with CELLERATOR_ENABLE_ANNDATA=OFF.")
endif()
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/anndatapp" "${PROJECT_BINARY_DIR}/extern/anndatapp")

add_library(cellerator_anndata STATIC cellerator.anndata.cc)
add_library(Cellerator::anndata ALIAS cellerator_anndata)
set_target_properties(cellerator_anndata PROPERTIES EXPORT_NAME anndata_binary CXX_EXTENSIONS OFF)
target_compile_features(cellerator_anndata PUBLIC cxx_std_20)
target_sources(cellerator_anndata PUBLIC FILE_SET CXX_MODULES FILES cellerator.anndata.cppm)
target_link_libraries(cellerator_anndata PUBLIC Cellerator::headers
    $<BUILD_INTERFACE:anndatapp::anndatapp> $<INSTALL_INTERFACE:anndatapp::anndatapp>)

install(TARGETS cellerator_anndata EXPORT CelleratorAnndataTargets
    FILE_SET CXX_MODULES DESTINATION "${CMAKE_INSTALL_DATADIR}/Cellerator/modules"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(EXPORT CelleratorAnndataTargets NAMESPACE Cellerator::
    CXX_MODULES_DIRECTORY anndata-cxx-modules DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator")
install(FILES "${PROJECT_SOURCE_DIR}/cmake/CelleratorModuleInterfaces.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator")
file(APPEND "${CELLERATOR_HOST_CONFIG}" [=[
if("anndata" IN_LIST Cellerator_FIND_COMPONENTS)
    if(Cellerator_FIND_REQUIRED_anndata)
        include(CMakeFindDependencyMacro)
        find_dependency(anndatapp 1.2 CONFIG)
    else()
        find_package(anndatapp 1.2 CONFIG QUIET)
    endif()
    if(anndatapp_FOUND AND TARGET anndatapp::anndatapp)
        if(NOT TARGET Cellerator::anndata)
            include("${CMAKE_CURRENT_LIST_DIR}/CelleratorAnndataTargets.cmake")
            include("${CMAKE_CURRENT_LIST_DIR}/CelleratorModuleInterfaces.cmake")
            cellerator_import_compiled_module(anndata Cellerator::anndata_binary
                Cellerator::headers anndatapp::anndatapp)
        endif()
        set(Cellerator_anndata_FOUND TRUE)
    endif()
endif()
]=])

if(CELLERATOR_BUILD_TESTS AND (ANNDATAPP_ENABLE_HDF5 OR ANNDATAPP_ENABLE_ZARR))
    add_executable(cellerator_anndata_test "${PROJECT_SOURCE_DIR}/tests/interop/anndata/read.cpp")
    set_target_properties(cellerator_anndata_test PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(cellerator_anndata_test PRIVATE Cellerator::anndata)
    add_test(NAME cellerator-anndata-read COMMAND cellerator_anndata_test)
endif()
