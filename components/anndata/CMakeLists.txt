cmake_minimum_required(VERSION 3.28)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(CelleratorAnnData LANGUAGES CXX)
    option(CELLERATOR_BUILD_TESTS "Build the AnnData reader check" ON)
    find_package(Cellerator CONFIG REQUIRED)
    find_package(anndatapp 1.2 CONFIG REQUIRED)
endif()
include(GNUInstallDirs)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
if(NOT TARGET anndatapp::anndatapp)
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/packages/anndatapp/CMakeLists.txt")
        message(FATAL_ERROR "Initialize packages/anndatapp or configure with CELLERATOR_ENABLE_ANNDATA=OFF.")
    endif()
    add_subdirectory("${PROJECT_SOURCE_DIR}/packages/anndatapp" "${PROJECT_BINARY_DIR}/packages/anndatapp")
endif()

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
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/CelleratorModuleInterfaces.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator")
install(FILES AnnDataComponent.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)

if(CELLERATOR_BUILD_TESTS AND (ANNDATAPP_ENABLE_HDF5 OR ANNDATAPP_ENABLE_ZARR OR anndatapp_FOUND))
    enable_testing()
    add_executable(cellerator_anndata_test "${CMAKE_CURRENT_SOURCE_DIR}/tests/read.cpp")
    set_target_properties(cellerator_anndata_test PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(cellerator_anndata_test PRIVATE Cellerator::anndata)
    add_test(NAME cellerator-anndata-read COMMAND cellerator_anndata_test)
endif()
