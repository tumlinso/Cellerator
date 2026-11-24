cmake_minimum_required(VERSION 3.28)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(CelleratorBiology LANGUAGES CXX)
    option(CELLERATOR_ENABLE_MODULES "Build the Cellerator Biology module" OFF)
    option(CELLERATOR_BUILD_TESTS "Build the Biology checks" ON)
    if(CELLERATOR_ENABLE_MODULES)
        include(../../cmake/CelleratorModuleInterfaces.cmake)
        cellerator_require_module_scanner()
    endif()
endif()
include(GNUInstallDirs)

add_library(cellerator_gene_annotation OBJECT src/gene_annotation.cc)
set_target_properties(cellerator_gene_annotation PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_compile_features(cellerator_gene_annotation PRIVATE cxx_std_20)
target_include_directories(cellerator_gene_annotation PRIVATE include)

add_library(cellerator_biology STATIC src/gene_catalog.cc
    $<TARGET_OBJECTS:cellerator_gene_annotation>)
add_library(Cellerator::biology ALIAS cellerator_biology)
set_target_properties(cellerator_biology PROPERTIES EXPORT_NAME biology CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_compile_features(cellerator_biology PUBLIC cxx_std_20)
target_include_directories(cellerator_biology PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set(CELLERATOR_BIOLOGY_CATALOG_MANIFEST "" CACHE FILEPATH "Release manifest for the built-in Biology gene catalog")
set(CELLERATOR_BIOLOGY_ANNOTATION "" CACHE FILEPATH "Local annotation file for the selected Biology release")
if(CELLERATOR_BIOLOGY_CATALOG_MANIFEST OR CELLERATOR_BUILD_TESTS)
    include(catalogs/CatalogGeneration.cmake)
endif()
if(CELLERATOR_BIOLOGY_CATALOG_MANIFEST)
    cellerator_generate_gene_catalog(cellerator_biology "${CMAKE_CURRENT_BINARY_DIR}/generated"
        "${CELLERATOR_BIOLOGY_CATALOG_MANIFEST}" "${CELLERATOR_BIOLOGY_ANNOTATION}")
endif()
if(CELLERATOR_ENABLE_MODULES)
    install(FILES ../../cmake/CelleratorModuleInterfaces.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
    add_library(cellerator_biology_interface STATIC)
    set_target_properties(cellerator_biology_interface PROPERTIES EXPORT_NAME biology_interface_binary CXX_EXTENSIONS OFF)
    target_compile_features(cellerator_biology_interface PUBLIC cxx_std_20)
    target_include_directories(cellerator_biology_interface PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_sources(cellerator_biology_interface PUBLIC FILE_SET CXX_MODULES
        BASE_DIRS src FILES src/cellerator.biology.ccm)
    target_link_libraries(cellerator_biology PUBLIC $<BUILD_INTERFACE:cellerator_biology_interface>)
    install(TARGETS cellerator_biology_interface EXPORT CelleratorBiologyModuleTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/Cellerator/modules)
    install(EXPORT CelleratorBiologyModuleTargets NAMESPACE Cellerator::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator CXX_MODULES_DIRECTORY biology-modules)
endif()
install(TARGETS cellerator_biology EXPORT CelleratorBiologyTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT CelleratorBiologyTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
install(DIRECTORY include/Cellerator DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES BiologyComponent.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)

if(CELLERATOR_BUILD_TESTS)
    enable_testing()
    add_executable(cellerator_biology_test tests/gene_annotation.cc)
    target_link_libraries(cellerator_biology_test PRIVATE Cellerator::biology)
    if(CELLERATOR_ENABLE_MODULES)
        target_compile_definitions(cellerator_biology_test PRIVATE CELLERATOR_BIOLOGY_TEST_MODULE)
        set_target_properties(cellerator_biology_test PROPERTIES CXX_SCAN_FOR_MODULES ON CXX_EXTENSIONS OFF)
    endif()
    add_test(NAME cellerator-biology-annotation COMMAND cellerator_biology_test)

    add_executable(cellerator_gene_catalog_test tests/gene_catalog.cc src/gene_catalog.cc)
    target_link_libraries(cellerator_gene_catalog_test PRIVATE Cellerator::biology)
    cellerator_generate_gene_catalog(cellerator_gene_catalog_test
        "${CMAKE_CURRENT_BINARY_DIR}/test_generated"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/catalog/release.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/catalog/reference.gff3")
    if(CELLERATOR_ENABLE_MODULES)
        target_compile_definitions(cellerator_gene_catalog_test PRIVATE CELLERATOR_BIOLOGY_TEST_MODULE)
        set_target_properties(cellerator_gene_catalog_test PROPERTIES CXX_SCAN_FOR_MODULES ON CXX_EXTENSIONS OFF)
    endif()
    add_test(NAME cellerator-biology-gene-catalog COMMAND cellerator_gene_catalog_test)
endif()
