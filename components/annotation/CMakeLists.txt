cmake_minimum_required(VERSION 3.28)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(CelleratorAnnotation LANGUAGES CXX)
    option(CELLERATOR_ENABLE_MODULES "Build the Cellerator Annotation module" OFF)
    option(CELLERATOR_BUILD_TESTS "Build the Annotation checks" ON)
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

add_library(cellerator_annotation STATIC src/gene_catalog.cc
    $<TARGET_OBJECTS:cellerator_gene_annotation>)
add_library(Cellerator::annotation ALIAS cellerator_annotation)
set_target_properties(cellerator_annotation PROPERTIES EXPORT_NAME annotation CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_compile_features(cellerator_annotation PUBLIC cxx_std_20)
target_include_directories(cellerator_annotation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set(CELLERATOR_ANNOTATION_CATALOG_MANIFEST "" CACHE FILEPATH "Release manifest for the built-in gene catalog")
set(CELLERATOR_ANNOTATION_FILE "" CACHE FILEPATH "Local annotation file for the selected catalog release")
if(CELLERATOR_ANNOTATION_CATALOG_MANIFEST OR CELLERATOR_BUILD_TESTS)
    include(catalogs/CatalogGeneration.cmake)
endif()
if(CELLERATOR_ANNOTATION_CATALOG_MANIFEST)
    cellerator_generate_gene_catalog(cellerator_annotation "${CMAKE_CURRENT_BINARY_DIR}/generated"
        "${CELLERATOR_ANNOTATION_CATALOG_MANIFEST}" "${CELLERATOR_ANNOTATION_FILE}")
endif()
if(CELLERATOR_ENABLE_MODULES)
    install(FILES ../../cmake/CelleratorModuleInterfaces.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
    add_library(cellerator_annotation_interface STATIC)
    set_target_properties(cellerator_annotation_interface PROPERTIES EXPORT_NAME annotation_interface_binary CXX_EXTENSIONS OFF)
    target_compile_features(cellerator_annotation_interface PUBLIC cxx_std_20)
    target_include_directories(cellerator_annotation_interface PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_sources(cellerator_annotation_interface PUBLIC FILE_SET CXX_MODULES
        BASE_DIRS src FILES src/cellerator.annotation.ccm)
    target_link_libraries(cellerator_annotation PUBLIC $<BUILD_INTERFACE:cellerator_annotation_interface>)
    install(TARGETS cellerator_annotation_interface EXPORT CelleratorAnnotationModuleTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/Cellerator/modules)
    install(EXPORT CelleratorAnnotationModuleTargets NAMESPACE Cellerator::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator CXX_MODULES_DIRECTORY annotation-modules)
endif()
install(TARGETS cellerator_annotation EXPORT CelleratorAnnotationTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT CelleratorAnnotationTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
install(DIRECTORY include/Cellerator DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES AnnotationComponent.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)

if(CELLERATOR_BUILD_TESTS)
    enable_testing()
    add_executable(cellerator_annotation_test tests/gene_annotation.cc)
    target_link_libraries(cellerator_annotation_test PRIVATE Cellerator::annotation)
    if(CELLERATOR_ENABLE_MODULES)
        target_compile_definitions(cellerator_annotation_test PRIVATE CELLERATOR_ANNOTATION_TEST_MODULE)
        set_target_properties(cellerator_annotation_test PROPERTIES CXX_SCAN_FOR_MODULES ON CXX_EXTENSIONS OFF)
    endif()
    add_test(NAME cellerator-annotation-gene-reader COMMAND cellerator_annotation_test)

    add_executable(cellerator_gene_catalog_test tests/gene_catalog.cc src/gene_catalog.cc)
    target_link_libraries(cellerator_gene_catalog_test PRIVATE Cellerator::annotation)
    cellerator_generate_gene_catalog(cellerator_gene_catalog_test
        "${CMAKE_CURRENT_BINARY_DIR}/test_generated"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/catalog/release.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/catalog/reference.gff3")
    if(CELLERATOR_ENABLE_MODULES)
        target_compile_definitions(cellerator_gene_catalog_test PRIVATE CELLERATOR_ANNOTATION_TEST_MODULE)
        set_target_properties(cellerator_gene_catalog_test PROPERTIES CXX_SCAN_FOR_MODULES ON CXX_EXTENSIONS OFF)
    endif()
    add_test(NAME cellerator-annotation-gene-catalog COMMAND cellerator_gene_catalog_test)
endif()
