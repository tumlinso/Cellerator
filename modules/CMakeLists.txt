include(GNUInstallDirs)

add_library(cellerator_core_module STATIC)
add_library(Cellerator::core_module ALIAS cellerator_core_module)
set_target_properties(cellerator_core_module PROPERTIES EXPORT_NAME core_module_binary)
target_compile_features(cellerator_core_module PUBLIC cxx_std_20)
target_include_directories(cellerator_core_module PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_sources(cellerator_core_module PUBLIC FILE_SET CXX_MODULES
    FILES native/cellerator.core.ccm)

# Compile imports in the normal build with CMake's dependency scanner.
add_executable(cellerator_module_contract "${PROJECT_SOURCE_DIR}/tests/modules/module_contract.cc")
target_link_libraries(cellerator_module_contract PRIVATE Cellerator::core_module anndata::anndata)

install(TARGETS cellerator_core_module EXPORT CelleratorCoreModulesTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/Cellerator/modules)
install(EXPORT CelleratorCoreModulesTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator
    CXX_MODULES_DIRECTORY core-modules)
install(FILES "${PROJECT_SOURCE_DIR}/cmake/CelleratorModuleInterfaces.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)

if(DEFINED CELLERATOR_HOST_CONFIG)
    file(APPEND "${CELLERATOR_HOST_CONFIG}"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorCoreModulesTargets.cmake\")\n"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorModuleInterfaces.cmake\")\n"
        "cellerator_import_module(core_module Cellerator::core_module_binary)\n"
        "set(Cellerator_core_module_FOUND TRUE)\n")
endif()
if(CELLERATOR_BUILD_TESTS)
    enable_testing()
    add_test(NAME cellerator_module_contract COMMAND cellerator_module_contract)
endif()
