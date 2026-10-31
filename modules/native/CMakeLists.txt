cmake_minimum_required(VERSION 3.28)

project(CelleratorNNModules VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(Cellerator CONFIG REQUIRED COMPONENTS nn)
if(NOT TARGET Cellerator::nn)
    message(FATAL_ERROR
        "The installed Cellerator package does not export Cellerator::nn")
endif()

add_library(cellerator_nn_modules STATIC)
add_library(CelleratorNNModules::nn ALIAS cellerator_nn_modules)
set_target_properties(cellerator_nn_modules PROPERTIES EXPORT_NAME nn)
target_compile_features(cellerator_nn_modules PUBLIC cxx_std_20)
target_sources(cellerator_nn_modules
    PUBLIC
        FILE_SET CXX_MODULES FILES
            cellerator.nn.ccm)
target_link_libraries(cellerator_nn_modules
    PUBLIC
        Cellerator::nn)

install(TARGETS cellerator_nn_modules
    EXPORT CelleratorNNModulesTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FILE_SET CXX_MODULES
        DESTINATION ${CMAKE_INSTALL_DATADIR}/CelleratorNNModules/modules
    CXX_MODULES_BMI
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cellerator/nn/modules)
install(EXPORT CelleratorNNModulesTargets
    FILE CelleratorNNModulesTargets.cmake
    NAMESPACE CelleratorNNModules::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CelleratorNNModules
    CXX_MODULES_DIRECTORY modules)

configure_package_config_file(
    CelleratorNNModulesConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/CelleratorNNModulesConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CelleratorNNModules)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CelleratorNNModulesConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CelleratorNNModulesConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CelleratorNNModulesConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CelleratorNNModules)
