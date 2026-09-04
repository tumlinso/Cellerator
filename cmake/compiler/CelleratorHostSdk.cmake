include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(libCellerator STATIC
    src/compiler/driver/deliver_the_driver_passthrough_milestone.cc
    src/compiler/tooling/freeze_the_celleratord_architecture.cc)
add_library(Cellerator::libCellerator ALIAS libCellerator)
target_include_directories(libCellerator PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(libCellerator PUBLIC cxx_std_17)
set_target_properties(libCellerator PROPERTIES OUTPUT_NAME Cellerator)

add_executable(cellerator tools/cellerator/deliver_the_driver_passthrough_milestone.cc)
target_link_libraries(cellerator PRIVATE libCellerator)
add_executable(celleratord tools/celleratord/freeze_the_celleratord_architecture.cc)
target_link_libraries(celleratord PRIVATE libCellerator)

install(TARGETS libCellerator cellerator celleratord
    EXPORT CelleratorPartOneTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY stdlib/ DESTINATION ${CMAKE_INSTALL_DATADIR}/cellerator/stdlib)
install(DIRECTORY profiles/ DESTINATION ${CMAKE_INSTALL_DATADIR}/cellerator/profiles)
install(EXPORT CelleratorPartOneTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)

set(CELLERATOR_HOST_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/CelleratorConfig.cmake")
file(WRITE "${CELLERATOR_HOST_CONFIG}"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorPartOneTargets.cmake\")\n"
    "set(Cellerator_HAS_COMPILER TRUE)\nset(Cellerator_HAS_RUNTIME TRUE)\n"
    "set(Cellerator_HAS_CUDA_BACKEND FALSE)\n")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CelleratorConfigVersion.cmake"
    VERSION 1.0.0 COMPATIBILITY SameMajorVersion)
install(FILES "${CELLERATOR_HOST_CONFIG}"
    "${CMAKE_CURRENT_BINARY_DIR}/CelleratorConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)

if(CELLERATOR_BUILD_TESTS)
    enable_testing()
    include(cmake/compiler/CelleratorPartOneAcceptance.cmake)
endif()
