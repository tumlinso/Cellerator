# Installed reference algorithms with a Cellerator-owned binary16 extension.
include_guard(GLOBAL)
option(CELLERATOR_ENABLE_VELOCITY "Use installed Velocity++ and build CPU FP16 support" ON)
if(NOT CELLERATOR_ENABLE_VELOCITY)
    return()
endif()
find_package(velocitypp 0.1.1 CONFIG REQUIRED)
if(NOT TARGET Cellerator::binary16)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../src/ops/numeric/binary16"
        "${CMAKE_CURRENT_BINARY_DIR}/src/ops/numeric/binary16")
endif()
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../src/ops/velocity"
    "${CMAKE_CURRENT_BINARY_DIR}/src/ops/velocity")
if(CELLERATOR_VELOCITY_BUILD_TESTS)
    enable_testing()
endif()

install(TARGETS cellerator_velocity EXPORT CelleratorVelocityTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/Cellerator/modules)
install(EXPORT CelleratorVelocityTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator
    CXX_MODULES_DIRECTORY velocity-modules)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/CelleratorVelocityModule.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
file(APPEND "${CELLERATOR_HOST_CONFIG}"
    "include(CMakeFindDependencyMacro)\n"
    "find_dependency(velocitypp 0.1.1 CONFIG)\n"
    "find_dependency(CelleratorBinary16 CONFIG)\n"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorVelocityTargets.cmake\")\n"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorVelocityModule.cmake\")\n"
    "cellerator_import_velocity()\n"
    "set(Cellerator_velocity_FOUND TRUE)\n")
