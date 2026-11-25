include("${CMAKE_CURRENT_LIST_DIR}/CelleratorVelocityTargets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CelleratorModuleInterfaces.cmake")
cellerator_import_compiled_module(velocity Cellerator::velocity_binary Cellerator::headers)
set(Cellerator_velocity_FOUND TRUE)
