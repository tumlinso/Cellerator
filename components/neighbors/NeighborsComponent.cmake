include(CMakeFindDependencyMacro)
find_dependency(CUDAToolkit)
include("${CMAKE_CURRENT_LIST_DIR}/CelleratorNeighborsTargets.cmake")
set(Cellerator_neighbors_FOUND TRUE)
