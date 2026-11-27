# CUDA operation owners are included at project scope.
include(src/ops/relation/CMakeLists.txt)
include(src/layout/projection/cuda_targets.cmake)
include(src/execution/planning/providers/targets.cmake)
include(src/ops/relation/algebra_targets.cmake)
include(src/execution/program/training/cuda_targets.cmake)
include(src/execution/planning/candidates/targets.cmake)
include(src/execution/preparation/relation_targets.cmake)
