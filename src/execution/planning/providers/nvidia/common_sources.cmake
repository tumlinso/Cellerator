# Target-neutral NVIDIA physical realization used by source-linked providers.
# The including compute manifest owns the concrete library target.
set(CELLERATOR_NVIDIA_COMMON_PROVIDER_SOURCES
    src/geometry/discovery/exact_rectangle_census.cc
    src/layout/projection/mma_projection_realization.cc
    src/execution/planning/providers/nvidia/mma_target_refinement.cc
    src/geometry/packing/sm70_grouping.cc
)
