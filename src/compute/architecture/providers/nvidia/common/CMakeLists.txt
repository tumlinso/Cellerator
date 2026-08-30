# Target-neutral NVIDIA physical realization used by source-linked providers.
# The including compute manifest owns the concrete library target.
set(CELLERATOR_NVIDIA_COMMON_PROVIDER_SOURCES
    src/compute/architecture/providers/nvidia/common/exact_rectangle_census.cc
    src/compute/architecture/providers/nvidia/common/mma_projection_realization.cc
    src/compute/architecture/providers/nvidia/common/mma_target_refinement.cc
    src/compute/architecture/providers/nvidia/common/sm70_grouping.cc
)
