# Volta source-linked provider and the first exact hybrid vertical slice.
# Advanced relation-algebra operations are integrated by CE-GEO-122.
set(CELLERATOR_NVIDIA_SM70_PROVIDER_SOURCES
    src/compute/architecture/providers/nvidia/sm70/catalog_fragment.cc
    src/compute/architecture/providers/nvidia/sm70/provider.cc
    src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cu
    src/compute/architecture/providers/nvidia/sm70/value_pack.cu
    src/compute/architecture/providers/nvidia/sm70/residual.cu
    src/compute/architecture/providers/nvidia/sm70/relation_apply_hybrid.cu
)
