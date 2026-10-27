#pragma once

#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

namespace cellerator::compute::projection {

physical_image_build_status_v1 build_physical_mma_hybrid_image_probe_v1(
    const physical_mma_hybrid_build_request_v1& request,
    std::uint32_t minimum_tile_edges,
    std::vector<std::uint8_t>* output) noexcept;

} // namespace cellerator::compute::projection
