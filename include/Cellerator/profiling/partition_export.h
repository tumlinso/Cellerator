#pragma once

#include <cstdint>

namespace cellerator::profiling {

struct partition_index_export_v1 {
    std::uint64_t partition_id = 0;
    std::uint64_t global_count = 0;
    const std::uint64_t* local_to_global = nullptr;
    std::uint64_t local_count = 0;
};

struct export_stage_v1 {
    std::uint64_t stage_id = 0;
    std::uint64_t kernel_id = 0;
    std::uint32_t stage_kind = 0;
    std::uint32_t launch_count = 0;
};

struct communication_boundary_v1 {
    std::uint64_t peer_partition_id = 0;
    std::uint64_t send_elements = 0;
    std::uint64_t receive_elements = 0;
};

struct generic_execution_export_v1 {
    std::uint32_t version = 1;
    std::uint32_t flags = 0;
    std::uint64_t semantic_geometry_id = 0;
    std::uint64_t projection_id = 0;
    std::uint64_t candidate_id = 0;
    std::uint64_t provider_id = 0;
    std::uint64_t capability_id = 0;
    std::uint64_t input_order_id = 0;
    std::uint64_t output_order_id = 0;
    partition_index_export_v1 partition{};
    const export_stage_v1* stages = nullptr;
    std::uint64_t stage_count = 0;
    const communication_boundary_v1* boundaries = nullptr;
    std::uint64_t boundary_count = 0;
    std::uint64_t persistent_bytes = 0;
    std::uint64_t transient_bytes = 0;
    std::uint64_t transform_bytes = 0;
    std::uint32_t minimum_compute_major = 0;
    std::uint32_t minimum_compute_minor = 0;
    bool graph_capture_compatible = false;
    std::uint8_t reserved[7]{};
};

enum class export_status : std::uint32_t {
    success = 0, invalid_argument, invalid_identity, invalid_index,
    invalid_stage, invalid_boundary
};

export_status validate_generic_execution_export_v1(
        const generic_execution_export_v1& value) noexcept;

}  // namespace cellerator::profiling
