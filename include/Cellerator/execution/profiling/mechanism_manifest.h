#pragma once

#include <cstdint>

namespace cellerator::profiling {

enum class manifest_numerics : std::uint32_t { precise = 0, relaxed = 1 };
enum class manifest_value_mode : std::uint32_t { logical = 0, projection = 1 };
enum class manifest_order_mode : std::uint32_t { canonical = 0, packed = 1 };

struct prepared_stage_manifest_v1 {
    std::uint64_t stable_stage_id = 0;
    std::uint64_t stable_kernel_id = 0;
    std::uint32_t stage_kind = 0;
    std::uint32_t launch_count = 0;
    std::uint32_t threads_per_cta = 0;
    std::uint32_t warps_per_cta = 0;
    std::uint32_t static_shared_bytes = 0;
    std::uint32_t reserved = 0;
    char stable_name[48]{};
};

struct mechanism_work_v1 {
    std::uint64_t logical_interactions = 0;
    std::uint64_t physical_interactions = 0;
    std::uint64_t useful_interactions = 0;
    std::uint64_t padded_interactions = 0;
    std::uint64_t residual_interactions = 0;
    std::uint64_t group_count = 0;
    std::uint64_t tile_count = 0;
    std::uint64_t owner_work_count = 0;
};

struct mechanism_bytes_v1 {
    std::uint64_t relation = 0;
    std::uint64_t dense_input = 0;
    std::uint64_t output = 0;
    std::uint64_t value_pack = 0;
    std::uint64_t persistent = 0;
    std::uint64_t transient = 0;
};

struct prepared_mechanism_manifest_v1 {
    std::uint32_t version = 1;
    std::uint32_t flags = 0;
    std::uint64_t operation_id = 0;
    std::uint64_t candidate_id = 0;
    std::uint64_t provider_id = 0;
    std::uint64_t capability_id = 0;
    std::uint64_t projection_id = 0;
    std::uint64_t geometry_id = 0;
    manifest_value_mode value_mode = manifest_value_mode::logical;
    manifest_order_mode input_order = manifest_order_mode::canonical;
    manifest_order_mode output_order = manifest_order_mode::canonical;
    manifest_numerics numerics = manifest_numerics::precise;
    mechanism_work_v1 work{};
    mechanism_bytes_v1 bytes{};
    const prepared_stage_manifest_v1* stages = nullptr;
    std::uint32_t stage_count = 0;
    bool graph_capture_compatible = false;
    bool requires_measurement = true;
    std::uint16_t reserved = 0;
};

enum class manifest_status : std::uint32_t {
    success = 0, invalid_argument, invalid_identity, invalid_work,
    invalid_stage, duplicate_stage
};

manifest_status validate_prepared_mechanism_manifest_v1(
        const prepared_mechanism_manifest_v1& manifest) noexcept;

}  // namespace cellerator::profiling
