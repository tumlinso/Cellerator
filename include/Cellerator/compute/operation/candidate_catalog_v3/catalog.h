#pragma once
#include <cstdint>

namespace cellerator::compute::operation::catalog_v3 {

enum class candidate_class : std::uint32_t { production = 0, experimental = 1 };
enum class numerical_mode : std::uint32_t { precise = 0, relaxed = 1 };

struct candidate_identity_v3 {
    std::uint64_t candidate_id = 0;
    std::uint64_t provider_id = 0;
    std::uint64_t device_class_id = 0;
    std::uint64_t projection_type_id = 0;
    std::uint64_t capability_id = 0;
    std::uint64_t operation_id = 0;
    std::uint32_t width_min = 0;
    std::uint32_t width_max = 0;
    numerical_mode numerics = numerical_mode::precise;
    candidate_class classification = candidate_class::production;
    bool requires_measurement = false;
    std::uint8_t reserved[7]{};
};

struct candidate_resource_v3 {
    std::uint64_t persistent_bytes = 0;
    std::uint64_t transient_bytes = 0;
    std::uint32_t threads_per_cta = 0;
    std::uint32_t shared_bytes_per_cta = 0;
};

struct candidate_stage_v3 {
    std::uint64_t stage_id = 0;
    std::uint64_t kernel_id = 0;
    std::uint32_t stage_kind = 0;
    std::uint32_t launch_count = 0;
    char stable_name[48]{};
};

struct candidate_descriptor_v3 {
    candidate_identity_v3 identity{};
    const candidate_stage_v3* stages = nullptr;
    std::uint32_t stage_count = 0;
    std::uint32_t reserved = 0;
    candidate_resource_v3 resources{};
};

struct candidate_catalog_view_v3 {
    const candidate_descriptor_v3* candidates = nullptr;
    std::uint64_t candidate_count = 0;
};

enum class catalog_status : std::uint32_t {
    success = 0, invalid_argument, duplicate_identity, invalid_width,
    invalid_stage, invalid_resource
};

catalog_status validate_candidate_catalog_v3(
        const candidate_catalog_view_v3& catalog) noexcept;

}  // namespace cellerator::compute::operation::catalog_v3
