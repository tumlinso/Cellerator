#pragma once

#include <cstdint>

namespace cellerator::geometry::optimizer::device {

enum class device_assisted_disposition : std::uint32_t {
    implemented_experimental = 0,
    evaluated_not_retained,
    evaluated_not_adopted,
};

struct device_assisted_optimizer_disposition_v1 {
    std::uint32_t version = 1;
    device_assisted_disposition disposition =
            device_assisted_disposition::implemented_experimental;
    bool cold_path_only = true;
    bool requires_measurement = true;
    bool production_promoted = false;
    bool steady_state_allowed = false;
    std::uint32_t cuda_version = 0;
    std::uint32_t compute_major = 0;
    std::uint32_t compute_minor = 0;
    std::uint64_t parity_score_count = 0;
    std::uint64_t parity_census_count = 0;
    char resource_receipt_uuid[37]{};
    char validated_device[32]{};
};

device_assisted_optimizer_disposition_v1
built_in_device_assisted_disposition_v1() noexcept;

bool validate_device_assisted_disposition_v1(
        const device_assisted_optimizer_disposition_v1& disposition) noexcept;

}  // namespace cellerator::geometry::optimizer::device
