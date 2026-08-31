#include "Cellerator/geometry/optimizer/device/device_assisted_disposition.h"

#include <cstdint>

namespace cellerator::geometry::optimizer::device {
namespace {

bool valid_uuid(const char* text) noexcept {
    for (std::uint32_t index = 0; index < 36; ++index) {
        const char value = text[index];
        if (index == 8 || index == 13 || index == 18 || index == 23) {
            if (value != '-') return false;
        } else if (!((value >= '0' && value <= '9') ||
                     (value >= 'a' && value <= 'f'))) {
            return false;
        }
    }
    return text[36] == 0;
}

}  // namespace

device_assisted_optimizer_disposition_v1
built_in_device_assisted_disposition_v1() noexcept {
    device_assisted_optimizer_disposition_v1 result{};
    result.cuda_version = 12000;
    result.compute_major = 7;
    result.compute_minor = 0;
    result.parity_score_count = 3;
    result.parity_census_count = 2;
    constexpr char receipt[] = "f7cbb206-3a50-4d46-89f3-99c2c2d21da2";
    constexpr char device[] = "Tesla V100-SXM2-16GB";
    for (std::uint32_t index = 0; index < sizeof(receipt); ++index) {
        result.resource_receipt_uuid[index] = receipt[index];
    }
    for (std::uint32_t index = 0; index < sizeof(device); ++index) {
        result.validated_device[index] = device[index];
    }
    return result;
}

bool validate_device_assisted_disposition_v1(
        const device_assisted_optimizer_disposition_v1& disposition) noexcept {
    return disposition.version == 1 &&
           disposition.disposition ==
                   device_assisted_disposition::implemented_experimental &&
           disposition.cold_path_only && disposition.requires_measurement &&
           !disposition.production_promoted &&
           !disposition.steady_state_allowed &&
           disposition.cuda_version == 12000 &&
           disposition.compute_major == 7 && disposition.compute_minor == 0 &&
           disposition.parity_score_count != 0 &&
           disposition.parity_census_count != 0 &&
           valid_uuid(disposition.resource_receipt_uuid) &&
           disposition.validated_device[0] != 0;
}

}  // namespace cellerator::geometry::optimizer::device
