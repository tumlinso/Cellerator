#include "Cellerator/profiling/resource_receipt.h"

namespace cellerator::profiling {

std::uint64_t make_profiling_correlation_id_v1(
        std::uint64_t candidate_id, std::uint64_t stage_id,
        std::uint64_t kernel_id) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    const std::uint64_t values[3]{candidate_id, stage_id, kernel_id};
    for (const auto value : values) {
        for (std::uint32_t byte = 0; byte < 8; ++byte) {
            hash ^= static_cast<std::uint8_t>(value >> (byte * 8U));
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

bool validate_cold_resource_receipt_v1(
        const cold_resource_receipt_v1& receipt) noexcept {
    const auto& identity = receipt.identity;
    return receipt.version == 1 && receipt.status == 0 &&
           receipt.queried_cold && !receipt.kernel_executed &&
           identity.candidate_id != 0 && identity.stage_id != 0 &&
           identity.kernel_id != 0 && identity.candidate_name[0] != 0 &&
           identity.stage_name[0] != 0 && identity.kernel_symbol[0] != 0 &&
           identity.correlation_id == make_profiling_correlation_id_v1(
                   identity.candidate_id, identity.stage_id,
                   identity.kernel_id) &&
           receipt.build.build_id != 0 &&
           receipt.build.device_identity != 0 &&
           receipt.maximum_threads_per_block != 0;
}

}  // namespace cellerator::profiling
