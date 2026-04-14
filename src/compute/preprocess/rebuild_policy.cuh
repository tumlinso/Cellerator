#pragma once

#include <cstddef>

namespace cellerator {
namespace compute {
namespace preprocess {

struct alignas(16) rebuild_policy {
    unsigned int expected_reuse_passes = 8u;
    double dead_byte_safety_factor = 1.25;
    double min_live_fill_ratio = 0.55;
    int allow_auto_rebuild = 1;
};

struct alignas(16) rebuild_decision {
    int should_rebuild = 0;
    double live_fill_ratio = 1.0;
    std::size_t dead_value_bytes_per_pass = 0u;
    std::size_t estimated_rebuild_bytes = 0u;
    double amortization_ratio = 0.0;
};

template<typename StatsT>
static inline rebuild_decision evaluate_rebuild_policy(const StatsT &stats,
                                                       std::size_t estimated_rebuild_bytes,
                                                       const rebuild_policy &policy = rebuild_policy()) {
    rebuild_decision out;
    const double projected_dead_bytes = (double) stats.dead_value_bytes * (double) policy.expected_reuse_passes;
    const double guarded_rebuild_bytes = (double) estimated_rebuild_bytes * policy.dead_byte_safety_factor;

    out.live_fill_ratio = stats.live_fill_ratio;
    out.dead_value_bytes_per_pass = stats.dead_value_bytes;
    out.estimated_rebuild_bytes = estimated_rebuild_bytes;
    out.amortization_ratio = guarded_rebuild_bytes > 0.0 ? projected_dead_bytes / guarded_rebuild_bytes : 0.0;

    if (!policy.allow_auto_rebuild) return out;
    if (stats.live_nnz == 0u || stats.output_rows == 0u || stats.output_cols == 0u) {
        out.should_rebuild = 0;
        return out;
    }

    out.should_rebuild = (stats.live_fill_ratio <= policy.min_live_fill_ratio)
        && (projected_dead_bytes > guarded_rebuild_bytes);
    return out;
}

template<typename StatsT>
static inline std::size_t estimate_filtered_compressed_bytes(const StatsT &stats) {
    return (std::size_t) (stats.output_rows + 1u) * sizeof(unsigned int)
        + (std::size_t) stats.live_nnz * sizeof(unsigned int)
        + (std::size_t) stats.live_nnz * sizeof(unsigned short);
}

} // namespace preprocess
} // namespace compute
} // namespace cellerator
