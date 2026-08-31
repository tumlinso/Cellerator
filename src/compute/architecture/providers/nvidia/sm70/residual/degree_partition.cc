#include "Cellerator/compute/architecture/providers/nvidia/sm70/residual/degree_partition.h"

#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::providers::nvidia::sm70::residual {
namespace {

residual_partition_status validate(
        const residual_support_view_v1& support,
        const residual_degree_thresholds_v1& thresholds) noexcept {
    if ((support.row_count != 0 && support.row_offsets == nullptr) ||
        thresholds.thread_max_degree >= thresholds.warp_max_degree) {
        return thresholds.thread_max_degree >= thresholds.warp_max_degree
                ? residual_partition_status::invalid_thresholds
                : residual_partition_status::invalid_argument;
    }
    if (support.row_count == 0) return support.edge_count == 0
            ? residual_partition_status::success
            : residual_partition_status::invalid_offsets;
    if (support.row_offsets[0] != 0 ||
        support.row_offsets[support.row_count] != support.edge_count) {
        return residual_partition_status::invalid_offsets;
    }
    for (std::uint64_t row = 0; row < support.row_count; ++row) {
        if (support.row_offsets[row] > support.row_offsets[row + 1]) {
            return residual_partition_status::invalid_offsets;
        }
    }
    return residual_partition_status::success;
}

}  // namespace

residual_partition_status query_residual_partition_v1(
        const residual_support_view_v1& support,
        const residual_degree_thresholds_v1& thresholds,
        residual_partition_requirements_v1* requirements) noexcept {
    if (requirements == nullptr) return residual_partition_status::invalid_argument;
    *requirements = {};
    const auto status = validate(support, thresholds);
    if (status != residual_partition_status::success) return status;
    for (std::uint64_t row = 0; row < support.row_count; ++row) {
        const auto degree = support.row_offsets[row + 1] - support.row_offsets[row];
        if (degree <= thresholds.thread_max_degree) ++requirements->thread_rows;
        else if (degree <= thresholds.warp_max_degree) ++requirements->warp_rows;
        else ++requirements->cta_rows;
    }
    requirements->covered_rows = support.row_count;
    requirements->covered_edges = support.edge_count;
    return residual_partition_status::success;
}

residual_partition_status build_residual_partition_v1(
        const residual_support_view_v1& support,
        const residual_degree_thresholds_v1& thresholds,
        residual_degree_partition_v1* partition) noexcept {
    if (partition == nullptr) return residual_partition_status::invalid_argument;
    residual_partition_requirements_v1 required{};
    const auto status = query_residual_partition_v1(support, thresholds, &required);
    if (status != residual_partition_status::success) return status;
    if (partition->thread_capacity < required.thread_rows ||
        partition->warp_capacity < required.warp_rows ||
        partition->cta_capacity < required.cta_rows ||
        (required.thread_rows != 0 && partition->thread_rows == nullptr) ||
        (required.warp_rows != 0 && partition->warp_rows == nullptr) ||
        (required.cta_rows != 0 && partition->cta_rows == nullptr)) {
        return residual_partition_status::insufficient_capacity;
    }
    partition->thread_count = 0;
    partition->warp_count = 0;
    partition->cta_count = 0;
    for (std::uint64_t row = 0; row < support.row_count; ++row) {
        const auto degree = support.row_offsets[row + 1] - support.row_offsets[row];
        if (degree <= thresholds.thread_max_degree) {
            partition->thread_rows[partition->thread_count++] = row;
        } else if (degree <= thresholds.warp_max_degree) {
            partition->warp_rows[partition->warp_count++] = row;
        } else {
            partition->cta_rows[partition->cta_count++] = row;
        }
    }
    partition->covered_edges = required.covered_edges;
    partition->pure_sparse_fallback = true;
    return residual_partition_status::success;
}

}  // namespace cellerator::compute::architecture::providers::nvidia::sm70::residual
