#pragma once

#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_cover_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose {

enum class transpose_candidate_kind_v1 : std::uint8_t {
    sparse_source_owner = 1u,
    mma16_source_owner = 2u,
};

struct transpose_candidate_v1 {
    std::uint64_t candidate_id = 0u;
    std::uint64_t stage_id = 0u;
    std::uint64_t kernel_id = 0u;
    const char *stable_name = nullptr;
    transpose_candidate_kind_v1 kind =
        transpose_candidate_kind_v1::sparse_source_owner;
    std::uint32_t width_min = 1u;
    std::uint32_t width_max = 0u;
    bool requires_full_mma_groups = false;
    bool experimental = false;
    bool requires_measurement = true;
    std::uint8_t reserved = 0u;
};

struct transpose_candidate_catalog_v1 {
    const transpose_candidate_v1 *candidates = nullptr;
    std::uint64_t candidate_count = 0u;
};

struct transpose_reference_request_v1 {
    transpose_cover_view_v1 cover{};
    const float *projection_values = nullptr;
    const float *destination_gradient = nullptr;
    std::uint64_t local_destination_count = 0u;
    std::uint32_t dense_width = 0u;
    float *source_gradient = nullptr;
    std::uint64_t source_gradient_count = 0u;
};

transpose_candidate_catalog_v1 query_transpose_candidates_v1() noexcept;

transpose_status_v1 validate_transpose_candidate_v1(
    const transpose_candidate_v1 &candidate) noexcept;

transpose_status_v1 execute_transpose_reference_v1(
    const transpose_reference_request_v1 &request) noexcept;

static_assert(std::is_trivially_copyable<transpose_candidate_v1>::value,
    "transpose candidate descriptors must remain cold POD metadata");

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose
