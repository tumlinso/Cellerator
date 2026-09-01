#pragma once

#include <Cellerator/execution/atom_fragment/atom_bound_candidate_v1.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

namespace cellerator::execution::atom_fragment {

enum class canonical_fallback_reason_v1 : std::uint8_t {
    bounded_frontier_empty = 1u,
    unmet_atom_requirement = 2u,
    unsupported_target = 3u,
    forced_by_caller = 4u,
};

struct canonical_fallback_request_v1 {
    std::uint64_t candidate_id = 0u;
    canonical_fallback_reason_v1 reason =
        canonical_fallback_reason_v1::bounded_frontier_empty;
    bool requires_order_transform = false;
    std::uint64_t visible_conversion_bytes = 0u;
};

struct canonical_fallback_v1 {
    atom_bound_candidate_v1 candidate{};
    order_id input_order{};
    order_id output_order{};
    canonical_fallback_reason_v1 reason =
        canonical_fallback_reason_v1::bounded_frontier_empty;
    bool requires_order_transform = false;
    std::uint64_t visible_conversion_bytes = 0u;
};

enum class canonical_fallback_diagnostic_code_v1 : std::uint32_t {
    selected = 1u,
    invalid_operation = 2u,
    invalid_request = 3u,
    invalid_candidates = 4u,
    candidate_missing = 5u,
    hidden_order_transform = 6u,
};

struct canonical_fallback_diagnostic_v1 {
    std::uint64_t subject = 0u;
    canonical_fallback_diagnostic_code_v1 code =
        canonical_fallback_diagnostic_code_v1::selected;
    std::uint64_t detail = 0u;
};

bool make_canonical_fallback_v1(
    const compute::operation::v2::operation_problem &operation,
    const atom_bound_candidate_v1 *candidates,
    std::uint64_t candidate_count,
    const canonical_fallback_request_v1 &request,
    canonical_fallback_v1 *fallback,
    canonical_fallback_diagnostic_v1 *diagnostic) noexcept;

} // namespace cellerator::execution::atom_fragment
