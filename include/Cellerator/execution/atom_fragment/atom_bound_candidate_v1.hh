#pragma once

#include <Cellerator/execution/atom_fragment/local_candidate_requirements_v1.hh>
#include <Cellerator/execution/preparation/atom_fragment_request_v1.hh>

namespace cellerator::execution::atom_fragment {

struct atom_bound_candidate_v1 {
    std::uint64_t candidate_id = 0u;
    preparation::persistent_identity_v1 atom_identity{};
    preparation::persistent_identity_v1 requirement_identity{};
    preparation::persistent_identity_v1 affordance_identity{};
};

enum class atom_bound_candidate_status_code_v1 : std::uint8_t {
    success = 0u,
    invalid_candidate_catalog,
    mismatched_requirement_count,
    invalid_requirement,
    missing_bindings,
    invalid_binding,
    missing_requirement_binding,
    ambiguous_requirement_binding,
    insufficient_capacity,
};

struct atom_bound_candidate_status_v1 {
    atom_bound_candidate_status_code_v1 code =
        atom_bound_candidate_status_code_v1::success;
    std::uint64_t index = 0u;
    std::uint64_t required_capacity = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == atom_bound_candidate_status_code_v1::success;
    }
};

atom_bound_candidate_status_v1 discover_atom_bound_candidates_v1(
    const compute::operation::catalog_v3::candidate_catalog_view_v3 &catalog,
    const preparation::atom_requirement_v1 *requirements,
    std::uint64_t requirement_count,
    const preparation::atom_binding_request_v1 *bindings,
    std::uint64_t binding_count,
    atom_bound_candidate_v1 *output,
    std::uint64_t output_capacity,
    std::uint64_t *written) noexcept;

} // namespace cellerator::execution::atom_fragment
