#pragma once

#include <Cellerator/compute/operation/candidate_catalog_v3/catalog.h>
#include <Cellerator/execution/joint_compiler/atom_requirement_v1.hh>

namespace cellerator::execution::atom_fragment {

struct local_candidate_atom_contract_v1 {
    std::uint64_t candidate_id = 0u;
    joint_compiler::atom_requirement_v1 requirement{};
};

enum class local_candidate_requirement_status_code_v1 : std::uint8_t {
    success = 0u,
    invalid_candidate_catalog,
    missing_contracts,
    duplicate_or_unordered_contract,
    missing_candidate_contract,
    invalid_requirement,
    insufficient_capacity,
};

struct local_candidate_requirement_status_v1 {
    local_candidate_requirement_status_code_v1 code =
        local_candidate_requirement_status_code_v1::success;
    std::uint64_t index = 0u;
    std::uint64_t required_capacity = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == local_candidate_requirement_status_code_v1::success;
    }
};

// Contracts carry the semantic facts absent from the low-level candidate
// catalog. Extraction never guesses biological identity, plane, or order.
local_candidate_requirement_status_v1 extract_local_candidate_requirements_v1(
    const compute::operation::catalog_v3::candidate_catalog_view_v3 &catalog,
    const local_candidate_atom_contract_v1 *contracts,
    std::uint64_t contract_count,
    joint_compiler::atom_requirement_v1 *requirements,
    std::uint64_t requirement_capacity,
    std::uint64_t *written) noexcept;

} // namespace cellerator::execution::atom_fragment
