#include <Cellerator/execution/atom_fragment/local_candidate_requirements_v1.hh>

namespace cellerator::execution::atom_fragment {

local_candidate_requirement_status_v1 extract_local_candidate_requirements_v1(
    const compute::operation::catalog_v3::candidate_catalog_view_v3 &catalog,
    const local_candidate_atom_contract_v1 *contracts,
    std::uint64_t contract_count,
    joint_compiler::atom_requirement_v1 *requirements,
    std::uint64_t requirement_capacity,
    std::uint64_t *written) noexcept {
    using code = local_candidate_requirement_status_code_v1;
    if (written == nullptr)
        return {code::insufficient_capacity, 0u, catalog.candidate_count};
    *written = 0u;
    if (compute::operation::catalog_v3::validate_candidate_catalog_v3(catalog)
        != compute::operation::catalog_v3::catalog_status::success)
        return {code::invalid_candidate_catalog, 0u, 0u};
    if (contract_count == 0u || contracts == nullptr)
        return {code::missing_contracts, 0u, catalog.candidate_count};
    for (std::uint64_t index = 0u; index < contract_count; ++index) {
        if (contracts[index].candidate_id == 0u
            || (index != 0u && contracts[index - 1u].candidate_id
                >= contracts[index].candidate_id))
            return {code::duplicate_or_unordered_contract, index,
                catalog.candidate_count};
        if (!joint_compiler::validate_atom_requirement_v1(
                contracts[index].requirement))
            return {code::invalid_requirement, index, catalog.candidate_count};
    }
    std::uint64_t contract_index = 0u;
    for (std::uint64_t index = 0u; index < catalog.candidate_count; ++index) {
        const std::uint64_t candidate_id =
            catalog.candidates[index].identity.candidate_id;
        while (contract_index < contract_count
            && contracts[contract_index].candidate_id < candidate_id)
            ++contract_index;
        if (contract_index == contract_count
            || contracts[contract_index].candidate_id != candidate_id)
            return {code::missing_candidate_contract, index,
                catalog.candidate_count};
    }
    if (requirement_capacity < catalog.candidate_count
        || (catalog.candidate_count != 0u && requirements == nullptr))
        return {code::insufficient_capacity, 0u, catalog.candidate_count};

    contract_index = 0u;
    for (std::uint64_t index = 0u; index < catalog.candidate_count; ++index) {
        const std::uint64_t candidate_id =
            catalog.candidates[index].identity.candidate_id;
        while (contracts[contract_index].candidate_id < candidate_id)
            ++contract_index;
        requirements[index] = contracts[contract_index].requirement;
    }
    *written = catalog.candidate_count;
    return {code::success, 0u, catalog.candidate_count};
}

} // namespace cellerator::execution::atom_fragment
