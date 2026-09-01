#include <Cellerator/execution/atom_fragment/atom_bound_candidate_v1.hh>

namespace cellerator::execution::atom_fragment {
namespace {

using identity = joint_compiler::persistent_identity_v1;

bool valid(identity value) noexcept {
    return static_cast<bool>(
        joint_compiler::validate_persistent_identity_v1(value));
}

bool same(identity lhs, identity rhs) noexcept {
    return lhs.producer_namespace == rhs.producer_namespace
        && lhs.local_identity == rhs.local_identity;
}

} // namespace

atom_bound_candidate_status_v1 discover_atom_bound_candidates_v1(
    const compute::operation::catalog_v3::candidate_catalog_view_v3 &catalog,
    const joint_compiler::atom_requirement_v1 *requirements,
    std::uint64_t requirement_count,
    const joint_compiler::atom_binding_request_v1 *bindings,
    std::uint64_t binding_count,
    atom_bound_candidate_v1 *output,
    std::uint64_t output_capacity,
    std::uint64_t *written) noexcept {
    using code = atom_bound_candidate_status_code_v1;
    if (written == nullptr)
        return {code::insufficient_capacity, 0u, catalog.candidate_count};
    *written = 0u;
    if (compute::operation::catalog_v3::validate_candidate_catalog_v3(catalog)
        != compute::operation::catalog_v3::catalog_status::success)
        return {code::invalid_candidate_catalog, 0u, 0u};
    if (requirement_count != catalog.candidate_count
        || (requirement_count != 0u && requirements == nullptr))
        return {code::mismatched_requirement_count, 0u,
            catalog.candidate_count};
    for (std::uint64_t index = 0u; index < requirement_count; ++index) {
        if (!joint_compiler::validate_atom_requirement_v1(
                requirements[index]))
            return {code::invalid_requirement, index,
                catalog.candidate_count};
    }
    if (binding_count == 0u || bindings == nullptr)
        return {code::missing_bindings, 0u, catalog.candidate_count};
    for (std::uint64_t index = 0u; index < binding_count; ++index) {
        if (!valid(bindings[index].atom_identity)
            || !valid(bindings[index].requirement_identity)
            || !valid(bindings[index].affordance_identity))
            return {code::invalid_binding, index, catalog.candidate_count};
    }
    for (std::uint64_t index = 0u; index < requirement_count; ++index) {
        std::uint64_t matches = 0u;
        for (std::uint64_t binding = 0u; binding < binding_count; ++binding) {
            if (same(requirements[index].requirement_identity,
                    bindings[binding].requirement_identity))
                ++matches;
        }
        if (matches == 0u)
            return {code::missing_requirement_binding, index,
                catalog.candidate_count};
        if (matches != 1u)
            return {code::ambiguous_requirement_binding, index,
                catalog.candidate_count};
    }
    if (output_capacity < catalog.candidate_count
        || (catalog.candidate_count != 0u && output == nullptr))
        return {code::insufficient_capacity, 0u, catalog.candidate_count};

    for (std::uint64_t index = 0u; index < requirement_count; ++index) {
        for (std::uint64_t binding = 0u; binding < binding_count; ++binding) {
            if (!same(requirements[index].requirement_identity,
                    bindings[binding].requirement_identity))
                continue;
            output[index] = {catalog.candidates[index].identity.candidate_id,
                bindings[binding].atom_identity,
                bindings[binding].requirement_identity,
                bindings[binding].affordance_identity};
            break;
        }
    }
    *written = requirement_count;
    return {code::success, 0u, requirement_count};
}

} // namespace cellerator::execution::atom_fragment
