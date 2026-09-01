#include <Cellerator/execution/atom_fragment/requirements_v1.hh>

#include <cstddef>
#include <limits>

namespace cellerator::execution::atom_fragment {
namespace {

bool checked_add(std::uint64_t value, std::uint64_t *total) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - *total)
        return false;
    *total += value;
    return true;
}

bool checked_product(std::uint64_t count, std::uint64_t item_bytes,
    std::uint64_t *result) noexcept {
    if (count != 0u
        && item_bytes > std::numeric_limits<std::uint64_t>::max() / count)
        return false;
    *result = count * item_bytes;
    return true;
}

bool add_product(std::uint64_t count, std::uint64_t item_bytes,
    std::uint64_t *total) noexcept {
    std::uint64_t bytes = 0u;
    return checked_product(count, item_bytes, &bytes) && checked_add(bytes, total);
}

} // namespace

atom_fragment_requirements_status_v1 query_atom_fragment_requirements_v1(
    const joint_compiler::atom_fragment_request_v1 &request,
    const atom_fragment_query_limits_v1 &limits,
    atom_fragment_requirements_v1 *requirements) noexcept {
    if (requirements == nullptr)
        return {atom_fragment_requirements_status_code_v1::null_output, 0u};
    *requirements = {};
    const auto request_result =
        joint_compiler::validate_atom_fragment_request_v1(request);
    if (!request_result)
        return {atom_fragment_requirements_status_code_v1::invalid_request,
            static_cast<std::uint64_t>(request_result.code)};
    if (limits.projection_capacity == 0u
        || limits.projection_chunk_capacity == 0u
        || limits.candidate_capacity == 0u
        || limits.prepared_stage_capacity == 0u
        || limits.diagnostic_capacity == 0u)
        return {atom_fragment_requirements_status_code_v1::invalid_limits, 0u};

    std::uint64_t component_count = 0u;
    std::uint64_t local_index_bytes = 0u;
    for (std::uint64_t space_index = 0u;
         space_index < request.local_index_space_count; ++space_index) {
        const auto &space = request.local_index_spaces[space_index];
        if (!checked_add(space.component_count, &component_count))
            return {atom_fragment_requirements_status_code_v1::
                arithmetic_overflow, space_index};
        for (std::uint64_t component_index = 0u;
             component_index < space.component_count; ++component_index) {
            const auto &local = space.components[component_index].index_space;
            if (!add_product(local.local_extent,
                    static_cast<std::uint64_t>(local.local_width),
                    &local_index_bytes)
                || (local.global_identity_sidecar != nullptr
                    && !add_product(local.local_extent, sizeof(std::uint64_t),
                        &local_index_bytes)))
                return {atom_fragment_requirements_status_code_v1::
                    arithmetic_overflow, component_index};
        }
    }

    std::uint64_t projection_bytes = 0u;
    if (!add_product(limits.projection_capacity,
            sizeof(acquisition_v2::projection_record), &projection_bytes)
        || !add_product(limits.projection_chunk_capacity,
            sizeof(acquisition_v2::projection_chunk), &projection_bytes))
        return {atom_fragment_requirements_status_code_v1::arithmetic_overflow,
            1u};

    std::uint64_t candidate_bytes = 0u;
    if (!add_product(limits.candidate_capacity,
            sizeof(planner::portfolio::candidate_workspace_state_v1),
            &candidate_bytes)
        || !add_product(limits.candidate_capacity,
            2u * sizeof(std::uint64_t) + sizeof(double), &candidate_bytes))
        return {atom_fragment_requirements_status_code_v1::arithmetic_overflow,
            2u};

    std::uint64_t program_bytes = 0u;
    if (!add_product(limits.prepared_stage_capacity,
            sizeof(program::prepared_stage_v2), &program_bytes)
        || !add_product(limits.dependency_capacity, sizeof(std::uint64_t),
            &program_bytes))
        return {atom_fragment_requirements_status_code_v1::arithmetic_overflow,
            3u};

    std::uint64_t binding_bytes = 0u;
    if (!checked_product(limits.prepared_stage_capacity,
            sizeof(program::launch_binding_v2), &binding_bytes))
        return {atom_fragment_requirements_status_code_v1::arithmetic_overflow,
            4u};
    std::uint64_t diagnostic_bytes = 0u;
    if (!checked_product(limits.diagnostic_capacity,
            sizeof(atom_fragment_diagnostic_record_v1), &diagnostic_bytes))
        return {atom_fragment_requirements_status_code_v1::arithmetic_overflow,
            5u};

    requirements->local_index_component_count = component_count;
    requirements->projection_capacity = limits.projection_capacity;
    requirements->projection_chunk_capacity = limits.projection_chunk_capacity;
    requirements->candidate_capacity = limits.candidate_capacity;
    requirements->prepared_stage_capacity = limits.prepared_stage_capacity;
    requirements->dependency_capacity = limits.dependency_capacity;
    requirements->binding_capacity = limits.prepared_stage_capacity;
    requirements->diagnostic_capacity = limits.diagnostic_capacity;
    requirements->local_indexes = {local_index_bytes, alignof(std::uint64_t)};
    requirements->projections = {projection_bytes,
        alignof(acquisition_v2::projection_record)};
    requirements->candidate_workspace = {candidate_bytes,
        alignof(planner::portfolio::candidate_workspace_state_v1)};
    requirements->prepared_program = {
        program_bytes, alignof(program::prepared_stage_v2)};
    requirements->bindings = {
        binding_bytes, alignof(program::launch_binding_v2)};
    requirements->diagnostics = {
        diagnostic_bytes, alignof(atom_fragment_diagnostic_record_v1)};
    requirements->transient = {limits.transient_bytes,
        alignof(std::max_align_t)};
    return {};
}

} // namespace cellerator::execution::atom_fragment
