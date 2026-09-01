#include <Cellerator/execution/atom_fragment/external_plane_binding_v1.hh>

#include <limits>

namespace cellerator::execution::atom_fragment {
namespace {

using identity = joint_compiler::persistent_identity_v1;

bool same(identity lhs, identity rhs) noexcept {
    return lhs.producer_namespace == rhs.producer_namespace
        && lhs.local_identity == rhs.local_identity;
}

const joint_compiler::atom_plane_affordance_v1 *find_plane(
    const joint_compiler::atom_affordance_v1 &affordance,
    identity plane) noexcept {
    for (std::uint64_t index = 0u; index < affordance.plane_count; ++index) {
        if (same(affordance.planes[index].plane_identity, plane))
            return &affordance.planes[index];
    }
    return nullptr;
}

bool generation_satisfies(const joint_compiler::atom_requirement_v1 &requirement,
    value_generation generation) noexcept {
    using policy = joint_compiler::generation_requirement_v1;
    if (requirement.generation_policy == policy::any_current)
        return generation.value != 0u;
    if (requirement.generation_policy == policy::exact)
        return generation.value == requirement.required_generation.value;
    return generation.value >= requirement.required_generation.value;
}

} // namespace

external_plane_binding_status_v1 bind_external_atom_planes_v1(
    const prepared_atom_fragment_v1 &prepared,
    const joint_compiler::atom_requirement_v1 &requirement,
    const joint_compiler::atom_affordance_v1 &affordance,
    const joint_compiler::external_binding_v1 *bindings,
    std::uint64_t binding_count,
    bound_atom_extent_v1 *output,
    std::uint64_t output_capacity,
    std::uint64_t *written) noexcept {
    using code = external_plane_binding_status_code_v1;
    if (written == nullptr)
        return {code::insufficient_capacity, 0u, 0u};
    *written = 0u;
    if (prepared.program == nullptr || prepared.candidate.candidate_id == 0u)
        return {code::invalid_prepared_fragment, 0u, 0u};
    if (!joint_compiler::validate_atom_requirement_v1(requirement))
        return {code::invalid_requirement, 0u, 0u};
    if (!joint_compiler::validate_atom_affordance_v1(affordance))
        return {code::invalid_affordance, 0u, 0u};
    if (!same(prepared.candidate.requirement_identity,
            requirement.requirement_identity)
        || !same(prepared.candidate.affordance_identity,
            affordance.affordance_identity))
        return {code::mismatched_contract, 0u, 0u};
    if (binding_count == 0u || bindings == nullptr)
        return {code::missing_bindings, 0u, 0u};

    std::uint64_t required_capacity = 0u;
    for (std::uint64_t index = 0u;
         index < requirement.required_plane_count; ++index) {
        const identity plane = requirement.required_planes[index];
        const auto *plane_affordance = find_plane(affordance, plane);
        if (plane_affordance == nullptr)
            return {code::missing_plane, index, required_capacity};
        std::uint64_t match = std::numeric_limits<std::uint64_t>::max();
        for (std::uint64_t binding = 0u; binding < binding_count; ++binding) {
            if (!same(bindings[binding].plane_identity, plane))
                continue;
            if (match != std::numeric_limits<std::uint64_t>::max())
                return {code::ambiguous_plane, index, required_capacity};
            match = binding;
        }
        if (match == std::numeric_limits<std::uint64_t>::max())
            return {code::missing_plane, index, required_capacity};
        const auto &binding = bindings[match];
        if (!joint_compiler::validate_external_binding_v1(binding)
            || !same(binding.atom_identity, prepared.candidate.atom_identity))
            return {code::invalid_binding, match, required_capacity};
        const auto &first = binding.extents[0];
        if (!same_identity(first.order, requirement.required_order)
            || !same_identity(first.order, plane_affordance->order))
            return {code::incompatible_order, index, required_capacity};
        if (first.generation.value != plane_affordance->generation.value
            || !generation_satisfies(requirement, first.generation))
            return {code::incompatible_generation, index, required_capacity};
        if (binding.extent_count
            > std::numeric_limits<std::uint64_t>::max() - required_capacity)
            return {code::insufficient_capacity, index,
                std::numeric_limits<std::uint64_t>::max()};
        required_capacity += binding.extent_count;
    }
    if (output_capacity < required_capacity
        || (required_capacity != 0u && output == nullptr))
        return {code::insufficient_capacity, 0u, required_capacity};

    std::uint64_t output_index = 0u;
    for (std::uint64_t index = 0u;
         index < requirement.required_plane_count; ++index) {
        const identity plane = requirement.required_planes[index];
        for (std::uint64_t binding = 0u; binding < binding_count; ++binding) {
            if (!same(bindings[binding].plane_identity, plane))
                continue;
            for (std::uint64_t extent = 0u;
                 extent < bindings[binding].extent_count; ++extent) {
                const auto &source = bindings[binding].extents[extent];
                output[output_index++] = {plane, source.address,
                    source.location, source.plane_byte_offset, source.bytes,
                    source.readiness, source.lease};
            }
            break;
        }
    }
    *written = output_index;
    return {code::success, 0u, required_capacity};
}

} // namespace cellerator::execution::atom_fragment
