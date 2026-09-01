#include <Cellerator/execution/joint_compiler/atom_fragment_request_v1.hh>

#include <limits>

namespace cellerator::execution::joint_compiler {
namespace {

atom_fragment_request_validation_result_v1 failure(
    atom_fragment_request_validation_code_v1 code,
    std::uint64_t index = 0u,
    std::uint64_t nested = 0u) noexcept {
    return {code, index, nested};
}

bool valid_id(persistent_identity_v1 value) noexcept {
    return static_cast<bool>(validate_persistent_identity_v1(value));
}

bool less_id(persistent_identity_v1 lhs, persistent_identity_v1 rhs) noexcept {
    return lhs.producer_namespace < rhs.producer_namespace
        || (lhs.producer_namespace == rhs.producer_namespace
            && lhs.local_identity < rhs.local_identity);
}

bool less_order(order_id lhs, order_id rhs) noexcept {
    return lhs.high < rhs.high || (lhs.high == rhs.high && lhs.low < rhs.low);
}

atom_fragment_request_validation_result_v1 validate_index_space(
    const hierarchical_index_space_view_v1 &space,
    std::uint64_t index) noexcept {
    if (space.relation_identity == 0u || space.aggregate_extent == 0u
        || space.component_count == 0u || space.components == nullptr)
        return failure(atom_fragment_request_validation_code_v1::
            invalid_index_space, index);
    std::uint64_t expected_begin = 0u;
    std::uint64_t previous_identity = 0u;
    for (std::uint64_t component_index = 0u;
         component_index < space.component_count; ++component_index) {
        const hierarchical_index_component_v1 &component =
            space.components[component_index];
        const local_index_space_view_v1 &local = component.index_space;
        if (component.component_identity == 0u
            || (component_index != 0u
                && component.component_identity <= previous_identity)
            || component.aggregate_begin != expected_begin
            || local.partition_identity == 0u || local.local_extent == 0u
            || local.global_extent < local.local_extent
            || local.local_to_global == nullptr
            || (local.local_width != local_index_width_v1::u16
                && local.local_width != local_index_width_v1::u32
                && local.local_width != local_index_width_v1::u64)
            || expected_begin > std::numeric_limits<std::uint64_t>::max()
                - local.local_extent)
            return failure(atom_fragment_request_validation_code_v1::
                invalid_index_component, index, component_index);
        previous_identity = component.component_identity;
        expected_begin += local.local_extent;
    }
    if (expected_begin != space.aggregate_extent)
        return failure(atom_fragment_request_validation_code_v1::
            invalid_index_space, index);
    return {};
}

}  // namespace

atom_fragment_request_validation_result_v1 validate_atom_fragment_request_v1(
    const atom_fragment_request_v1 &request) noexcept {
    if (request.schema_version != atom_fragment_request_schema_version_v1)
        return failure(
            atom_fragment_request_validation_code_v1::unsupported_schema);
    if (request.record_bytes != sizeof(atom_fragment_request_v1))
        return failure(
            atom_fragment_request_validation_code_v1::invalid_record_bytes);
    if (!valid_id(request.request_identity))
        return failure(atom_fragment_request_validation_code_v1::
            invalid_request_identity);
    if (request.operation == nullptr
        || !compute::operation::v2::validate_operation_problem(
            *request.operation))
        return failure(atom_fragment_request_validation_code_v1::
            invalid_operation);
    if (request.exact_coverage_count == 0u
        || request.exact_coverages == nullptr)
        return failure(
            atom_fragment_request_validation_code_v1::missing_coverages);
    for (std::uint64_t index = 0u; index < request.exact_coverage_count;
         ++index) {
        if (!validate_logical_coverage_v1(request.exact_coverages[index]))
            return failure(atom_fragment_request_validation_code_v1::
                invalid_coverage, index);
        if (index != 0u && !less_id(
                request.exact_coverages[index - 1u].coverage_identity,
                request.exact_coverages[index].coverage_identity))
            return failure(atom_fragment_request_validation_code_v1::
                duplicate_or_unordered_coverage, index);
    }
    if (request.local_index_space_count == 0u
        || request.local_index_spaces == nullptr)
        return failure(
            atom_fragment_request_validation_code_v1::missing_index_spaces);
    for (std::uint64_t index = 0u; index < request.local_index_space_count;
         ++index) {
        const auto result =
            validate_index_space(request.local_index_spaces[index], index);
        if (!result) return result;
        if (index != 0u
            && request.local_index_spaces[index].relation_identity
                <= request.local_index_spaces[index - 1u].relation_identity)
            return failure(atom_fragment_request_validation_code_v1::
                duplicate_or_unordered_index_space, index);
    }
    if (request.external_order_count == 0u
        || request.external_orders == nullptr)
        return failure(
            atom_fragment_request_validation_code_v1::invalid_external_order);
    for (std::uint64_t index = 0u; index < request.external_order_count;
         ++index) {
        if (!valid_identity(request.external_orders[index]))
            return failure(atom_fragment_request_validation_code_v1::
                invalid_external_order, index);
        if (index != 0u && !less_order(request.external_orders[index - 1u],
                request.external_orders[index]))
            return failure(atom_fragment_request_validation_code_v1::
                duplicate_or_unordered_external_order, index);
    }
    if (request.decomposition == nullptr
        || !compute::decomposition::validate_decomposition_portfolio_v1(
            *request.decomposition))
        return failure(atom_fragment_request_validation_code_v1::
            invalid_decomposition);
    if (request.atom_binding_count == 0u || request.atom_bindings == nullptr)
        return failure(
            atom_fragment_request_validation_code_v1::missing_atom_bindings);
    for (std::uint64_t index = 0u; index < request.atom_binding_count; ++index) {
        const atom_binding_request_v1 &binding = request.atom_bindings[index];
        if (!valid_id(binding.atom_identity)
            || !valid_id(binding.requirement_identity)
            || !valid_id(binding.affordance_identity))
            return failure(atom_fragment_request_validation_code_v1::
                invalid_atom_binding, index);
        if (index != 0u && !less_id(
                request.atom_bindings[index - 1u].atom_identity,
                binding.atom_identity))
            return failure(atom_fragment_request_validation_code_v1::
                duplicate_or_unordered_atom_binding, index);
    }
    if (!valid_id(request.global_cost_contract))
        return failure(atom_fragment_request_validation_code_v1::
            invalid_global_cost_contract);
    if (!valid_id(request.target_profile))
        return failure(
            atom_fragment_request_validation_code_v1::invalid_target_profile);
    if (!valid_id(request.desired_output_affordance))
        return failure(atom_fragment_request_validation_code_v1::
            invalid_output_affordance);
    if (!valid_id(request.lowering_resumption_stage))
        return failure(atom_fragment_request_validation_code_v1::
            invalid_resumption_stage);
    return {};
}

}  // namespace cellerator::execution::joint_compiler
