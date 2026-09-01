#include <Cellerator/compute/decomposition/sparse_update_conflict_v1.hh>

namespace cellerator::compute::decomposition {
namespace {

sparse_update_conflict_validation_result_v1 failure(
    sparse_update_conflict_validation_code_v1 code) noexcept {
    return {code};
}

bool all_zero(const std::uint8_t *values, std::uint64_t count) noexcept {
    for (std::uint64_t index = 0u; index < count; ++index) {
        if (values[index] != 0u)
            return false;
    }
    return true;
}

bool valid_policy(sparse_update_conflict_policy_v1 policy) noexcept {
    return policy >= sparse_update_conflict_policy_v1::unique_proven
        && policy <= sparse_update_conflict_policy_v1::atomic_unordered;
}

bool valid_order(sparse_update_order_v1 order) noexcept {
    return order >= sparse_update_order_v1::none
        && order <= sparse_update_order_v1::canonical_index_then_input;
}

bool unordered_merge_legal(operation::v2::sparse_update_operation update) {
    return update == operation::v2::sparse_update_operation::add
        || update == operation::v2::sparse_update_operation::multiply
        || update == operation::v2::sparse_update_operation::maximum;
}

}  // namespace

sparse_update_conflict_validation_result_v1
validate_sparse_update_conflict_contract_v1(
    const sparse_update_conflict_contract_v1 &contract) noexcept {
    using code = sparse_update_conflict_validation_code_v1;

    if (contract.schema_version != sparse_update_conflict_schema_version_v1)
        return failure(code::unsupported_schema);
    if (contract.reserved != 0u
        || !all_zero(contract.reserved2, sizeof(contract.reserved2)))
        return failure(code::nonzero_reserved);
    if (!operation::v2::valid_stable_id(contract.identity))
        return failure(code::invalid_identity);
    if (contract.update == nullptr)
        return failure(code::missing_update);
    if (!operation::v2::validate_sparse_axis_update(*contract.update))
        return failure(code::invalid_update);
    if (contract.fragment_count == 0u)
        return failure(code::invalid_fragment_count);
    if (!valid_policy(contract.conflict_policy))
        return failure(code::invalid_policy);
    if (!valid_order(contract.order))
        return failure(code::invalid_order);
    if (!contract.preserves_all_updates)
        return failure(code::update_loss_not_permitted);

    if (contract.update->indices_are_unique) {
        if (contract.conflicts_possible)
            return failure(code::unique_conflict_mismatch);
        if (contract.conflict_policy
                != sparse_update_conflict_policy_v1::unique_proven
            || contract.order != sparse_update_order_v1::none)
            return failure(code::invalid_unique_policy);
        return {};
    }
    if (!contract.conflicts_possible)
        return failure(code::unique_conflict_mismatch);
    if (contract.conflict_policy
        == sparse_update_conflict_policy_v1::unique_proven)
        return failure(code::conflicts_not_handled);
    if (contract.conflict_policy
        == sparse_update_conflict_policy_v1::reject_conflicts)
        return {};

    const bool stable_order = contract.order != sparse_update_order_v1::none;
    if (contract.update->update
            == operation::v2::sparse_update_operation::assign
        && (!stable_order
            || contract.conflict_policy
                == sparse_update_conflict_policy_v1::atomic_unordered))
        return failure(code::ordered_operation_requires_stable_order);
    if (contract.conflict_policy
        == sparse_update_conflict_policy_v1::atomic_unordered) {
        if (!unordered_merge_legal(contract.update->update))
            return failure(code::unordered_policy_not_legal);
        if (contract.deterministic_required || stable_order)
            return failure(code::deterministic_order_required);
    } else if (contract.deterministic_required && !stable_order) {
        return failure(code::deterministic_order_required);
    }
    return {};
}

}  // namespace cellerator::compute::decomposition
