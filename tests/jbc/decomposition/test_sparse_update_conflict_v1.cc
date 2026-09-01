#include <Cellerator/compute/decomposition/sparse_update_conflict_v1.hh>

#include <cassert>
#include <cstdint>

namespace decomposition = cellerator::compute::decomposition;
namespace operation = cellerator::compute::operation::v2;
namespace execution = cellerator::execution;

namespace {

template<typename Identity>
Identity identity(std::uint64_t value) { return {value, value + 1u}; }

execution::persistent_axis_identity axis(std::uint64_t seed) {
    return {{execution::biological_abi_version,
                execution::serialized_record_kind::persistent_axis_identity,
                sizeof(execution::persistent_axis_identity)},
        identity<execution::domain_id>(seed),
        identity<execution::order_id>(seed + 2u),
        identity<execution::geometry_id>(seed + 4u),
        identity<execution::partition_id>(seed + 6u)};
}

}  // namespace

int main() {
    operation::sparse_axis_update_descriptor update{};
    update.target_axis = axis(10u);
    update.target_operand = 0u;
    update.index_operand = 1u;
    update.update_operand = 2u;
    update.value_type = execution::numeric_type::f32;
    update.indices_are_unique = true;
    assert(operation::validate_sparse_axis_update(update));

    decomposition::sparse_update_conflict_contract_v1 contract{};
    contract.identity = {20u, 21u};
    contract.update = &update;
    contract.fragment_count = 2u;
    assert(decomposition::validate_sparse_update_conflict_contract_v1(
        contract));

    update.indices_are_unique = false;
    contract.conflicts_possible = true;
    contract.conflict_policy =
        decomposition::sparse_update_conflict_policy_v1::deterministic_serial;
    contract.order = decomposition::
        sparse_update_order_v1::stable_fragment_then_input;
    assert(decomposition::validate_sparse_update_conflict_contract_v1(
        contract));

    contract.conflict_policy =
        decomposition::sparse_update_conflict_policy_v1::atomic_unordered;
    contract.order = decomposition::sparse_update_order_v1::none;
    contract.deterministic_required = false;
    auto status = decomposition::
        validate_sparse_update_conflict_contract_v1(contract);
    assert(status.code == decomposition::
        sparse_update_conflict_validation_code_v1::
            ordered_operation_requires_stable_order);

    update.update = operation::sparse_update_operation::add;
    assert(decomposition::validate_sparse_update_conflict_contract_v1(
        contract));

    contract.deterministic_required = true;
    status = decomposition::validate_sparse_update_conflict_contract_v1(
        contract);
    assert(status.code == decomposition::
        sparse_update_conflict_validation_code_v1::
            deterministic_order_required);
    return 0;
}
