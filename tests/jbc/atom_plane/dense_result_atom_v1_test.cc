#include <Cellerator/execution/atom_plane/dense_result_atom_v1.hh>

#include <array>

namespace atom = cellerator::execution::atom_plane;
namespace execution = cellerator::execution;

int main() {
    std::array<float, 8> storage{};
    atom::mutable_state_atom_plane_v1 state{};
    state.plane_identity = {1u, 10u};
    state.axis = {{1u, 1u}, {2u, 1u}, {3u, 1u}, {4u, 1u}};
    state.persistent_order = {20u, 1u};
    state.generation = {5u};
    state.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    state.quantization = {execution::quantization_kind::none,
        execution::numeric_type::invalid, execution::numeric_type::invalid,
        0u, nullptr, nullptr, 0u};
    state.values = storage.data();
    state.location = {execution::residency_kind::host, {}, -1, 0};
    state.element_count = storage.size();
    state.value_bytes = sizeof(storage);
    state.value_capacity_bytes = sizeof(storage);
    atom::dense_result_atom_v1 result{};
    if (!atom::emit_persistent_order_dense_result_atom_v1(
            state, state.generation, {1u, 30u}, &result)
        || !atom::validate_persistent_order_dense_result_atom_v1(state, result)
        || result.values != state.values
        || !execution::same_identity(
            result.persistent_order, state.persistent_order)) {
        return 1;
    }
    // Emission aliases the selected persistent order and never canonicalizes.
    if (execution::same_identity(result.persistent_order,
            execution::order_id{99u, 1u})) {
        return 2;
    }
    atom::dense_result_atom_v1 rejected{};
    rejected.element_count = 99u;
    if (atom::emit_persistent_order_dense_result_atom_v1(
            state, {4u}, {1u, 31u}, &rejected).code
            != atom::dense_result_atom_code_v1::stale_source_generation
        || rejected.element_count != 0u) {
        return 3;
    }
    result.persistent_order = {99u, 1u};
    if (atom::validate_persistent_order_dense_result_atom_v1(state, result).code
        != atom::dense_result_atom_code_v1::persistent_order_mismatch) {
        return 4;
    }
    result.persistent_order = state.persistent_order;
    result.generation = {4u};
    return atom::validate_persistent_order_dense_result_atom_v1(
            state, result).code
            == atom::dense_result_atom_code_v1::generation_mismatch
        ? 0 : 5;
}
