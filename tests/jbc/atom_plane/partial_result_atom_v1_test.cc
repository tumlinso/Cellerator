#include <Cellerator/execution/atom_plane/partial_result_atom_v1.hh>

#include <array>

namespace atom = cellerator::execution::atom_plane;
namespace execution = cellerator::execution;

int main() {
    std::array<float, 16> storage{};
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
    const std::array<atom::state_dirty_extent_v1, 2> coverage{{
        {2u, 3u}, {9u, 4u}
    }};
    atom::partial_result_atom_v1 result{};
    if (!atom::emit_partial_result_atom_v1(state, state.generation,
            {1u, 30u}, {1u, 40u}, coverage.data(), coverage.size(), &result)
        || !atom::validate_partial_result_atom_v1(result)
        || result.covered_element_count != 7u
        || result.values != state.values) {
        return 1;
    }
    std::array<atom::state_dirty_extent_v1, 2> overlapping{{
        {2u, 8u}, {9u, 2u}
    }};
    if (atom::emit_partial_result_atom_v1(state, state.generation,
            {1u, 31u}, {1u, 40u}, overlapping.data(), overlapping.size(),
            &result).code
        != atom::partial_result_atom_code_v1::
            overlapping_or_unsorted_covered_extent) {
        return 2;
    }
    const std::array<atom::state_dirty_extent_v1, 1> complete{{{0u, 16u}}};
    if (atom::emit_partial_result_atom_v1(state, state.generation,
            {1u, 32u}, {1u, 40u}, complete.data(), complete.size(),
            &result).code
        != atom::partial_result_atom_code_v1::complete_result_not_partial) {
        return 3;
    }
    if (atom::emit_partial_result_atom_v1(state, {4u}, {1u, 33u},
            {1u, 40u}, coverage.data(), coverage.size(), &result).code
        != atom::partial_result_atom_code_v1::stale_source_generation) {
        return 4;
    }
    return atom::emit_partial_result_atom_v1(state, state.generation,
            {1u, 34u}, {}, coverage.data(), coverage.size(), &result).code
            == atom::partial_result_atom_code_v1::
                invalid_merge_algebra_identity
        ? 0 : 5;
}
