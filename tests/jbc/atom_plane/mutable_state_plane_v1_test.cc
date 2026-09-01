#include <Cellerator/execution/atom_plane/mutable_state_plane_v1.hh>

#include <array>

namespace atom = cellerator::execution::atom_plane;
namespace execution = cellerator::execution;

int main() {
    std::array<float, 16> state{};
    const std::array<atom::state_dirty_extent_v1, 2> dirty{{
        {2u, 3u}, {9u, 4u}
    }};
    atom::mutable_state_atom_plane_v1 plane{};
    plane.plane_identity = {1u, 30u};
    plane.axis = {{1u, 1u}, {2u, 1u}, {3u, 1u}, {4u, 1u}};
    plane.persistent_order = {20u, 1u};
    plane.generation = {5u};
    plane.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.quantization = {execution::quantization_kind::none,
        execution::numeric_type::invalid, execution::numeric_type::invalid,
        0u, nullptr, nullptr, 0u};
    plane.values = state.data();
    plane.location = {execution::residency_kind::host, {}, -1, 0};
    plane.dirty_extents = dirty.data();
    plane.element_count = state.size();
    plane.value_bytes = sizeof(state);
    plane.value_capacity_bytes = sizeof(state);
    plane.dirty_extent_count = dirty.size();
    if (!atom::validate_mutable_state_atom_plane_v1(plane)) {
        return 1;
    }

    // Generation and allocation address may advance without changing axis or
    // persistent execution order.
    std::array<float, 16> next_state{};
    plane.values = next_state.data();
    plane.generation = {6u};
    if (!atom::validate_mutable_state_atom_plane_v1(plane)) {
        return 2;
    }
    plane.value_capacity_bytes = sizeof(state) - 1u;
    if (atom::validate_mutable_state_atom_plane_v1(plane).code
        != atom::mutable_state_atom_plane_code_v1::insufficient_capacity) {
        return 3;
    }
    plane.value_capacity_bytes = sizeof(state);
    std::array<atom::state_dirty_extent_v1, 2> overlapping{{
        {2u, 8u}, {9u, 2u}
    }};
    plane.dirty_extents = overlapping.data();
    if (atom::validate_mutable_state_atom_plane_v1(plane).code
        != atom::mutable_state_atom_plane_code_v1::
            overlapping_or_unsorted_dirty_extent) {
        return 4;
    }
    std::array<atom::state_dirty_extent_v1, 1> out_of_range{{{15u, 2u}}};
    plane.dirty_extents = out_of_range.data();
    plane.dirty_extent_count = out_of_range.size();
    if (atom::validate_mutable_state_atom_plane_v1(plane).code
        != atom::mutable_state_atom_plane_code_v1::dirty_extent_out_of_range) {
        return 5;
    }
    plane.dirty_extents = nullptr;
    return atom::validate_mutable_state_atom_plane_v1(plane).code
            == atom::mutable_state_atom_plane_code_v1::missing_dirty_extents
        ? 0 : 6;
}
