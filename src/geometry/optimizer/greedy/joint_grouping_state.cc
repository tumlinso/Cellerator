#include "Cellerator/geometry/optimizer/greedy/joint_grouping_state.h"

#include <cstring>
#include <limits>

namespace cellerator::geometry::optimizer::greedy {
namespace {

std::uint64_t rectangle_hash(
        std::uint32_t source_group,
        std::uint32_t destination_group) noexcept {
    std::uint64_t value =
            (static_cast<std::uint64_t>(source_group) << 32U) |
            destination_group;
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

joint_grouping_status find_rectangle_slot(
        const joint_grouping_storage& storage,
        std::uint32_t source_group,
        std::uint32_t destination_group,
        bool allow_empty,
        std::uint32_t* slot) noexcept {
    if (slot == nullptr || storage.rectangle_capacity == 0 || storage.rectangles == nullptr) {
        return joint_grouping_status::invalid_argument;
    }
    const std::uint32_t initial = static_cast<std::uint32_t>(
            rectangle_hash(source_group, destination_group) % storage.rectangle_capacity);
    for (std::uint32_t probe = 0; probe < storage.rectangle_capacity; ++probe) {
        const std::uint32_t candidate =
                static_cast<std::uint32_t>((initial + probe) % storage.rectangle_capacity);
        const auto& record = storage.rectangles[candidate];
        if (record.occupied == 0) {
            if (allow_empty) {
                *slot = candidate;
                return joint_grouping_status::success;
            }
            return joint_grouping_status::state_mismatch;
        }
        if (record.source_group == source_group &&
            record.destination_group == destination_group) {
            *slot = candidate;
            return joint_grouping_status::success;
        }
    }
    return joint_grouping_status::rectangle_table_full;
}

joint_grouping_status validate_storage(
        const joint_grouping_problem_view& problem,
        std::uint32_t source_group_count,
        std::uint32_t destination_group_count,
        const joint_grouping_storage& storage) noexcept {
    if (problem.edge_count != 0 &&
        (problem.edge_sources == nullptr || problem.edge_destinations == nullptr)) {
        return joint_grouping_status::invalid_argument;
    }
    if ((problem.source_count != 0 && storage.source_groups == nullptr) ||
        storage.source_group_capacity < problem.source_count ||
        (problem.destination_count != 0 && storage.destination_groups == nullptr) ||
        storage.destination_group_capacity < problem.destination_count ||
        (source_group_count != 0 && storage.source_group_sizes == nullptr) ||
        storage.source_group_size_capacity < source_group_count ||
        (destination_group_count != 0 && storage.destination_group_sizes == nullptr) ||
        storage.destination_group_size_capacity < destination_group_count ||
        (problem.edge_count != 0 && storage.edge_rectangle_slots == nullptr) ||
        storage.edge_rectangle_capacity < problem.edge_count ||
        storage.rectangles == nullptr || storage.rectangle_capacity == 0) {
        return joint_grouping_status::insufficient_storage;
    }
    if ((problem.source_count != 0 && source_group_count == 0) ||
        (problem.destination_count != 0 && destination_group_count == 0)) {
        return joint_grouping_status::invalid_problem;
    }
    return joint_grouping_status::success;
}

}  // namespace

joint_grouping_status initialize_joint_grouping_state(
        const joint_grouping_problem_view& problem,
        const std::uint32_t* initial_source_groups,
        const std::uint32_t* initial_destination_groups,
        std::uint32_t source_group_count,
        std::uint32_t destination_group_count,
        const joint_grouping_storage& storage,
        mutable_joint_grouping_state* state) noexcept {
    if (state == nullptr ||
        (problem.source_count != 0 && initial_source_groups == nullptr) ||
        (problem.destination_count != 0 && initial_destination_groups == nullptr)) {
        return joint_grouping_status::invalid_argument;
    }
    const auto storage_status = validate_storage(
            problem, source_group_count, destination_group_count, storage);
    if (storage_status != joint_grouping_status::success) {
        return storage_status;
    }
    for (std::uint64_t edge = 0; edge < problem.edge_count; ++edge) {
        if (problem.edge_sources[edge] >= problem.source_count ||
            problem.edge_destinations[edge] >= problem.destination_count) {
            return joint_grouping_status::invalid_problem;
        }
    }
    for (std::uint32_t source = 0; source < problem.source_count; ++source) {
        if (initial_source_groups[source] >= source_group_count) {
            return joint_grouping_status::invalid_problem;
        }
    }
    for (std::uint32_t destination = 0;
         destination < problem.destination_count;
         ++destination) {
        if (initial_destination_groups[destination] >= destination_group_count) {
            return joint_grouping_status::invalid_problem;
        }
    }

    state->problem = problem;
    state->storage = storage;
    state->source_group_count = source_group_count;
    state->destination_group_count = destination_group_count;
    state->occupied_rectangle_count = 0;
    state->generation = 0;
    if (problem.source_count != 0) {
        std::memcpy(storage.source_groups, initial_source_groups,
                    sizeof(std::uint32_t) * problem.source_count);
    }
    if (problem.destination_count != 0) {
        std::memcpy(storage.destination_groups, initial_destination_groups,
                    sizeof(std::uint32_t) * problem.destination_count);
    }
    return rebuild_joint_rectangle_census(state);
}

joint_grouping_status rebuild_joint_rectangle_census(
        mutable_joint_grouping_state* state) noexcept {
    if (state == nullptr) {
        return joint_grouping_status::invalid_argument;
    }
    const auto storage_status = validate_storage(
            state->problem,
            state->source_group_count,
            state->destination_group_count,
            state->storage);
    if (storage_status != joint_grouping_status::success) {
        return storage_status;
    }
    std::memset(state->storage.source_group_sizes, 0,
                sizeof(std::uint64_t) * state->source_group_count);
    std::memset(state->storage.destination_group_sizes, 0,
                sizeof(std::uint64_t) * state->destination_group_count);
    for (std::uint32_t slot = 0;
         slot < state->storage.rectangle_capacity;
         ++slot) {
        state->storage.rectangles[slot] = {};
    }
    state->occupied_rectangle_count = 0;

    for (std::uint32_t source = 0; source < state->problem.source_count; ++source) {
        const std::uint32_t group = state->storage.source_groups[source];
        if (group >= state->source_group_count) {
            return joint_grouping_status::invalid_problem;
        }
        ++state->storage.source_group_sizes[group];
    }
    for (std::uint32_t destination = 0;
         destination < state->problem.destination_count;
         ++destination) {
        const std::uint32_t group = state->storage.destination_groups[destination];
        if (group >= state->destination_group_count) {
            return joint_grouping_status::invalid_problem;
        }
        ++state->storage.destination_group_sizes[group];
    }
    for (std::uint64_t edge = 0; edge < state->problem.edge_count; ++edge) {
        const std::uint32_t source = state->problem.edge_sources[edge];
        const std::uint32_t destination = state->problem.edge_destinations[edge];
        if (source >= state->problem.source_count ||
            destination >= state->problem.destination_count) {
            return joint_grouping_status::invalid_problem;
        }
        const std::uint32_t source_group = state->storage.source_groups[source];
        const std::uint32_t destination_group =
                state->storage.destination_groups[destination];
        std::uint32_t slot = 0;
        const auto find_status = find_rectangle_slot(
                state->storage, source_group, destination_group, true, &slot);
        if (find_status != joint_grouping_status::success) {
            return find_status;
        }
        auto& rectangle = state->storage.rectangles[slot];
        if (rectangle.occupied == 0) {
            rectangle.source_group = source_group;
            rectangle.destination_group = destination_group;
            rectangle.contribution_count = 0;
            rectangle.occupied = 1;
            ++state->occupied_rectangle_count;
        }
        if (rectangle.contribution_count == std::numeric_limits<std::uint64_t>::max()) {
            return joint_grouping_status::arithmetic_overflow;
        }
        ++rectangle.contribution_count;
        state->storage.edge_rectangle_slots[edge] = slot;
    }
    if (state->generation != std::numeric_limits<std::uint64_t>::max()) {
        ++state->generation;
    } else {
        return joint_grouping_status::arithmetic_overflow;
    }
    return joint_grouping_status::success;
}

joint_grouping_status validate_joint_grouping_state(
        const mutable_joint_grouping_state& state,
        const joint_grouping_validation_workspace& workspace,
        std::uint64_t* validated_contributions) noexcept {
    if (validated_contributions == nullptr) {
        return joint_grouping_status::invalid_argument;
    }
    *validated_contributions = 0;
    const auto storage_status = validate_storage(
            state.problem,
            state.source_group_count,
            state.destination_group_count,
            state.storage);
    if (storage_status != joint_grouping_status::success) {
        return storage_status;
    }
    if (workspace.source_group_capacity < state.source_group_count ||
        workspace.destination_group_capacity < state.destination_group_count ||
        (state.source_group_count != 0 && workspace.source_group_sizes == nullptr) ||
        (state.destination_group_count != 0 &&
         workspace.destination_group_sizes == nullptr)) {
        return joint_grouping_status::insufficient_storage;
    }
    if (state.source_group_count != 0) {
        std::memset(workspace.source_group_sizes, 0,
                    sizeof(std::uint64_t) * state.source_group_count);
    }
    if (state.destination_group_count != 0) {
        std::memset(workspace.destination_group_sizes, 0,
                    sizeof(std::uint64_t) * state.destination_group_count);
    }
    for (std::uint32_t source = 0; source < state.problem.source_count; ++source) {
        const std::uint32_t group = state.storage.source_groups[source];
        if (group >= state.source_group_count) {
            return joint_grouping_status::state_mismatch;
        }
        ++workspace.source_group_sizes[group];
    }
    for (std::uint32_t destination = 0;
         destination < state.problem.destination_count;
         ++destination) {
        const std::uint32_t group = state.storage.destination_groups[destination];
        if (group >= state.destination_group_count) {
            return joint_grouping_status::state_mismatch;
        }
        ++workspace.destination_group_sizes[group];
    }
    for (std::uint32_t group = 0; group < state.source_group_count; ++group) {
        if (workspace.source_group_sizes[group] !=
            state.storage.source_group_sizes[group]) {
            return joint_grouping_status::state_mismatch;
        }
    }
    for (std::uint32_t group = 0; group < state.destination_group_count; ++group) {
        if (workspace.destination_group_sizes[group] !=
            state.storage.destination_group_sizes[group]) {
            return joint_grouping_status::state_mismatch;
        }
    }
    std::uint32_t occupied = 0;
    for (std::uint32_t slot = 0; slot < state.storage.rectangle_capacity; ++slot) {
        const auto& rectangle = state.storage.rectangles[slot];
        if (rectangle.occupied == 0) {
            continue;
        }
        if (rectangle.source_group >= state.source_group_count ||
            rectangle.destination_group >= state.destination_group_count ||
            rectangle.contribution_count == 0) {
            return joint_grouping_status::state_mismatch;
        }
        if (*validated_contributions > std::numeric_limits<std::uint64_t>::max() -
                                           rectangle.contribution_count) {
            return joint_grouping_status::arithmetic_overflow;
        }
        *validated_contributions += rectangle.contribution_count;
        ++occupied;
    }
    if (*validated_contributions != state.problem.edge_count ||
        occupied != state.occupied_rectangle_count) {
        return joint_grouping_status::state_mismatch;
    }
    for (std::uint64_t edge = 0; edge < state.problem.edge_count; ++edge) {
        const std::uint32_t slot = state.storage.edge_rectangle_slots[edge];
        if (slot >= state.storage.rectangle_capacity) {
            return joint_grouping_status::state_mismatch;
        }
        const auto& rectangle = state.storage.rectangles[slot];
        const std::uint32_t source_group =
                state.storage.source_groups[state.problem.edge_sources[edge]];
        const std::uint32_t destination_group =
                state.storage.destination_groups[state.problem.edge_destinations[edge]];
        if (rectangle.occupied == 0 ||
            rectangle.source_group != source_group ||
            rectangle.destination_group != destination_group) {
            return joint_grouping_status::state_mismatch;
        }
    }
    return joint_grouping_status::success;
}

}  // namespace cellerator::geometry::optimizer::greedy
