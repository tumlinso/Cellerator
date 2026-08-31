#include <Cellerator/geometry/validation/scalable_validation_v2.hh>

#include <limits>

namespace cellerator::geometry {
namespace {

scale_validation_result_v2 failure(scale_validation_code_v2 code,
                                   std::uint64_t component,
                                   std::uint64_t item,
                                   std::uint64_t operations) noexcept {
    scale_validation_result_v2 result{};
    result.code = code;
    result.component = component;
    result.item = item;
    result.operations = operations;
    return result;
}

}  // namespace

bool checked_add_u64_v2(std::uint64_t left, std::uint64_t right,
                        std::uint64_t *out) noexcept {
    if (out == nullptr || right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    *out = left + right;
    return true;
}

bool checked_multiply_u64_v2(std::uint64_t left, std::uint64_t right,
                             std::uint64_t *out) noexcept {
    if (out == nullptr
        || (left != 0u && right > std::numeric_limits<std::uint64_t>::max() / left)) {
        return false;
    }
    *out = left * right;
    return true;
}

bool local_width_can_represent_v2(execution::local_index_width_v1 width,
                                  std::uint64_t extent) noexcept {
    switch (width) {
        case execution::local_index_width_v1::u16:
            return extent <= std::uint64_t{1} << 16u;
        case execution::local_index_width_v1::u32:
            return extent <= std::uint64_t{1} << 32u;
        case execution::local_index_width_v1::u64:
            return true;
    }
    return false;
}

bool load_compact_index_v2(const void *data,
                           execution::local_index_width_v1 width,
                           std::uint64_t position,
                           std::uint64_t *out) noexcept {
    if (data == nullptr || out == nullptr) {
        return false;
    }
    switch (width) {
        case execution::local_index_width_v1::u16:
            *out = static_cast<const std::uint16_t *>(data)[position];
            return true;
        case execution::local_index_width_v1::u32:
            *out = static_cast<const std::uint32_t *>(data)[position];
            return true;
        case execution::local_index_width_v1::u64:
            *out = static_cast<const std::uint64_t *>(data)[position];
            return true;
    }
    return false;
}

scale_validation_result_v2 validate_hierarchical_index_space_v1(
    const execution::hierarchical_index_space_view_v1 &view) noexcept {
    std::uint64_t operations = 0u;
    if (view.component_count != 0u && view.components == nullptr) {
        return failure(scale_validation_code_v2::null_pointer, 0u, 0u, operations);
    }

    std::uint64_t aggregate = 0u;
    std::uint64_t previous_identity = 0u;
    for (std::uint64_t component = 0u; component < view.component_count; ++component) {
        const auto &current = view.components[component];
        ++operations;
        if (component != 0u && current.component_identity <= previous_identity) {
            return failure(scale_validation_code_v2::component_order,
                           component, 0u, operations);
        }
        if (current.aggregate_begin != aggregate) {
            return failure(scale_validation_code_v2::aggregate_discontinuity,
                           component, 0u, operations);
        }
        const auto &space = current.index_space;
        if (!local_width_can_represent_v2(space.local_width, space.local_extent)) {
            return failure(scale_validation_code_v2::width_too_small,
                           component, 0u, operations);
        }
        if (space.local_extent != 0u && space.local_to_global == nullptr) {
            return failure(scale_validation_code_v2::null_pointer,
                           component, 0u, operations);
        }
        for (std::uint64_t local = 0u; local < space.local_extent; ++local) {
            ++operations;
            if (space.local_to_global[local] >= space.global_extent) {
                return failure(scale_validation_code_v2::global_index_out_of_range,
                               component, local, operations);
            }
        }
        if (!checked_add_u64_v2(aggregate, space.local_extent, &aggregate)) {
            return failure(scale_validation_code_v2::arithmetic_overflow,
                           component, 0u, operations);
        }
        previous_identity = current.component_identity;
    }
    if (aggregate != view.aggregate_extent) {
        return failure(scale_validation_code_v2::aggregate_extent_mismatch,
                       view.component_count, aggregate, operations);
    }
    scale_validation_result_v2 result{};
    result.operations = operations;
    return result;
}

scale_validation_result_v2 validate_scalable_support_v2(
    const scalable_support_view_v2 &view) noexcept {
    std::uint64_t operations = 0u;
    if (view.component_count != 0u && view.components == nullptr) {
        return failure(scale_validation_code_v2::null_pointer, 0u, 0u, operations);
    }
    std::uint64_t aggregate = 0u;
    std::uint64_t previous_identity = 0u;
    for (std::uint64_t component = 0u; component < view.component_count; ++component) {
        const auto &current = view.components[component];
        ++operations;
        if (component != 0u && current.component_identity <= previous_identity) {
            return failure(scale_validation_code_v2::component_order,
                           component, 0u, operations);
        }
        if (current.edge_map.component_identity != current.component_identity
            || current.edge_map.aggregate_begin != aggregate
            || current.edge_map.local_count != current.local_edge_count) {
            return failure(scale_validation_code_v2::aggregate_discontinuity,
                           component, 0u, operations);
        }
        if (current.local_edge_count == std::numeric_limits<std::uint64_t>::max()
            || !local_width_can_represent_v2(current.offset_width,
                                             current.local_edge_count + 1u)
            || !local_width_can_represent_v2(current.source_width,
                                             current.source_space.local_extent)) {
            return failure(scale_validation_code_v2::width_too_small,
                           component, 0u, operations);
        }
        if (current.destination_offsets == nullptr
            || (current.local_edge_count != 0u && current.source_indices == nullptr)
            || (current.local_edge_count != 0u
                && current.edge_map.local_to_aggregate == nullptr)) {
            return failure(scale_validation_code_v2::null_pointer,
                           component, 0u, operations);
        }
        if (current.destination_count == std::numeric_limits<std::uint64_t>::max()) {
            return failure(scale_validation_code_v2::arithmetic_overflow,
                           component, 0u, operations);
        }
        std::uint64_t previous = 0u;
        for (std::uint64_t destination = 0u;
             destination <= current.destination_count; ++destination) {
            std::uint64_t offset = 0u;
            ++operations;
            if (!load_compact_index_v2(current.destination_offsets,
                                       current.offset_width, destination, &offset)) {
                return failure(scale_validation_code_v2::invalid_width,
                               component, destination, operations);
            }
            if (offset < previous || offset > current.local_edge_count) {
                return failure(scale_validation_code_v2::offset_not_monotonic,
                               component, destination, operations);
            }
            previous = offset;
        }
        if (previous != current.local_edge_count) {
            return failure(scale_validation_code_v2::offset_extent_mismatch,
                           component, current.destination_count, operations);
        }
        for (std::uint64_t edge = 0u; edge < current.local_edge_count; ++edge) {
            std::uint64_t source = 0u;
            ++operations;
            if (!load_compact_index_v2(current.source_indices,
                                       current.source_width, edge, &source)) {
                return failure(scale_validation_code_v2::invalid_width,
                               component, edge, operations);
            }
            if (source >= current.source_space.local_extent
                || current.edge_map.local_to_aggregate[edge] >= view.aggregate_edge_count) {
                return failure(scale_validation_code_v2::local_index_out_of_range,
                               component, edge, operations);
            }
        }
        if (!checked_add_u64_v2(aggregate, current.local_edge_count, &aggregate)) {
            return failure(scale_validation_code_v2::arithmetic_overflow,
                           component, 0u, operations);
        }
        previous_identity = current.component_identity;
    }
    if (aggregate != view.aggregate_edge_count) {
        return failure(scale_validation_code_v2::aggregate_extent_mismatch,
                       view.component_count, aggregate, operations);
    }
    scale_validation_result_v2 result{};
    result.operations = operations;
    return result;
}

scale_validation_result_v2 validate_exact_cover_v2(
    const scalable_cover_view_v2 &cover,
    const scalable_support_view_v2 &support,
    cover_validation_workspace_v2 workspace) noexcept {
    std::uint64_t operations = 0u;
    if (cover.relation_identity != support.relation_identity
        || cover.aggregate_edge_count != support.aggregate_edge_count) {
        return failure(scale_validation_code_v2::aggregate_extent_mismatch,
                       0u, 0u, operations);
    }
    if (cover.work_item_count != 0u && cover.work_items == nullptr) {
        return failure(scale_validation_code_v2::null_pointer, 0u, 0u, operations);
    }

    std::uint64_t item = 0u;
    std::uint64_t covered = 0u;
    for (std::uint64_t component = 0u; component < support.component_count; ++component) {
        const auto &support_component = support.components[component];
        if (workspace.capacity < support_component.local_edge_count
            || (support_component.local_edge_count != 0u && workspace.marks == nullptr)) {
            return failure(scale_validation_code_v2::workspace_too_small,
                           component, 0u, operations);
        }
        if (workspace.generation == 0u) {
            for (std::uint64_t slot = 0u; slot < workspace.capacity; ++slot) {
                workspace.marks[slot] = 0u;
                ++operations;
            }
            workspace.generation = 1u;
        }
        const std::uint64_t generation = workspace.generation++;
        std::uint64_t component_covered = 0u;
        while (item < cover.work_item_count
               && cover.work_items[item].component_identity
                    == support_component.component_identity) {
            const auto &work_item = cover.work_items[item];
            for (std::uint64_t position = 0u;
                 position < work_item.local_edge_indices.count; ++position) {
                std::uint64_t edge = 0u;
                ++operations;
                if (!load_compact_index_v2(work_item.local_edge_indices.data,
                                           work_item.local_edge_indices.width,
                                           position, &edge)
                    || edge >= support_component.local_edge_count) {
                    return failure(scale_validation_code_v2::local_index_out_of_range,
                                   component, item, operations);
                }
                if (workspace.marks[edge] == generation) {
                    return failure(scale_validation_code_v2::duplicate_edge,
                                   component, edge, operations);
                }
                workspace.marks[edge] = generation;
                ++component_covered;
            }
            ++item;
        }
        if (component_covered != support_component.local_edge_count) {
            return failure(scale_validation_code_v2::missing_edge,
                           component, component_covered, operations);
        }
        if (!checked_add_u64_v2(covered, component_covered, &covered)) {
            return failure(scale_validation_code_v2::arithmetic_overflow,
                           component, 0u, operations);
        }
    }
    if (item != cover.work_item_count || covered != cover.aggregate_edge_count) {
        return failure(scale_validation_code_v2::missing_edge,
                       support.component_count, item, operations);
    }
    scale_validation_result_v2 result{};
    result.operations = operations;
    return result;
}

scale_validation_code_v2 exclusive_scan_counts_v2(
    const std::uint64_t *counts, std::uint64_t count,
    std::uint64_t *output, std::uint64_t output_capacity,
    std::uint64_t *operations) noexcept {
    if (output == nullptr || (count != 0u && counts == nullptr)) {
        return scale_validation_code_v2::null_pointer;
    }
    if (output_capacity < count || output_capacity - count < 1u) {
        return scale_validation_code_v2::workspace_too_small;
    }
    output[0] = 0u;
    for (std::uint64_t index = 0u; index < count; ++index) {
        if (!checked_add_u64_v2(output[index], counts[index], output + index + 1u)) {
            return scale_validation_code_v2::arithmetic_overflow;
        }
        if (operations != nullptr) {
            ++*operations;
        }
    }
    return scale_validation_code_v2::valid;
}

}  // namespace cellerator::geometry
