#include <Cellerator/compute/operation/edge/gate_update_validation_v1.hh>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::compute::operation::edge {
namespace {

constexpr std::uint64_t fnv1a(const char *text,
    std::uint64_t value = 1469598103934665603ull) noexcept {
    return *text == '\0' ? value
        : fnv1a(text + 1,
            (value ^ static_cast<std::uint8_t>(*text)) * 1099511628211ull);
}

#define CE_EXOP_GATE_ENTRY(name, kind) \
    {fnv1a(name), name, registered_operation_v1::kind, true, true, true, false}

constexpr registry_entry_v1 entries[] = {
    CE_EXOP_GATE_ENTRY("edge.general-map.v1", general_map),
    CE_EXOP_GATE_ENTRY("edge.gate.per-edge-multiplicative.v1",
        per_edge_multiplicative),
    CE_EXOP_GATE_ENTRY("edge.gate.per-edge-predicate.v1", per_edge_predicate),
    CE_EXOP_GATE_ENTRY("edge.gate.per-source.v1", per_source_gate),
    CE_EXOP_GATE_ENTRY("edge.gate.per-destination.v1", per_destination_gate),
    CE_EXOP_GATE_ENTRY("edge.gate.per-component.v1", per_component_gate),
    CE_EXOP_GATE_ENTRY("edge.gate.factorized-source-destination.v1",
        factorized_source_destination_gate),
    CE_EXOP_GATE_ENTRY("edge.dynamic-support.byte-mask.v1",
        dynamic_support_byte_mask),
    CE_EXOP_GATE_ENTRY("edge.dynamic-support.bit-mask.v1",
        dynamic_support_bit_mask),
    CE_EXOP_GATE_ENTRY("sparse-axis-update.assign.v1", sparse_assign),
    CE_EXOP_GATE_ENTRY("sparse-axis-update.add.v1", sparse_add),
    CE_EXOP_GATE_ENTRY("sparse-axis-update.subtract.v1", sparse_subtract),
    CE_EXOP_GATE_ENTRY("sparse-axis-update.multiply.v1", sparse_multiply),
    CE_EXOP_GATE_ENTRY("sparse-axis-update.maximum.v1", sparse_maximum),
};

#undef CE_EXOP_GATE_ENTRY

constexpr std::size_t entry_count = sizeof(entries) / sizeof(entries[0]);

} // namespace

const registry_entry_v1 *registry_v1(std::size_t *count) noexcept {
    if (count != nullptr) *count = entry_count;
    return entries;
}

status_v1 validate_edge_coordinates_v1(const edge_coordinate_v1 *coordinates,
    local_edge_slice_v1 edges, std::uint32_t source_count,
    std::uint32_t destination_count, std::uint32_t component_count,
    validation_result_v1 *result) noexcept {
    if (coordinates == nullptr || result == nullptr
        || edges.local_edge_count == 0u || source_count == 0u
        || destination_count == 0u || component_count == 0u
        || edges.global_edge_begin > std::numeric_limits<std::uint64_t>::max()
                - edges.local_edge_count)
        return status_v1::invalid_argument;
    validation_result_v1 checked{};
    checked.valid = true;
    checked.first_invalid_global_item =
        std::numeric_limits<std::uint64_t>::max();
    for (std::uint32_t edge = 0u; edge < edges.local_edge_count; ++edge) {
        const edge_coordinate_v1 coordinate = coordinates[edge];
        const bool valid = coordinate.source_local < source_count
            && coordinate.destination_local < destination_count
            && coordinate.component_local < component_count;
        if (!valid && checked.valid)
            checked.first_invalid_global_item = edges.global_edge_begin + edge;
        checked.valid = checked.valid && valid;
        ++checked.checked_item_count;
    }
    *result = checked;
    return status_v1::success;
}

status_v1 reference_indexed_gate_v1(const edge_coordinate_v1 *coordinates,
    std::uint32_t edge_count, const float *input, const float *primary_gate,
    const float *secondary_gate, indexed_gate_kind_v1 kind,
    float *output) noexcept {
    if (coordinates == nullptr || edge_count == 0u || input == nullptr
        || primary_gate == nullptr || output == nullptr
        || kind > indexed_gate_kind_v1::factorized_source_destination
        || ((kind == indexed_gate_kind_v1::factorized_source_destination)
            != (secondary_gate != nullptr)))
        return status_v1::invalid_argument;
    for (std::uint32_t edge = 0u; edge < edge_count; ++edge) {
        if (!std::isfinite(input[edge])) return status_v1::invalid_argument;
        const edge_coordinate_v1 coordinate = coordinates[edge];
        float gate = 0.0f;
        switch (kind) {
            case indexed_gate_kind_v1::per_source:
                gate = primary_gate[coordinate.source_local];
                break;
            case indexed_gate_kind_v1::per_destination:
                gate = primary_gate[coordinate.destination_local];
                break;
            case indexed_gate_kind_v1::per_component:
                gate = primary_gate[coordinate.component_local];
                break;
            case indexed_gate_kind_v1::factorized_source_destination:
                gate = primary_gate[coordinate.source_local]
                    * secondary_gate[coordinate.destination_local];
                break;
        }
        if (!std::isfinite(gate)) return status_v1::invalid_argument;
        output[edge] = input[edge] * gate;
    }
    return status_v1::success;
}

sparse_axis_update::status_v1 reference_sparse_axis_update_v1(
    float *target, std::uint32_t local_axis_count,
    std::uint32_t component_count, const std::uint64_t *global_indices,
    std::uint64_t global_axis_begin, const float *updates,
    std::uint32_t update_count, sparse_axis_update::operation_v1 operation,
    validation_result_v1 *result) noexcept {
    using sparse_status = sparse_axis_update::status_v1;
    if (target == nullptr || local_axis_count == 0u || component_count == 0u
        || global_indices == nullptr || updates == nullptr || update_count == 0u
        || result == nullptr || operation > sparse_axis_update::operation_v1::maximum
        || global_axis_begin > std::numeric_limits<std::uint64_t>::max()
                - local_axis_count)
        return sparse_status::invalid_argument;
    validation_result_v1 checked{};
    checked.valid = true;
    checked.first_invalid_global_item =
        std::numeric_limits<std::uint64_t>::max();
    for (std::uint32_t update = 0u; update < update_count; ++update) {
        const std::uint64_t global = global_indices[update];
        bool valid = global >= global_axis_begin
            && global - global_axis_begin < local_axis_count;
        for (std::uint32_t prior = 0u; prior < update; ++prior)
            valid = valid && global_indices[prior] != global;
        if (!valid && checked.valid) checked.first_invalid_global_item = global;
        checked.valid = checked.valid && valid;
        ++checked.checked_item_count;
        if (!valid) continue;
        const std::uint64_t local = global - global_axis_begin;
        for (std::uint32_t component = 0u; component < component_count;
            ++component) {
            const std::size_t target_position = local * component_count
                + component;
            const std::size_t update_position =
                static_cast<std::size_t>(update) * component_count + component;
            const float current = target[target_position];
            const float value = updates[update_position];
            if (!std::isfinite(current) || !std::isfinite(value)) {
                checked.valid = false;
                continue;
            }
            switch (operation) {
                case sparse_axis_update::operation_v1::assign:
                    target[target_position] = value;
                    break;
                case sparse_axis_update::operation_v1::add:
                    target[target_position] = current + value;
                    break;
                case sparse_axis_update::operation_v1::subtract:
                    target[target_position] = current - value;
                    break;
                case sparse_axis_update::operation_v1::multiply:
                    target[target_position] = current * value;
                    break;
                case sparse_axis_update::operation_v1::maximum:
                    target[target_position] = std::fmax(current, value);
                    break;
            }
        }
    }
    *result = checked;
    return sparse_status::success;
}

} // namespace cellerator::compute::operation::edge
