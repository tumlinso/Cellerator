#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_cover_v1.hh>

#include <limits>

namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose {
namespace {

bool source_less(const global_relation_edge_v1 &left,
    const global_relation_edge_v1 &right) noexcept {
    if (left.global_source_id != right.global_source_id)
        return left.global_source_id < right.global_source_id;
    if (left.global_destination_id != right.global_destination_id)
        return left.global_destination_id < right.global_destination_id;
    return left.logical_edge_id < right.logical_edge_id;
}

transpose_status_v1 validate_input(const transpose_cover_input_v1 &input,
    std::uint64_t *owner_count) noexcept {
    if (owner_count == nullptr || input.edges == nullptr
        || input.source_order == nullptr || input.identity_order == nullptr
        || input.edge_count == 0u || input.forward_cover_id == 0u
        || input.transpose_cover_id == 0u
        || input.forward_cover_id == input.transpose_cover_id)
        return transpose_status_v1::invalid_argument;

    *owner_count = 0u;
    for (std::uint64_t position = 0u; position < input.edge_count; ++position) {
        const std::uint64_t source_index = input.source_order[position];
        const std::uint64_t identity_index = input.identity_order[position];
        if (source_index >= input.edge_count || identity_index >= input.edge_count)
            return transpose_status_v1::invalid_order;
        const global_relation_edge_v1 &source_edge = input.edges[source_index];
        const global_relation_edge_v1 &identity_edge = input.edges[identity_index];
        if (source_edge.logical_edge_id == 0u
            || source_edge.global_source_id == 0u
            || source_edge.global_destination_id == 0u)
            return transpose_status_v1::invalid_argument;
        if (position != 0u) {
            const global_relation_edge_v1 &prior_source =
                input.edges[input.source_order[position - 1u]];
            if (!source_less(prior_source, source_edge))
                return prior_source.logical_edge_id == source_edge.logical_edge_id
                    ? transpose_status_v1::duplicate_identity
                    : transpose_status_v1::invalid_order;
            const global_relation_edge_v1 &prior_identity =
                input.edges[input.identity_order[position - 1u]];
            if (prior_identity.logical_edge_id >= identity_edge.logical_edge_id)
                return prior_identity.logical_edge_id == identity_edge.logical_edge_id
                    ? transpose_status_v1::duplicate_identity
                    : transpose_status_v1::invalid_order;
        }
        if (position == 0u
            || input.edges[input.source_order[position - 1u]].global_source_id
                != source_edge.global_source_id) {
            if (*owner_count == std::numeric_limits<std::uint64_t>::max())
                return transpose_status_v1::arithmetic_overflow;
            ++*owner_count;
        }
    }
    return transpose_status_v1::success;
}

} // namespace

transpose_status_v1 query_transpose_cover_requirements_v1(
    const transpose_cover_input_v1 &input,
    transpose_cover_requirements_v1 *requirements) noexcept {
    if (requirements == nullptr) return transpose_status_v1::invalid_argument;
    std::uint64_t owner_count = 0u;
    const transpose_status_v1 status = validate_input(input, &owner_count);
    if (status != transpose_status_v1::success) return status;
    if (owner_count > std::numeric_limits<std::uint32_t>::max())
        return transpose_status_v1::arithmetic_overflow;
    requirements->placement_count = input.edge_count;
    requirements->owner_count = owner_count;
    return transpose_status_v1::success;
}

transpose_status_v1 build_transpose_cover_v1(
    const transpose_cover_input_v1 &input,
    const transpose_cover_storage_v1 &storage,
    transpose_cover_view_v1 *cover) noexcept {
    if (cover == nullptr || storage.placements == nullptr
        || storage.owners == nullptr)
        return transpose_status_v1::invalid_argument;
    transpose_cover_requirements_v1 requirements{};
    const transpose_status_v1 status =
        query_transpose_cover_requirements_v1(input, &requirements);
    if (status != transpose_status_v1::success) return status;
    if (storage.placement_capacity < requirements.placement_count
        || storage.owner_capacity < requirements.owner_count)
        return transpose_status_v1::insufficient_capacity;

    std::uint64_t owner_index = 0u;
    for (std::uint64_t position = 0u; position < input.edge_count; ++position) {
        const global_relation_edge_v1 &edge =
            input.edges[input.source_order[position]];
        if (position == 0u || storage.placements[position - 1u].global_source_id
            != edge.global_source_id) {
            storage.owners[owner_index] = {edge.global_source_id, position, 0u,
                static_cast<std::uint32_t>(owner_index), 0u};
            ++owner_index;
        }
        source_owner_schedule_v1 &owner = storage.owners[owner_index - 1u];
        ++owner.placement_count;
        storage.placements[position] = {edge.logical_edge_id,
            edge.global_source_id, edge.global_destination_id,
            owner.local_source_index, invalid_local_index_v1, position};
    }

    *cover = {transpose_cover_schema_v1, 0u, input.forward_cover_id,
        input.transpose_cover_id, storage.placements, requirements.placement_count,
        storage.owners, requirements.owner_count};
    return validate_transpose_cover_v1(*cover);
}

transpose_status_v1 validate_transpose_cover_v1(
    const transpose_cover_view_v1 &cover) noexcept {
    if (cover.schema_version != transpose_cover_schema_v1
        || cover.forward_cover_id == 0u || cover.transpose_cover_id == 0u
        || cover.forward_cover_id == cover.transpose_cover_id
        || cover.placements == nullptr || cover.placement_count == 0u
        || cover.owners == nullptr || cover.owner_count == 0u)
        return transpose_status_v1::invalid_cover;
    std::uint64_t expected_begin = 0u;
    for (std::uint64_t owner_index = 0u; owner_index < cover.owner_count;
        ++owner_index) {
        const source_owner_schedule_v1 &owner = cover.owners[owner_index];
        if (owner.global_source_id == 0u || owner.placement_count == 0u
            || owner.placement_begin != expected_begin
            || owner.local_source_index != owner_index
            || owner.placement_count > cover.placement_count - expected_begin)
            return transpose_status_v1::invalid_cover;
        for (std::uint64_t local = 0u; local < owner.placement_count; ++local) {
            const std::uint64_t position = owner.placement_begin + local;
            const transpose_edge_placement_v1 &edge = cover.placements[position];
            if (edge.logical_edge_id == 0u
                || edge.global_source_id != owner.global_source_id
                || edge.global_destination_id == 0u
                || edge.local_source_index != owner.local_source_index
                || edge.projection_value_position != position)
                return transpose_status_v1::invalid_cover;
        }
        expected_begin += owner.placement_count;
    }
    return expected_begin == cover.placement_count
        ? transpose_status_v1::success
        : transpose_status_v1::invalid_cover;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose
