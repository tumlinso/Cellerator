#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_value_map_v1.hh>

#include <limits>

namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose {
namespace {

bool valid_dictionary(const local_destination_dictionary_v1 &dictionary) noexcept {
    if (dictionary.global_destination_ids == nullptr || dictionary.count == 0u
        || dictionary.count > std::numeric_limits<std::uint32_t>::max())
        return false;
    for (std::uint64_t index = 0u; index < dictionary.count; ++index) {
        if (dictionary.global_destination_ids[index] == 0u
            || (index != 0u && dictionary.global_destination_ids[index - 1u]
                >= dictionary.global_destination_ids[index]))
            return false;
    }
    return true;
}

bool find_destination(const local_destination_dictionary_v1 &dictionary,
    std::uint64_t identity, std::uint32_t *local_index) noexcept {
    std::uint64_t begin = 0u;
    std::uint64_t end = dictionary.count;
    while (begin < end) {
        const std::uint64_t middle = begin + (end - begin) / 2u;
        if (dictionary.global_destination_ids[middle] < identity)
            begin = middle + 1u;
        else
            end = middle;
    }
    if (begin == dictionary.count
        || dictionary.global_destination_ids[begin] != identity)
        return false;
    *local_index = static_cast<std::uint32_t>(begin);
    return true;
}

} // namespace

transpose_status_v1 bind_transpose_local_maps_v1(
    const transpose_local_map_request_v1 &request,
    const transpose_local_map_storage_v1 &storage,
    transpose_cover_view_v1 *bound_cover,
    direct_gradient_order_v1 *gradient_order) noexcept {
    if (bound_cover == nullptr || gradient_order == nullptr
        || validate_transpose_cover_v1(request.cover)
            != transpose_status_v1::success
        || !valid_dictionary(request.destinations)
        || request.identity_order == nullptr || request.structure_epoch == 0u
        || request.value_generation == 0u || request.projection_order_id == 0u
        || storage.placements == nullptr
        || storage.placement_capacity < request.cover.placement_count
        || storage.projection_order == nullptr
        || storage.projection_order_capacity < request.cover.placement_count
        || storage.logical_order_to_projection == nullptr
        || storage.logical_order_capacity < request.cover.placement_count)
        return transpose_status_v1::invalid_argument;

    for (std::uint64_t position = 0u;
        position < request.cover.placement_count; ++position) {
        storage.placements[position] = request.cover.placements[position];
        std::uint32_t local_destination = invalid_local_index_v1;
        if (!find_destination(request.destinations,
                storage.placements[position].global_destination_id,
                &local_destination))
            return transpose_status_v1::invalid_cover;
        storage.placements[position].local_destination_index = local_destination;
        storage.projection_order[position] = {
            storage.placements[position].logical_edge_id, position};
    }

    std::uint64_t prior_identity = 0u;
    for (std::uint64_t logical_position = 0u;
        logical_position < request.cover.placement_count; ++logical_position) {
        const std::uint64_t projection_position =
            request.identity_order[logical_position];
        if (projection_position >= request.cover.placement_count)
            return transpose_status_v1::invalid_order;
        const std::uint64_t identity =
            storage.placements[projection_position].logical_edge_id;
        if (identity == 0u || (logical_position != 0u && identity <= prior_identity))
            return identity == prior_identity
                ? transpose_status_v1::duplicate_identity
                : transpose_status_v1::invalid_order;
        storage.logical_order_to_projection[logical_position] = projection_position;
        prior_identity = identity;
    }

    *bound_cover = request.cover;
    bound_cover->placements = storage.placements;
    *gradient_order = {request.structure_epoch, request.value_generation,
        request.projection_order_id, storage.projection_order,
        storage.logical_order_to_projection, request.cover.placement_count};
    return validate_direct_gradient_order_v1(*gradient_order);
}

transpose_status_v1 validate_direct_gradient_order_v1(
    const direct_gradient_order_v1 &order) noexcept {
    if (order.structure_epoch == 0u || order.value_generation == 0u
        || order.projection_order_id == 0u || order.projection_order == nullptr
        || order.logical_order_to_projection == nullptr || order.edge_count == 0u)
        return transpose_status_v1::invalid_argument;
    std::uint64_t prior_identity = 0u;
    for (std::uint64_t logical = 0u; logical < order.edge_count; ++logical) {
        const std::uint64_t projection = order.logical_order_to_projection[logical];
        if (projection >= order.edge_count
            || order.projection_order[projection].projection_position != projection)
            return transpose_status_v1::invalid_order;
        const std::uint64_t identity =
            order.projection_order[projection].logical_edge_id;
        if (identity == 0u || (logical != 0u && identity <= prior_identity))
            return transpose_status_v1::invalid_order;
        prior_identity = identity;
    }
    return transpose_status_v1::success;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose
