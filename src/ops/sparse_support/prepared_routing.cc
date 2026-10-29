#include <Cellerator/ops/sparse_support/prepared_routing.hh>

#include <algorithm>
#include <limits>
#include <utility>

namespace cellerator::ops::sparse_support {
namespace {

constexpr std::uint64_t maximum_cuda_grid_x =
    static_cast<std::uint64_t>(std::numeric_limits<int>::max());

bool multiply_fits_u64(std::uint64_t left, std::uint64_t right,
                       std::uint64_t *result) noexcept {
    if (result == nullptr
        || (right != 0u && left > std::numeric_limits<std::uint64_t>::max() / right))
        return false;
    *result = left * right;
    return true;
}

bool valid_identity(const prepared_routing_identity_v1 &identity) noexcept {
    return identity.owner_identity != 0u && identity.relation_id != 0u
        && identity.structure_id != 0u && identity.structure_epoch != 0u
        && identity.binding_generation != 0u;
}

} // namespace

compute::indexing::cuda_reverse_posting_view_v1
prepared_entity_mechanism_index_v1::bind_device_view(
    const compute::indexing::posting_term_v1 *terms,
    const std::uint64_t *term_offsets,
    const compute::indexing::indexed_work_item_v1 *items) const noexcept {
    return {identity.relation_id, identity.structure_id, identity.structure_epoch,
        terms, term_offsets,
        static_cast<std::uint64_t>(device_terms.size()), items,
        static_cast<std::uint64_t>(device_items.size())};
}

prepared_routing_status_v1 prepare_entity_mechanism_index_v1(
    const std::vector<std::vector<std::int64_t>> &entity_mechanisms,
    std::uint64_t entity_count, std::uint64_t mechanism_count,
    prepared_routing_identity_v1 identity,
    prepared_entity_mechanism_index_v1 *output) {
    if (output == nullptr || entity_count == 0u || mechanism_count == 0u
        || entity_count > static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max())
        || mechanism_count > static_cast<std::uint64_t>(
            std::numeric_limits<std::int32_t>::max()))
        return prepared_routing_status_v1::invalid_argument;
    if (!valid_identity(identity)) return prepared_routing_status_v1::invalid_identity;
    if (entity_mechanisms.size() != entity_count)
        return prepared_routing_status_v1::entity_count_mismatch;

    prepared_entity_mechanism_index_v1 candidate;
    candidate.identity = identity;
    candidate.entity_count = entity_count;
    candidate.mechanism_count = mechanism_count;
    std::vector<std::vector<std::uint64_t>> entities_by_mechanism(
        static_cast<std::size_t>(mechanism_count));
    for (std::uint64_t entity = 0u; entity < entity_count; ++entity) {
        std::vector<std::uint64_t> routes;
        routes.reserve(entity_mechanisms[entity].size());
        for (const auto raw_mechanism : entity_mechanisms[entity]) {
            if (raw_mechanism < 0
                || static_cast<std::uint64_t>(raw_mechanism) >= mechanism_count)
                return prepared_routing_status_v1::mechanism_out_of_range;
            routes.push_back(static_cast<std::uint64_t>(raw_mechanism));
        }
        std::sort(routes.begin(), routes.end());
        routes.erase(std::unique(routes.begin(), routes.end()), routes.end());
        candidate.maximum_routes_per_entity = std::max(
            candidate.maximum_routes_per_entity,
            static_cast<std::uint64_t>(routes.size()));
        for (const auto mechanism : routes)
            entities_by_mechanism[mechanism].push_back(entity + 1u);
    }

    std::uint64_t participant_count = 0u;
    for (std::uint64_t mechanism = 0u; mechanism < mechanism_count; ++mechanism) {
        auto &entities = entities_by_mechanism[mechanism];
        if (entities.empty()) continue;
        std::sort(entities.begin(), entities.end());
        entities.erase(std::unique(entities.begin(), entities.end()), entities.end());
        if (entities.size() > std::numeric_limits<std::uint32_t>::max()
            || participant_count > std::numeric_limits<std::uint64_t>::max()
                - entities.size())
            return prepared_routing_status_v1::capacity_overflow;
        candidate.edges.push_back({mechanism + 1u, participant_count,
            static_cast<std::uint32_t>(entities.size()), 0u});
        participant_count += entities.size();
        for (const auto entity_id : entities)
            candidate.participants.push_back({0u, 0u, entity_id});
    }

    const compute::support::logical_support_view_v1 relation{
        compute::support::logical_support_schema_version_v1, 0u,
        identity.relation_id, identity.structure_id, identity.structure_epoch,
        candidate.edges.empty() ? nullptr : candidate.edges.data(),
        static_cast<std::uint64_t>(candidate.edges.size()),
        candidate.participants.empty() ? nullptr : candidate.participants.data(),
        static_cast<std::uint64_t>(candidate.participants.size())};
    const compute::support::support_snapshot_pair_v1 snapshot{
        &relation, nullptr, identity.structure_epoch};
    if (compute::support::validate_support_snapshot_pair_v1(snapshot)
        != compute::support::support_snapshot_status_v1::success)
        return prepared_routing_status_v1::invalid_support;

    const auto requirements = compute::indexing::query_reverse_posting_requirements_v1(
        snapshot);
    if (requirements.record_capacity > std::numeric_limits<std::size_t>::max()
        || requirements.maximum_unique_entries > std::numeric_limits<std::size_t>::max())
        return prepared_routing_status_v1::capacity_overflow;
    candidate.record_scratch.resize(
        static_cast<std::size_t>(requirements.record_capacity));
    candidate.posting_entries.resize(
        static_cast<std::size_t>(requirements.maximum_unique_entries));
    const auto posting_status = compute::indexing::build_reverse_postings_v1(
        snapshot,
        candidate.record_scratch.empty() ? nullptr : candidate.record_scratch.data(),
        requirements.record_capacity,
        candidate.posting_entries.empty() ? nullptr : candidate.posting_entries.data(),
        requirements.maximum_unique_entries, &candidate.host_postings);
    if (posting_status != compute::indexing::posting_status_v1::success)
        return prepared_routing_status_v1::invalid_support;

    const auto projection = compute::indexing::cuda_posting_projection_requirements_v1(
        candidate.host_postings);
    if (projection.term_count > std::numeric_limits<std::size_t>::max()
        || projection.item_count > std::numeric_limits<std::size_t>::max())
        return prepared_routing_status_v1::capacity_overflow;
    candidate.device_terms.resize(static_cast<std::size_t>(projection.term_count));
    if (projection.term_count == std::numeric_limits<std::uint64_t>::max())
        return prepared_routing_status_v1::capacity_overflow;
    candidate.device_term_offsets.resize(
        static_cast<std::size_t>(projection.term_count + 1u));
    candidate.device_items.resize(static_cast<std::size_t>(projection.item_count));
    compute::indexing::cuda_posting_prepare_requirements_v1 prepared{};
    const auto projection_status = compute::indexing::build_cuda_posting_projection_v1(
        candidate.host_postings,
        candidate.device_terms.empty() ? nullptr : candidate.device_terms.data(),
        projection.term_count,
        candidate.device_term_offsets.data(), candidate.device_term_offsets.size(),
        candidate.device_items.empty() ? nullptr : candidate.device_items.data(),
        projection.item_count, &prepared);
    if (projection_status != compute::indexing::cuda_posting_prepare_status_v1::success
        || prepared.term_count != projection.term_count
        || prepared.item_count != projection.item_count)
        return prepared_routing_status_v1::insufficient_workspace;

    *output = std::move(candidate);
    output->host_postings.entries = output->posting_entries.empty()
        ? nullptr : output->posting_entries.data();
    return prepared_routing_status_v1::success;
}

prepared_routing_status_v1 query_requirements_v1(
    const prepared_entity_mechanism_index_v1 &index,
    std::uint64_t cells, std::uint64_t participants_per_cell,
    prepared_routing_query_requirements_v1 *output) noexcept {
    if (output == nullptr || index.entity_count == 0u
        || index.mechanism_count == 0u || index.identity.structure_epoch == 0u)
        return prepared_routing_status_v1::invalid_argument;
    *output = {};
    std::uint64_t query_count = 0u;
    if (!multiply_fits_u64(cells, participants_per_cell, &query_count))
        return prepared_routing_status_v1::capacity_overflow;
    std::uint64_t per_cell_bound = 0u;
    if (!multiply_fits_u64(participants_per_cell,
            index.maximum_routes_per_entity, &per_cell_bound))
        return prepared_routing_status_v1::capacity_overflow;
    per_cell_bound = std::min(per_cell_bound, index.mechanism_count);
    std::uint64_t output_capacity = 0u;
    if (!multiply_fits_u64(cells, per_cell_bound, &output_capacity))
        return prepared_routing_status_v1::capacity_overflow;
    if (query_count > maximum_cuda_grid_x || cells > maximum_cuda_grid_x)
        return prepared_routing_status_v1::capacity_overflow;
    *output = {query_count, query_count, per_cell_bound, output_capacity};
    return prepared_routing_status_v1::success;
}

} // namespace cellerator::ops::sparse_support
