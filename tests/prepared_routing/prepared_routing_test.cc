#include <Cellerator/ops/sparse_support/prepared_routing.hh>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

using namespace cellerator::ops::sparse_support;
namespace indexing = cellerator::compute::indexing;

static bool same_projection(const prepared_entity_mechanism_index_v1 &left,
                            const prepared_entity_mechanism_index_v1 &right) {
    if (left.device_terms.size() != right.device_terms.size()
        || left.device_term_offsets != right.device_term_offsets
        || left.device_items.size() != right.device_items.size())
        return false;
    for (std::size_t i = 0u; i < left.device_terms.size(); ++i) {
        const auto &a = left.device_terms[i];
        const auto &b = right.device_terms[i];
        if (a.role != b.role || a.reserved != b.reserved
            || a.entity_id != b.entity_id)
            return false;
    }
    for (std::size_t i = 0u; i < left.device_items.size(); ++i) {
        const auto &a = left.device_items[i];
        const auto &b = right.device_items[i];
        if (a.logical_edge_id != b.logical_edge_id
            || a.logical_edge_ordinal != b.logical_edge_ordinal)
            return false;
    }
    return true;
}

static std::vector<std::uint64_t> host_query(
    const prepared_entity_mechanism_index_v1 &index,
    const std::vector<indexing::posting_term_v1> &terms) {
    const auto capacity = index.host_postings.entry_count;
    std::vector<indexing::indexed_work_item_v1> scratch(capacity), output_items(capacity);
    indexing::candidate_output_v1 output{
        output_items.empty() ? nullptr : output_items.data(), capacity, 0u, 0u};
    const auto status = indexing::query_reverse_postings_v1(index.host_postings,
        terms.empty() ? nullptr : terms.data(), terms.size(),
        indexing::posting_match_v1::any,
        {scratch.empty() ? nullptr : scratch.data(), capacity}, &output);
    assert(status == indexing::posting_status_v1::success);
    std::vector<std::uint64_t> result;
    for (std::uint64_t i = 0u; i < output.written; ++i)
        result.push_back(output_items[i].logical_edge_id - 1u);
    return result;
}

int main() {
    const prepared_routing_identity_v1 key{101u, 201u, 301u, 7u, 11u, 3u};
    const std::vector<std::vector<std::int64_t>> routes{
        {2, 1, 1}, {0, 2}, {}, {1, 2, 1}};
    prepared_entity_mechanism_index_v1 index;
    assert(prepare_entity_mechanism_index_v1(routes, 4u, 4u, key, &index)
        == prepared_routing_status_v1::success);
    auto initial_generation_key = key;
    initial_generation_key.route_generation = 0u;
    prepared_entity_mechanism_index_v1 initial_generation_index;
    assert(prepare_entity_mechanism_index_v1(routes, 4u, 4u,
        initial_generation_key, &initial_generation_index)
        == prepared_routing_status_v1::success);
    assert(initial_generation_index.identity.route_generation == 0u);
    assert(index.edges.size() == 3u); // mechanism 3 has no route row
    assert(index.participants.size() == 6u); // duplicate memberships collapsed
    assert(index.host_postings.relation_id == key.relation_id
        && index.host_postings.structure_id == key.structure_id
        && index.host_postings.structure_epoch == key.structure_epoch);
    assert(index.device_terms.size() == 3u
        && index.device_term_offsets.size() == 4u
        && index.device_items.size() == 6u);

    // Unsorted duplicate entity IDs still produce the exact sorted mechanism set.
    const auto result = host_query(index, {{0u, 0u, 4u}, {0u, 0u, 2u},
                                           {0u, 0u, 4u}});
    assert((result == std::vector<std::uint64_t>{0u, 1u, 2u}));
    assert((host_query(index, {{0u, 0u, 2u}})
        == std::vector<std::uint64_t>{0u, 2u}));
    assert((host_query(index, {{0u, 0u, 4u}})
        == std::vector<std::uint64_t>{1u, 2u}));
    assert(host_query(index, {}).empty());

    // Runtime activity changes query terms while the prepared relation stays
    // fixed. Route/value and binding changes produce distinct prepared keys.
    auto next_value_key = key;
    ++next_value_key.route_generation;
    prepared_entity_mechanism_index_v1 next_value_index;
    assert(prepare_entity_mechanism_index_v1(routes, 4u, 4u, next_value_key,
        &next_value_index) == prepared_routing_status_v1::success);
    assert(next_value_index.identity.route_generation
            != index.identity.route_generation
        && same_projection(next_value_index, index));
    auto next_binding_key = key;
    ++next_binding_key.binding_generation;
    prepared_entity_mechanism_index_v1 next_binding_index;
    assert(prepare_entity_mechanism_index_v1(routes, 4u, 4u, next_binding_key,
        &next_binding_index) == prepared_routing_status_v1::success);
    assert(next_binding_index.identity.binding_generation
            != index.identity.binding_generation
        && same_projection(next_binding_index, index));

    prepared_routing_query_requirements_v1 query{};
    assert(query_requirements_v1(index, 5u, 6u, &query)
        == prepared_routing_status_v1::success);
    assert(query.query_count == 30u && query.cursor_count == 30u
        && query.maximum_items_per_cell == 4u
        && query.output_item_capacity == 20u);
    assert(query_requirements_v1(index, 5u, 0u, &query)
        == prepared_routing_status_v1::success);
    assert(query.query_count == 0u && query.maximum_items_per_cell == 0u
        && query.output_item_capacity == 0u);
    assert(query_requirements_v1(index,
        std::numeric_limits<std::uint64_t>::max(), 2u, &query)
        == prepared_routing_status_v1::capacity_overflow);
    assert(query_requirements_v1(index,
        std::numeric_limits<std::uint32_t>::max(), 1u, &query)
        == prepared_routing_status_v1::capacity_overflow);

    const std::vector<std::vector<std::int64_t>> invalid_route{{-1}};
    assert(prepare_entity_mechanism_index_v1(invalid_route, 1u, 4u, key, &index)
        == prepared_routing_status_v1::mechanism_out_of_range);
    assert(index.entity_count == 4u && index.host_postings.entry_count == 6u);
    assert(prepare_entity_mechanism_index_v1(routes, 3u, 4u, key, &index)
        == prepared_routing_status_v1::entity_count_mismatch);
    auto invalid_key = key;
    invalid_key.owner_identity = 0u;
    assert(prepare_entity_mechanism_index_v1(routes, 4u, 4u, invalid_key, &index)
        == prepared_routing_status_v1::invalid_identity);

    const std::vector<std::vector<std::int64_t>> no_routes(3);
    prepared_entity_mechanism_index_v1 empty_index;
    assert(prepare_entity_mechanism_index_v1(no_routes, 3u, 2u, key, &empty_index)
        == prepared_routing_status_v1::success);
    assert(empty_index.host_postings.entry_count == 0u
        && empty_index.device_terms.empty() && empty_index.device_items.empty());
    assert(host_query(empty_index, {{0u, 0u, 1u}}).empty());
}
