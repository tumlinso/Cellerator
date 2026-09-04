#include <Cellerator/compiler/discovery/import_co_support_and_overlap_discovery_v1.hh>

#include <algorithm>
#include <map>
#include <utility>

namespace Cellerator::compiler::discovery {

co_support_status_v1 discover_co_support_and_overlap_v1(
    support_relation_view_v1 relation, std::uint64_t source_count,
    std::uint32_t top_l, co_support_discovery_v1* output) noexcept {
    if (output == nullptr || !valid_persistent_atom_identity_v1(relation.relation_identity) ||
        relation.destination_offsets == nullptr ||
        (relation.edge_count != 0 && relation.source_identities == nullptr) ||
        relation.destination_count == 0 || relation.destination_offsets[0] != 0 ||
        relation.destination_offsets[relation.destination_count] != relation.edge_count)
        return co_support_status_v1::invalid_relation;
    if (source_count == 0 || top_l == 0) return co_support_status_v1::invalid_config;

    co_support_discovery_v1 result;
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> counts;
    try {
        result.source_prevalence.assign(source_count, 0);
        result.destination_convergence.resize(relation.destination_count);
        for (std::uint32_t destination = 0; destination < relation.destination_count;
             ++destination) {
            const auto begin = relation.destination_offsets[destination];
            const auto end = relation.destination_offsets[destination + 1];
            if (end < begin || end > relation.edge_count)
                return co_support_status_v1::invalid_relation;
            result.destination_convergence[destination] = end - begin;
            for (auto edge = begin; edge < end; ++edge) {
                const auto source = relation.source_identities[edge];
                if (source >= source_count ||
                    (edge != begin && source <= relation.source_identities[edge - 1]))
                    return co_support_status_v1::invalid_source;
                ++result.source_prevalence[source];
            }
            for (auto left = begin; left < end; ++left)
                for (auto right = left + 1; right < end; ++right) {
                    ++counts[{relation.source_identities[left],
                              relation.source_identities[right]}];
                    ++result.enumerated_pairs;
                }
        }
        for (const auto& entry : counts) {
            const auto first = entry.first.first;
            const auto second = entry.first.second;
            result.proposals.push_back({
                first, second, entry.second, result.source_prevalence[first],
                result.source_prevalence[second],
                result.source_prevalence[first] * result.source_prevalence[second],
                relation.destination_count});
        }
        std::sort(result.proposals.begin(), result.proposals.end(),
                  [](const auto& left, const auto& right) {
            if (left.observed_together != right.observed_together)
                return left.observed_together > right.observed_together;
            if (left.first_source != right.first_source)
                return left.first_source < right.first_source;
            return left.second_source < right.second_source;
        });
        if (result.proposals.size() > top_l) result.proposals.resize(top_l);
    } catch (...) {
        return co_support_status_v1::invalid_config;
    }
    *output = std::move(result);
    return co_support_status_v1::success;
}

}  // namespace Cellerator::compiler::discovery
