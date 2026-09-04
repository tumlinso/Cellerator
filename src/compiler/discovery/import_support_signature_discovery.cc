#include <Cellerator/compiler/discovery/import_support_signature_discovery_v1.hh>

#include <algorithm>
#include <limits>

namespace Cellerator::compiler::discovery {
namespace {

std::uint64_t mix(std::uint64_t value) noexcept {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

bool better(const support_signature_proposal_v1& left,
            const support_signature_proposal_v1& right) noexcept {
    if (left.matching_minima != right.matching_minima)
        return left.matching_minima > right.matching_minima;
    if (left.first_destination != right.first_destination)
        return left.first_destination < right.first_destination;
    return left.second_destination < right.second_destination;
}

}  // namespace

support_signature_status_v1 discover_support_signatures_v1(
    support_relation_view_v1 relation,
    support_signature_config_v1 config,
    support_signature_discovery_v1* output) noexcept {
    if (output == nullptr || !valid_persistent_atom_identity_v1(relation.relation_identity) ||
        relation.destination_offsets == nullptr ||
        (relation.edge_count != 0 && relation.source_identities == nullptr) ||
        relation.destination_count == 0)
        return support_signature_status_v1::invalid_relation;
    if (relation.destination_offsets[0] != 0 ||
        relation.destination_offsets[relation.destination_count] != relation.edge_count)
        return support_signature_status_v1::invalid_offsets;
    for (std::uint32_t destination = 0; destination < relation.destination_count; ++destination)
        if (relation.destination_offsets[destination] >
            relation.destination_offsets[destination + 1])
            return support_signature_status_v1::invalid_offsets;
    if (config.sketch_size == 0 || config.top_l == 0 || config.seed_namespace == 0 ||
        !valid_persistent_atom_identity_v1(config.biological_stratum))
        return support_signature_status_v1::invalid_config;

    support_signature_discovery_v1 result;
    try {
        result.minima.assign(static_cast<std::size_t>(relation.destination_count) *
                                 config.sketch_size,
                             std::numeric_limits<std::uint64_t>::max());
        for (std::uint32_t destination = 0; destination < relation.destination_count;
             ++destination) {
            const auto begin = relation.destination_offsets[destination];
            const auto end = relation.destination_offsets[destination + 1];
            for (auto edge = begin; edge < end; ++edge) {
                for (std::uint32_t sketch = 0; sketch < config.sketch_size; ++sketch) {
                    const auto hash = mix(relation.source_identities[edge] ^
                        mix(config.seed_namespace + sketch));
                    auto& minimum = result.minima[
                        static_cast<std::size_t>(destination) * config.sketch_size + sketch];
                    minimum = std::min(minimum, hash);
                    ++result.hashed_edges;
                }
            }
        }
        for (std::uint32_t left = 0; left < relation.destination_count; ++left) {
            for (std::uint32_t right = left + 1; right < relation.destination_count; ++right) {
                support_signature_proposal_v1 proposal;
                proposal.first_destination = left;
                proposal.second_destination = right;
                proposal.sketch_size = config.sketch_size;
                proposal.first_degree = relation.destination_offsets[left + 1] -
                    relation.destination_offsets[left];
                proposal.second_degree = relation.destination_offsets[right + 1] -
                    relation.destination_offsets[right];
                proposal.biological_stratum = config.biological_stratum;
                for (std::uint32_t sketch = 0; sketch < config.sketch_size; ++sketch)
                    proposal.matching_minima +=
                        result.minima[static_cast<std::size_t>(left) * config.sketch_size + sketch] ==
                        result.minima[static_cast<std::size_t>(right) * config.sketch_size + sketch];
                ++result.compared_pairs;
                result.proposals.push_back(proposal);
            }
        }
        std::sort(result.proposals.begin(), result.proposals.end(), better);
        if (result.proposals.size() > config.top_l) result.proposals.resize(config.top_l);
    } catch (...) {
        return support_signature_status_v1::invalid_config;
    }
    *output = std::move(result);
    return support_signature_status_v1::success;
}

}  // namespace Cellerator::compiler::discovery
