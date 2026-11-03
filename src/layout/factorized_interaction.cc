#include <Cellerator/layout/factorized_interaction.hh>
#include "factorized_interaction_test_support.hh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cellerator::layout {
namespace {

constexpr std::uint8_t invalid_local_ref = 0xffu;

using IndexedPair = std::pair<std::uint64_t, std::uint64_t>;

struct UniqueSpan {
    std::uint64_t begin = 0;
    std::uint32_t count = 0;
};

struct MechanismMetadata {
    UniqueSpan inputs{};
    UniqueSpan contexts{};
};

struct PacketBuild {
    std::uint32_t arity = 0;
    bool direct_context = false;
    std::vector<std::uint64_t> mechanisms;
    std::vector<std::uint64_t> inputs;
    std::vector<std::uint64_t> contexts;
    std::vector<std::uint64_t> outputs;
};

struct CandidateScore {
    std::uint32_t inputs = 0;
    std::uint32_t contexts = 0;
    std::uint32_t outputs = 0;
};

struct AffinityPair {
    std::uint32_t arity = 0;
    std::uint64_t key = 0;
    std::uint64_t mechanism = 0;
};

struct AffinityRange {
    std::uint32_t arity = 0;
    std::uint64_t key = 0;
    std::size_t begin = 0, end = 0;
};

struct AffinityCursorIndex {
    std::vector<AffinityPair> entries;
    std::vector<AffinityRange> ranges;

    static constexpr std::size_t candidate_window =
        factorized_interaction_max_packet_mechanisms;

    void initialize(std::vector<AffinityPair> postings,
                    const std::vector<FactorizedInteractionMechanism>& mechanisms) {
        std::sort(postings.begin(), postings.end(), [&](const auto& lhs, const auto& rhs) {
            if (lhs.arity != rhs.arity) return lhs.arity < rhs.arity;
            if (lhs.key != rhs.key) return lhs.key < rhs.key;
            const auto& left = mechanisms[static_cast<std::size_t>(lhs.mechanism)];
            const auto& right = mechanisms[static_cast<std::size_t>(rhs.mechanism)];
            return left.logical_id < right.logical_id;
        });
        entries = std::move(postings);
        std::size_t range_count = 0;
        for (std::size_t i = 0; i < entries.size(); ++i)
            if (i == 0 || entries[i].arity != entries[i - 1].arity
                || entries[i].key != entries[i - 1].key)
                ++range_count;
        ranges.reserve(range_count);
        std::size_t begin = 0;
        while (begin < entries.size()) {
            std::size_t end = begin + 1;
            while (end < entries.size()
                   && entries[end].arity == entries[begin].arity
                   && entries[end].key == entries[begin].key)
                ++end;
            ranges.push_back({entries[begin].arity, entries[begin].key,
                              begin, end});
            begin = end;
        }
    }

    template<class Visitor>
    void visit_window(std::uint32_t arity, std::uint64_t key,
                      std::uint64_t minimum_logical_id,
                      const std::vector<std::uint8_t>& assigned,
                      const std::vector<FactorizedInteractionMechanism>& mechanisms,
                      test_support::FactorizedInteractionPreparationWork* work,
                      Visitor&& visitor) const {
        if (work != nullptr && work->affinity_key_lookups != UINT64_MAX)
            ++work->affinity_key_lookups;
        const auto range = std::lower_bound(ranges.begin(), ranges.end(),
            std::pair<std::uint32_t, std::uint64_t>{arity, key},
            [](const AffinityRange& lhs, const auto& rhs) {
                return lhs.arity < rhs.first
                    || (lhs.arity == rhs.first && lhs.key < rhs.second);
            });
        if (range == ranges.end() || range->arity != arity || range->key != key)
            return;

        auto cursor = entries.begin() + static_cast<std::ptrdiff_t>(range->begin);
        const auto end = entries.begin() + static_cast<std::ptrdiff_t>(range->end);
        cursor = std::lower_bound(cursor, end, minimum_logical_id,
            [&](const AffinityPair& entry, std::uint64_t logical_id) {
                return mechanisms[static_cast<std::size_t>(entry.mechanism)].logical_id
                    < logical_id;
            });
        std::size_t visited = 0;
        for (; cursor != end && visited < candidate_window; ++cursor, ++visited) {
            if (work != nullptr && work->affinity_entries_visited != UINT64_MAX)
                ++work->affinity_entries_visited;
            if (!assigned[static_cast<std::size_t>(cursor->mechanism)])
                visitor(cursor->mechanism);
        }
    }
};

void add_work_count(std::uint64_t& counter, std::size_t amount) noexcept {
    const auto value = static_cast<std::uint64_t>(amount);
    if (value > UINT64_MAX - counter) counter = UINT64_MAX;
    else counter += value;
}

std::size_t checked_size(std::uint64_t value, const char* what) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::overflow_error(what);
    return static_cast<std::size_t>(value);
}

std::size_t posting_offset_count(std::uint64_t extent, const char* what) {
    if (extent >= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::overflow_error(what);
    return static_cast<std::size_t>(extent) + 1;
}

void validate_options(const FactorizedInteractionOptions& options) {
    if (options.max_mechanisms == 0
        || options.max_mechanisms > factorized_interaction_max_packet_mechanisms)
        throw std::invalid_argument("factorized packet mechanism limit is outside [1,32]");
    if (options.max_inputs == 0
        || options.max_inputs > factorized_interaction_max_packet_inputs)
        throw std::invalid_argument("factorized packet input limit is outside [1,64]");
    if (options.max_contexts > factorized_interaction_max_packet_contexts)
        throw std::invalid_argument("factorized packet context limit exceeds 64");
    if (options.pattern_id == 0 || options.structure_id == 0
        || options.structure_epoch == 0)
        throw std::invalid_argument("factorized structure identities and epoch must be nonzero");
}

void validate_extents(const FactorizedInteractionExtents& extents) {
    (void)posting_offset_count(extents.inputs,
        "factorized input posting extent overflows host size");
    (void)posting_offset_count(extents.contexts,
        "factorized context posting extent overflows host size");
    (void)posting_offset_count(extents.outputs,
        "factorized output posting extent overflows host size");
    (void)posting_offset_count(extents.alpha_rows,
        "factorized alpha posting extent overflows host size");
    (void)posting_offset_count(extents.biases,
        "factorized bias posting extent overflows host size");
    (void)posting_offset_count(extents.gammas,
        "factorized gamma posting extent overflows host size");
}

std::vector<MechanismMetadata> collect_metadata(
    const std::vector<FactorizedInteractionMechanism>& input,
    const FactorizedInteractionExtents& extents,
    std::vector<std::uint64_t>& unique_inputs,
    std::vector<std::uint64_t>& unique_contexts,
    std::vector<std::uint64_t>& stable_order) {
    if (input.size() > static_cast<std::size_t>(UINT64_MAX))
        throw std::overflow_error("factorized mechanism count exceeds uint64");

    const auto mechanism_count = input.size();
    std::vector<MechanismMetadata> metadata(mechanism_count);
    std::vector<IndexedPair> input_pairs;
    std::vector<IndexedPair> context_pairs;
    std::vector<std::uint64_t> logical_ids;
    std::uint64_t total_terms = 0;
    std::size_t total_input_slots = 0;

    logical_ids.reserve(mechanism_count);
    for (std::size_t m = 0; m < mechanism_count; ++m) {
        const auto& mechanism = input[m];
        if (mechanism.logical_id == 0)
            throw std::invalid_argument("factorized logical mechanism IDs must be nonzero");
        logical_ids.push_back(mechanism.logical_id);
        if (mechanism.arity < 2 || mechanism.arity > factorized_interaction_max_arity)
            throw std::invalid_argument("factorized mechanism arity must be 2, 3, or 4");
        if (mechanism.target >= extents.outputs)
            throw std::invalid_argument("factorized target is out of range");
        if (mechanism.alpha >= extents.alpha_rows)
            throw std::invalid_argument("factorized alpha row is out of range");
        if (mechanism.bias >= extents.biases)
            throw std::invalid_argument("factorized bias scalar is out of range");
        if (mechanism.terms.size() > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("factorized mechanism context term count exceeds uint32");
        if (mechanism.terms.size() > UINT64_MAX - total_terms)
            throw std::overflow_error("factorized total context term count exceeds uint64");
        total_terms += static_cast<std::uint64_t>(mechanism.terms.size());
        if (mechanism.arity > std::numeric_limits<std::size_t>::max() - total_input_slots)
            throw std::overflow_error("factorized participant incidence count exceeds host size");
        total_input_slots += mechanism.arity;
    }

    std::sort(logical_ids.begin(), logical_ids.end());
    if (std::adjacent_find(logical_ids.begin(), logical_ids.end()) != logical_ids.end())
        throw std::invalid_argument("factorized logical mechanism IDs must be unique");

    input_pairs.reserve(total_input_slots);
    context_pairs.reserve(checked_size(total_terms,
        "factorized context term count exceeds host size"));
    unique_inputs.reserve(total_input_slots);
    unique_contexts.reserve(checked_size(total_terms,
        "factorized context term count exceeds host size"));
    stable_order.resize(mechanism_count);
    std::iota(stable_order.begin(), stable_order.end(), std::uint64_t{0});
    std::sort(stable_order.begin(), stable_order.end(), [&](std::uint64_t lhs,
                                                            std::uint64_t rhs) {
        return input[static_cast<std::size_t>(lhs)].logical_id
             < input[static_cast<std::size_t>(rhs)].logical_id;
    });

    for (std::size_t m = 0; m < mechanism_count; ++m) {
        const auto& mechanism = input[m];
        for (std::uint32_t role = 0; role < mechanism.arity; ++role) {
            const auto occurrence = mechanism.inputs[role];
            if (occurrence >= extents.inputs)
                throw std::invalid_argument("factorized participant occurrence is out of range");
            input_pairs.emplace_back(occurrence, static_cast<std::uint64_t>(m));
        }

        for (const auto& term : mechanism.terms) {
            if (term.context >= extents.contexts)
                throw std::invalid_argument("factorized context occurrence is out of range");
            if (term.gamma >= extents.gammas)
                throw std::invalid_argument("factorized gamma scalar is out of range");
            context_pairs.emplace_back(term.context, static_cast<std::uint64_t>(m));
        }
    }

    const auto sort_unique_pairs = [](std::vector<IndexedPair>& pairs) {
        std::sort(pairs.begin(), pairs.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.second < rhs.second
                || (lhs.second == rhs.second && lhs.first < rhs.first);
        });
        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    };
    sort_unique_pairs(input_pairs);
    sort_unique_pairs(context_pairs);

    std::uint64_t current_mechanism = UINT64_MAX;
    for (const auto& pair : input_pairs) {
        if (pair.second != current_mechanism) {
            current_mechanism = pair.second;
            metadata[static_cast<std::size_t>(current_mechanism)].inputs.begin =
                static_cast<std::uint64_t>(unique_inputs.size());
        }
        auto& count = metadata[static_cast<std::size_t>(current_mechanism)].inputs.count;
        if (count == UINT32_MAX)
            throw std::overflow_error("factorized unique input count exceeds uint32");
        ++count;
        unique_inputs.push_back(pair.first);
    }
    current_mechanism = UINT64_MAX;
    for (const auto& pair : context_pairs) {
        if (pair.second != current_mechanism) {
            current_mechanism = pair.second;
            metadata[static_cast<std::size_t>(current_mechanism)].contexts.begin =
                static_cast<std::uint64_t>(unique_contexts.size());
        }
        auto& count = metadata[static_cast<std::size_t>(current_mechanism)].contexts.count;
        if (count == UINT32_MAX)
            throw std::overflow_error("factorized unique context count exceeds uint32");
        ++count;
        unique_contexts.push_back(pair.first);
    }
    return metadata;
}

bool union_fits(const std::vector<std::uint64_t>& existing,
                const std::uint64_t* additions, std::size_t addition_count,
                std::uint32_t limit) {
    if (addition_count == 0) return existing.size() <= limit;
    std::size_t i = 0, j = 0, count = 0;
    while (i < existing.size() || j < addition_count) {
        std::uint64_t next = 0;
        if (j == addition_count
            || (i < existing.size() && existing[i] < additions[j])) {
            next = existing[i++];
        } else if (i == existing.size() || additions[j] < existing[i]) {
            next = additions[j++];
        } else {
            next = existing[i++];
            ++j;
        }
        (void)next;
        if (++count > limit) return false;
    }
    return true;
}

void insert_sorted_unique(std::vector<std::uint64_t>& existing,
                          const std::uint64_t* additions,
                          std::size_t addition_count) {
    for (std::size_t i = 0; i < addition_count; ++i) {
        const auto position = std::lower_bound(existing.begin(), existing.end(), additions[i]);
        if (position == existing.end() || *position != additions[i])
            existing.insert(position, additions[i]);
    }
}

const std::uint64_t* span_begin(const std::vector<std::uint64_t>& values,
                                UniqueSpan span) {
    if (values.empty()) return nullptr;
    return values.data() + static_cast<std::size_t>(span.begin);
}

bool packet_can_add(const PacketBuild& packet, std::uint64_t candidate,
                    const FactorizedInteractionMechanism& mechanism,
                    const std::vector<MechanismMetadata>& metadata,
                    const std::vector<std::uint64_t>& unique_inputs,
                    const std::vector<std::uint64_t>& unique_contexts,
                    const FactorizedInteractionOptions& options) {
    if (packet.direct_context || packet.mechanisms.size() >= options.max_mechanisms
        || mechanism.arity != packet.arity)
        return false;
    const auto& input_span = metadata[static_cast<std::size_t>(candidate)].inputs;
    const auto& context_span = metadata[static_cast<std::size_t>(candidate)].contexts;
    if (context_span.count > options.max_contexts) return false;
    return union_fits(packet.inputs, span_begin(unique_inputs, input_span),
                      input_span.count, options.max_inputs)
        && union_fits(packet.contexts, span_begin(unique_contexts, context_span),
                      context_span.count, options.max_contexts);
}

void add_to_packet(PacketBuild& packet, std::uint64_t canonical,
                   const FactorizedInteractionMechanism& mechanism,
                   const std::vector<MechanismMetadata>& metadata,
                   const std::vector<std::uint64_t>& unique_inputs,
                   const std::vector<std::uint64_t>& unique_contexts,
                   const FactorizedInteractionOptions& options) {
    const auto& input_span = metadata[static_cast<std::size_t>(canonical)].inputs;
    const auto& context_span = metadata[static_cast<std::size_t>(canonical)].contexts;
    packet.mechanisms.push_back(canonical);
    insert_sorted_unique(packet.inputs, span_begin(unique_inputs, input_span),
                         input_span.count);
    const auto output_position = std::lower_bound(packet.outputs.begin(), packet.outputs.end(),
                                                  mechanism.target);
    if (output_position == packet.outputs.end() || *output_position != mechanism.target)
        packet.outputs.insert(output_position, mechanism.target);
    if (context_span.count > options.max_contexts) {
        if (packet.mechanisms.size() != 1)
            throw std::logic_error("oversized context mechanism must own a singleton packet");
        packet.direct_context = true;
        packet.contexts.clear();
    } else if (!packet.direct_context) {
        insert_sorted_unique(packet.contexts, span_begin(unique_contexts, context_span),
                             context_span.count);
    }
}

struct PostingPair {
    std::uint64_t row = 0;
    std::uint64_t entry = 0;
};

FactorizedInteractionPostings make_postings(std::uint64_t row_extent,
                                             std::vector<PostingPair> pairs) {
    const auto offset_count = posting_offset_count(row_extent,
        "factorized posting row extent overflows host size");
    std::sort(pairs.begin(), pairs.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.row < rhs.row || (lhs.row == rhs.row && lhs.entry < rhs.entry);
    });
    for (const auto& pair : pairs)
        if (pair.row >= row_extent)
            throw std::logic_error("factorized posting key exceeds its declared extent");

    FactorizedInteractionPostings result;
    result.offsets.resize(offset_count);
    result.entries.reserve(pairs.size());
    std::size_t cursor = 0;
    for (std::size_t row = 0; row < offset_count; ++row) {
        result.offsets[row] = static_cast<std::uint64_t>(cursor);
        while (cursor < pairs.size() && pairs[cursor].row == row) {
            result.entries.push_back(pairs[cursor].entry);
            ++cursor;
        }
    }
    if (cursor != pairs.size())
        throw std::logic_error("factorized posting construction did not consume every entry");
    return result;
}

std::uint64_t find_ref(const std::vector<std::uint64_t>& dictionary,
                       std::uint64_t value, const char* what) {
    const auto found = std::lower_bound(dictionary.begin(), dictionary.end(), value);
    if (found == dictionary.end() || *found != value)
        throw std::logic_error(what);
    const auto result = static_cast<std::uint64_t>(found - dictionary.begin());
    if (result >= invalid_local_ref)
        throw std::logic_error("factorized local reference cannot use sentinel value");
    return result;
}

FactorizedInteractionLayout build_layout(
    const std::vector<FactorizedInteractionMechanism>& input,
    FactorizedInteractionExtents extents,
    FactorizedInteractionOptions options,
    test_support::FactorizedInteractionPreparationWork* work = nullptr) {
    validate_options(options);
    validate_extents(extents);

    FactorizedInteractionLayout layout;
    layout.options = options;
    layout.extents = extents;
    layout.mechanisms.resize(input.size());

    std::vector<std::uint64_t> unique_inputs;
    std::vector<std::uint64_t> unique_contexts;
    std::vector<std::uint64_t> stable_order;
    auto metadata = collect_metadata(input, extents, unique_inputs,
                                     unique_contexts, stable_order);
    std::array<std::vector<std::uint64_t>, factorized_interaction_max_arity + 1>
        stable_by_arity;
    std::vector<std::size_t> arity_position(input.size());
    for (const auto canonical : stable_order) {
        const auto index = static_cast<std::size_t>(canonical);
        if (metadata[index].contexts.count > options.max_contexts) continue;
        const auto arity = input[index].arity;
        arity_position[index] = stable_by_arity[arity].size();
        stable_by_arity[arity].push_back(canonical);
    }
    for (const auto& item : metadata)
        if (item.inputs.count > options.max_inputs)
            throw std::invalid_argument(
                "one factorized mechanism exceeds the packet input budget");

    std::uint64_t total_terms = 0;
    std::size_t total_term_storage = 0;
    for (const auto& mechanism : input) {
        if (mechanism.terms.size() > std::numeric_limits<std::size_t>::max()
                                         - total_term_storage)
            throw std::overflow_error("factorized term storage exceeds host size");
        total_term_storage += mechanism.terms.size();
    }
    layout.terms.reserve(total_term_storage);
    for (std::size_t m = 0; m < input.size(); ++m) {
        const auto& source = input[m];
        auto& record = layout.mechanisms[m];
        record.logical_id = source.logical_id;
        record.target = source.target;
        record.alpha = source.alpha;
        record.bias = source.bias;
        record.arity = source.arity;
        record.term_begin = total_terms;
        record.term_count = static_cast<std::uint32_t>(source.terms.size());
        for (std::uint32_t role = 0; role < source.arity; ++role)
            record.inputs[role] = source.inputs[role];
        for (const auto& term : source.terms) {
            layout.terms.push_back({term.context, term.gamma,
                                    static_cast<std::uint64_t>(m)});
            ++total_terms;
        }
    }

    // Flat sorted inverted lists avoid dense affinity metadata. Each key query
    // examines one packet-width window in stable logical-ID order, bounding
    // high-degree postings while retaining deterministic local-affinity grouping.
    std::vector<AffinityPair> input_affinity, context_affinity, target_affinity;
    for (std::size_t ordinal = 0; ordinal < input.size(); ++ordinal) {
        const auto& input_span = metadata[ordinal].inputs;
        const auto* input_ids = span_begin(unique_inputs, input_span);
        for (std::uint32_t i = 0; i < input_span.count; ++i)
            input_affinity.push_back({input[ordinal].arity, input_ids[i],
                                      static_cast<std::uint64_t>(ordinal)});
        const auto& context_span = metadata[ordinal].contexts;
        const auto* context_ids = span_begin(unique_contexts, context_span);
        for (std::uint32_t i = 0; i < context_span.count; ++i)
            context_affinity.push_back({input[ordinal].arity, context_ids[i],
                                        static_cast<std::uint64_t>(ordinal)});
        target_affinity.push_back({input[ordinal].arity, input[ordinal].target,
                                   static_cast<std::uint64_t>(ordinal)});
    }
    AffinityCursorIndex by_input, by_context, by_target;
    by_input.initialize(std::move(input_affinity), input);
    by_context.initialize(std::move(context_affinity), input);
    by_target.initialize(std::move(target_affinity), input);

    std::vector<std::uint8_t> assigned(input.size(), 0);
    std::vector<std::uint64_t> remaining_generation(input.size(), 0);
    std::vector<CandidateScore> scores(input.size());
    std::vector<std::uint64_t> touched;
    touched.reserve(input.size());
    std::uint64_t generation = 0;
    std::vector<PacketBuild> packet_builds;
    std::array<std::size_t, factorized_interaction_max_arity + 1> regular_by_arity{};
    std::size_t direct_context_count = 0;
    for (std::size_t m = 0; m < input.size(); ++m) {
        if (metadata[m].contexts.count > options.max_contexts)
            ++direct_context_count;
        else
            ++regular_by_arity[input[m].arity];
    }
    std::size_t estimated_packets = direct_context_count;
    for (std::uint32_t arity = 2; arity <= factorized_interaction_max_arity; ++arity) {
        const auto count = regular_by_arity[arity];
        estimated_packets += count / options.max_mechanisms
            + (count % options.max_mechanisms != 0 ? 1 : 0);
    }
    packet_builds.reserve(estimated_packets);

    const auto score_candidates = [&](const PacketBuild& packet,
                                      std::uint64_t minimum_logical_id,
                                      std::vector<std::uint64_t>& candidate_ids,
                                      std::uint64_t current_generation) {
        candidate_ids.clear();
        const auto visit = [&](const AffinityCursorIndex& index, std::uint64_t key,
                               std::uint32_t component) {
            index.visit_window(packet.arity, key, minimum_logical_id, assigned,
                input, work, [&](std::uint64_t candidate_id) {
                    const auto candidate = static_cast<std::size_t>(candidate_id);
                    if (remaining_generation[candidate] != current_generation) {
                        remaining_generation[candidate] = current_generation;
                        scores[candidate] = {};
                        candidate_ids.push_back(candidate_id);
                    }
                    auto& score = scores[candidate];
                    if (component == 0) ++score.inputs;
                    else if (component == 1) ++score.contexts;
                    else ++score.outputs;
                });
        };
        for (const auto id : packet.inputs)
            visit(by_input, id, 0);
        if (!packet.direct_context)
            for (const auto id : packet.contexts)
                visit(by_context, id, 1);
        for (const auto id : packet.outputs)
            visit(by_target, id, 2);
        if (work != nullptr) add_work_count(work->candidates_scored, candidate_ids.size());
    };

    std::array<std::size_t, factorized_interaction_max_arity + 1> fallback_cursor{};
    for (std::size_t stable_position = 0;
         stable_position < stable_order.size(); ++stable_position) {
        const auto seed_value = stable_order[stable_position];
        const auto seed = static_cast<std::size_t>(seed_value);
        if (assigned[seed]) continue;
        PacketBuild packet;
        packet.arity = input[seed].arity;
        const bool direct_context_seed =
            metadata[seed].contexts.count > options.max_contexts;
        packet.mechanisms.reserve(direct_context_seed ? 1 : options.max_mechanisms);
        packet.inputs.reserve(direct_context_seed
            ? metadata[seed].inputs.count : options.max_inputs);
        if (!direct_context_seed) packet.contexts.reserve(options.max_contexts);
        packet.outputs.reserve(direct_context_seed ? 1 : options.max_mechanisms);
        add_to_packet(packet, seed_value, input[seed], metadata,
                      unique_inputs, unique_contexts, options);
        assigned[seed] = 1;

        while (!packet.direct_context
               && packet.mechanisms.size() < options.max_mechanisms) {
            if (generation == UINT64_MAX) {
                std::fill(remaining_generation.begin(), remaining_generation.end(), 0);
                generation = 1;
            } else {
                ++generation;
                if (generation == 0) generation = 1;
            }
            score_candidates(packet, input[seed].logical_id, touched, generation);

            std::uint64_t best = UINT64_MAX;
            std::uint64_t best_score = 0;
            for (const auto candidate : touched) {
                if (!packet_can_add(packet, candidate,
                                    input[static_cast<std::size_t>(candidate)],
                                    metadata,
                                    unique_inputs, unique_contexts, options))
                    continue;
                const auto& score = scores[static_cast<std::size_t>(candidate)];
                const std::uint64_t weighted = static_cast<std::uint64_t>(score.inputs) * 4
                    + static_cast<std::uint64_t>(score.contexts) * 2 + score.outputs;
                if (weighted > best_score
                    || (weighted == best_score && weighted != 0
                        && (best == UINT64_MAX
                            || input[static_cast<std::size_t>(candidate)].logical_id
                                < input[static_cast<std::size_t>(best)].logical_id))) {
                    best = candidate;
                    best_score = weighted;
                }
            }

            // Candidate search is bounded by one packet-width prefix in each
            // incidence posting, beginning at this packet's stable seed ID.
            // This is a deterministic clustering heuristic, not an exact
            // global affinity ranking. If no visible candidate fits, take
            // the next stable unassigned mechanism only when it exactly fits.
            // Otherwise the outer stable-ID pass seeds it in the next packet.
            bool selected_by_fallback = false;
            if (best == UINT64_MAX) {
                auto& arity_cursor = fallback_cursor[packet.arity];
                const auto& arity_order = stable_by_arity[packet.arity];
                arity_cursor = std::max(arity_cursor,
                    arity_position[seed] + 1);
                while (arity_cursor < arity_order.size()
                       && assigned[static_cast<std::size_t>(arity_order[arity_cursor])]) {
                    if (work != nullptr && work->fallback_entries_visited != UINT64_MAX)
                        ++work->fallback_entries_visited;
                    ++arity_cursor;
                }
                if (arity_cursor < arity_order.size()) {
                    const auto candidate_value = arity_order[arity_cursor];
                    if (work != nullptr && work->fallback_entries_visited != UINT64_MAX)
                        ++work->fallback_entries_visited;
                    const auto candidate = static_cast<std::size_t>(candidate_value);
                    if (packet_can_add(packet, candidate_value, input[candidate],
                                       metadata, unique_inputs,
                                       unique_contexts, options)) {
                        best = candidate_value;
                        selected_by_fallback = true;
                    }
                }
            }
            if (best == UINT64_MAX) break;
            const auto chosen = static_cast<std::size_t>(best);
            add_to_packet(packet, best, input[chosen], metadata,
                          unique_inputs, unique_contexts, options);
            assigned[chosen] = 1;
            if (selected_by_fallback) ++fallback_cursor[packet.arity];
        }
        packet_builds.push_back(std::move(packet));
    }

    const auto mechanism_count = static_cast<std::uint64_t>(input.size());
    layout.mechanism_packets.resize(input.size());
    std::uint64_t packet_input_count = 0, packet_context_count = 0,
                  packet_output_count = 0;
    std::uint64_t packed_mechanism_count = 0;
    for (const auto& packet : packet_builds) {
        if (packet.mechanisms.empty())
            throw std::logic_error("factorized preparation produced an empty packet");
        if (packet.mechanisms.size() > options.max_mechanisms
            || packet.inputs.size() > options.max_inputs
            || packet.contexts.size() > options.max_contexts
            || packet.outputs.size() > options.max_mechanisms)
            throw std::logic_error("factorized packet exceeded a configured structural bound");
        if (packet.direct_context
            && (packet.mechanisms.size() != 1 || !packet.contexts.empty()))
            throw std::logic_error("direct-context packets must be singleton packets without a dictionary");
        const auto next_input_count = static_cast<std::uint64_t>(packet.inputs.size());
        const auto next_context_count = static_cast<std::uint64_t>(packet.contexts.size());
        const auto next_output_count = static_cast<std::uint64_t>(packet.outputs.size());
        if (packet_input_count > UINT64_MAX - next_input_count
            || packet_context_count > UINT64_MAX - next_context_count
            || packet_output_count > UINT64_MAX - next_output_count
            || packed_mechanism_count > UINT64_MAX - packet.mechanisms.size())
            throw std::overflow_error("factorized packet dictionary extent exceeds uint64");
        packet_input_count += next_input_count;
        packet_context_count += next_context_count;
        packet_output_count += next_output_count;
        packed_mechanism_count += static_cast<std::uint64_t>(packet.mechanisms.size());
    }
    if (packed_mechanism_count != mechanism_count)
        throw std::logic_error("factorized packetization did not preserve every mechanism");
    if (packed_mechanism_count > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max() / 4))
        throw std::overflow_error("factorized role-reference array exceeds host size");

    layout.packets.reserve(packet_builds.size());
    layout.packet_mechanisms.reserve(static_cast<std::size_t>(mechanism_count));
    layout.packet_inputs.reserve(checked_size(packet_input_count,
        "factorized packet input dictionary exceeds host size"));
    layout.packet_contexts.reserve(checked_size(packet_context_count,
        "factorized packet context dictionary exceeds host size"));
    layout.packet_outputs.reserve(checked_size(packet_output_count,
        "factorized packet output dictionary exceeds host size"));
    layout.role_refs.assign(static_cast<std::size_t>(mechanism_count) * 4,
                            invalid_local_ref);
    layout.context_refs.assign(layout.terms.size(), invalid_local_ref);

    std::vector<PostingPair> local_input_pairs, local_output_pairs, local_context_pairs;
    std::vector<PostingPair> input_packet_pairs, context_packet_pairs, output_packet_pairs;
    std::vector<PostingPair> alpha_pairs, bias_pairs, gamma_pairs, global_context_pairs;
    std::size_t input_incidence_count = 0;
    for (const auto& mechanism : input) {
        if (mechanism.arity > std::numeric_limits<std::size_t>::max() - input_incidence_count)
            throw std::overflow_error("factorized input posting count exceeds host size");
        input_incidence_count += mechanism.arity;
    }
    local_input_pairs.reserve(input_incidence_count);
    local_output_pairs.reserve(input.size());
    local_context_pairs.reserve(layout.terms.size());
    alpha_pairs.reserve(input.size());
    bias_pairs.reserve(input.size());
    gamma_pairs.reserve(layout.terms.size());
    global_context_pairs.reserve(layout.terms.size());

    std::uint64_t packed_begin = 0, input_begin = 0, context_begin = 0,
                  output_begin = 0;
    for (std::size_t packet_index = 0; packet_index < packet_builds.size(); ++packet_index) {
        const auto& source_packet = packet_builds[packet_index];
        FactorizedInteractionPacket packet{};
        packet.mechanism_begin = packed_begin;
        packet.input_begin = input_begin;
        packet.context_begin = context_begin;
        packet.output_begin = output_begin;
        packet.mechanism_count = static_cast<std::uint32_t>(source_packet.mechanisms.size());
        packet.input_count = static_cast<std::uint32_t>(source_packet.inputs.size());
        packet.context_count = static_cast<std::uint32_t>(source_packet.contexts.size());
        packet.output_count = static_cast<std::uint32_t>(source_packet.outputs.size());
        packet.direct_context = source_packet.direct_context ? 1u : 0u;
        layout.packets.push_back(packet);

        layout.packet_inputs.insert(layout.packet_inputs.end(), source_packet.inputs.begin(),
                                    source_packet.inputs.end());
        layout.packet_contexts.insert(layout.packet_contexts.end(), source_packet.contexts.begin(),
                                      source_packet.contexts.end());
        layout.packet_outputs.insert(layout.packet_outputs.end(), source_packet.outputs.begin(),
                                     source_packet.outputs.end());

        for (std::size_t local = 0; local < source_packet.inputs.size(); ++local) {
            const auto global = source_packet.inputs[local];
            input_packet_pairs.push_back({global, input_begin + local});
        }
        if (!source_packet.direct_context) {
            for (std::size_t local = 0; local < source_packet.contexts.size(); ++local) {
                const auto global = source_packet.contexts[local];
                context_packet_pairs.push_back({global, context_begin + local});
            }
        }
        for (std::size_t local = 0; local < source_packet.outputs.size(); ++local) {
            const auto global = source_packet.outputs[local];
            output_packet_pairs.push_back({global, output_begin + local});
        }

        for (std::size_t slot = 0; slot < source_packet.mechanisms.size(); ++slot) {
            const auto canonical = source_packet.mechanisms[slot];
            const auto canonical_index = static_cast<std::size_t>(canonical);
            const auto& mechanism = input[canonical_index];
            const auto packed = packed_begin + slot;
            layout.packet_mechanisms.push_back(canonical);
            layout.mechanism_packets[canonical_index] = static_cast<std::uint64_t>(packet_index);
            alpha_pairs.push_back({mechanism.alpha, canonical});
            bias_pairs.push_back({mechanism.bias, canonical});

            const auto role_base = static_cast<std::size_t>(4 * packed_begin);
            for (std::uint32_t role = 0; role < mechanism.arity; ++role) {
                const auto ref = find_ref(source_packet.inputs, mechanism.inputs[role],
                                          "factorized participant missing from packet dictionary");
                layout.role_refs[role_base + static_cast<std::size_t>(role)
                                      * source_packet.mechanisms.size() + slot] =
                    static_cast<std::uint8_t>(ref);
                local_input_pairs.push_back({input_begin + ref, 4 * packed + role});
            }
            const auto output_ref = find_ref(source_packet.outputs, mechanism.target,
                                             "factorized output missing from packet dictionary");
            local_output_pairs.push_back({output_begin + output_ref, packed});

            const auto& record = layout.mechanisms[canonical_index];
            for (std::uint64_t term_offset = 0; term_offset < record.term_count; ++term_offset) {
                const auto term_ordinal = record.term_begin + term_offset;
                const auto& term = layout.terms[static_cast<std::size_t>(term_ordinal)];
                gamma_pairs.push_back({term.gamma, term_ordinal});
                global_context_pairs.push_back({term.context, term_ordinal});
                if (!source_packet.direct_context) {
                    const auto context_ref = find_ref(source_packet.contexts, term.context,
                        "factorized context missing from packet dictionary");
                    layout.context_refs[static_cast<std::size_t>(term_ordinal)] =
                        static_cast<std::uint8_t>(context_ref);
                    local_context_pairs.push_back({context_begin + context_ref, term_ordinal});
                }
            }
        }
        packed_begin += packet.mechanism_count;
        input_begin += packet.input_count;
        context_begin += packet.context_count;
        output_begin += packet.output_count;
    }

    layout.local_input_roles = make_postings(packet_input_count, std::move(local_input_pairs));
    layout.local_output_mechanisms = make_postings(packet_output_count,
                                                   std::move(local_output_pairs));
    layout.local_context_terms = make_postings(packet_context_count,
                                               std::move(local_context_pairs));
    layout.input_packets = make_postings(extents.inputs, std::move(input_packet_pairs));
    layout.context_packets = make_postings(extents.contexts, std::move(context_packet_pairs));
    layout.output_packets = make_postings(extents.outputs, std::move(output_packet_pairs));
    layout.alpha_mechanisms = make_postings(extents.alpha_rows, std::move(alpha_pairs));
    layout.bias_mechanisms = make_postings(extents.biases, std::move(bias_pairs));
    layout.gamma_terms = make_postings(extents.gammas, std::move(gamma_pairs));
    layout.context_terms = make_postings(extents.contexts, std::move(global_context_pairs));
    return layout;
}

bool same_options(const FactorizedInteractionOptions& a,
                  const FactorizedInteractionOptions& b) noexcept {
    return a.max_mechanisms == b.max_mechanisms && a.max_inputs == b.max_inputs
        && a.max_contexts == b.max_contexts && a.pattern_id == b.pattern_id
        && a.structure_id == b.structure_id && a.structure_epoch == b.structure_epoch;
}

bool same_extents(const FactorizedInteractionExtents& a,
                  const FactorizedInteractionExtents& b) noexcept {
    return a.inputs == b.inputs && a.contexts == b.contexts
        && a.outputs == b.outputs && a.alpha_rows == b.alpha_rows
        && a.biases == b.biases && a.gammas == b.gammas;
}

bool same_record(const FactorizedInteractionRecord& a,
                 const FactorizedInteractionRecord& b) noexcept {
    if (a.logical_id != b.logical_id || a.target != b.target || a.alpha != b.alpha
        || a.bias != b.bias || a.term_begin != b.term_begin
        || a.term_count != b.term_count || a.arity != b.arity)
        return false;
    for (std::uint32_t role = 0; role < factorized_interaction_max_arity; ++role)
        if (a.inputs[role] != b.inputs[role]) return false;
    return true;
}

bool same_term(const FactorizedInteractionTermRecord& a,
               const FactorizedInteractionTermRecord& b) noexcept {
    return a.context == b.context && a.gamma == b.gamma
        && a.mechanism == b.mechanism;
}

bool same_packet(const FactorizedInteractionPacket& a,
                 const FactorizedInteractionPacket& b) noexcept {
    return a.mechanism_begin == b.mechanism_begin
        && a.input_begin == b.input_begin && a.context_begin == b.context_begin
        && a.output_begin == b.output_begin && a.mechanism_count == b.mechanism_count
        && a.input_count == b.input_count && a.context_count == b.context_count
        && a.output_count == b.output_count && a.direct_context == b.direct_context
        && a.reserved == b.reserved;
}

bool same_postings(const FactorizedInteractionPostings& a,
                   const FactorizedInteractionPostings& b) noexcept {
    return a.offsets == b.offsets && a.entries == b.entries;
}

bool same_layout(const FactorizedInteractionLayout& a,
                 const FactorizedInteractionLayout& b) noexcept {
    if (!same_options(a.options, b.options) || !same_extents(a.extents, b.extents)
        || a.mechanisms.size() != b.mechanisms.size() || a.terms.size() != b.terms.size()
        || a.packets.size() != b.packets.size()
        || a.packet_mechanisms != b.packet_mechanisms
        || a.mechanism_packets != b.mechanism_packets
        || a.packet_inputs != b.packet_inputs || a.packet_contexts != b.packet_contexts
        || a.packet_outputs != b.packet_outputs || a.role_refs != b.role_refs
        || a.context_refs != b.context_refs
        || !same_postings(a.local_input_roles, b.local_input_roles)
        || !same_postings(a.local_output_mechanisms, b.local_output_mechanisms)
        || !same_postings(a.local_context_terms, b.local_context_terms)
        || !same_postings(a.input_packets, b.input_packets)
        || !same_postings(a.context_packets, b.context_packets)
        || !same_postings(a.output_packets, b.output_packets)
        || !same_postings(a.alpha_mechanisms, b.alpha_mechanisms)
        || !same_postings(a.bias_mechanisms, b.bias_mechanisms)
        || !same_postings(a.gamma_terms, b.gamma_terms)
        || !same_postings(a.context_terms, b.context_terms))
        return false;
    for (std::size_t i = 0; i < a.mechanisms.size(); ++i)
        if (!same_record(a.mechanisms[i], b.mechanisms[i])) return false;
    for (std::size_t i = 0; i < a.terms.size(); ++i)
        if (!same_term(a.terms[i], b.terms[i])) return false;
    for (std::size_t i = 0; i < a.packets.size(); ++i)
        if (!same_packet(a.packets[i], b.packets[i])) return false;
    return true;
}

std::vector<FactorizedInteractionMechanism> reconstruct_input(
    const FactorizedInteractionLayout& layout) {
    std::vector<FactorizedInteractionMechanism> result;
    result.reserve(layout.mechanisms.size());
    for (std::size_t m = 0; m < layout.mechanisms.size(); ++m) {
        const auto& record = layout.mechanisms[m];
        if (record.arity < 2 || record.arity > factorized_interaction_max_arity)
            throw std::invalid_argument("factorized record has invalid arity");
        if (record.term_begin > layout.terms.size()
            || record.term_count > layout.terms.size() - record.term_begin)
            throw std::invalid_argument("factorized record has invalid canonical term range");
        FactorizedInteractionMechanism mechanism;
        mechanism.logical_id = record.logical_id;
        mechanism.target = record.target;
        mechanism.alpha = record.alpha;
        mechanism.bias = record.bias;
        mechanism.arity = record.arity;
        for (std::uint32_t role = 0; role < record.arity; ++role)
            mechanism.inputs[role] = record.inputs[role];
        for (std::uint64_t t = 0; t < record.term_count; ++t) {
            const auto& term = layout.terms[static_cast<std::size_t>(record.term_begin + t)];
            mechanism.terms.push_back({term.context, term.gamma});
        }
        result.push_back(std::move(mechanism));
    }
    return result;
}

} // namespace

FactorizedInteractionView FactorizedInteractionLayout::view() const noexcept {
    const auto ptr = [](const auto& vector) { return vector.empty() ? nullptr : vector.data(); };
    std::uint32_t max_inputs = 0, max_contexts = 0,
                  max_mechanisms = 0, max_outputs = 0;
    for (const auto& packet : packets) {
        max_inputs = std::max(max_inputs, packet.input_count);
        max_contexts = std::max(max_contexts, packet.context_count);
        max_mechanisms = std::max(max_mechanisms, packet.mechanism_count);
        max_outputs = std::max(max_outputs, packet.output_count);
    }
    const auto max_posting_length = [](const FactorizedInteractionPostings& postings) {
        std::uint64_t result = 0;
        for (std::size_t row = 0; row + 1 < postings.offsets.size(); ++row) {
            const auto begin = postings.offsets[row];
            const auto end = postings.offsets[row + 1];
            if (begin > end || end > postings.entries.size()) return UINT64_MAX;
            result = std::max(result, end - begin);
        }
        return result;
    };
    const auto max_global_local_contributions = [](
        const FactorizedInteractionPostings& global,
        const FactorizedInteractionPostings& local) {
        std::uint64_t result = 0;
        for (std::size_t row = 0; row + 1 < global.offsets.size(); ++row) {
            const auto begin = global.offsets[row];
            const auto end = global.offsets[row + 1];
            if (begin > end || end > global.entries.size()) return UINT64_MAX;
            std::uint64_t count = 0;
            for (auto entry = begin; entry < end; ++entry) {
                const auto local_row = global.entries[static_cast<std::size_t>(entry)];
                if (local.offsets.empty() || local_row >= local.offsets.size() - 1)
                    return UINT64_MAX;
                const auto local_begin = local.offsets[static_cast<std::size_t>(local_row)];
                const auto local_end = local.offsets[static_cast<std::size_t>(local_row + 1)];
                if (local_begin > local_end || local_end > local.entries.size()
                    || local_end - local_begin > UINT64_MAX - count)
                    return UINT64_MAX;
                count += local_end - local_begin;
            }
            result = std::max(result, count);
        }
        return result;
    };
    const auto max_output_mechanisms = max_global_local_contributions(
        output_packets, local_output_mechanisms);
    const auto max_input_roles = max_global_local_contributions(
        input_packets, local_input_roles);
    const auto max_context_terms = max_posting_length(context_terms);
    const auto max_alpha_mechanisms = max_posting_length(alpha_mechanisms);
    const auto max_bias_mechanisms = max_posting_length(bias_mechanisms);
    const auto max_gamma_terms = max_posting_length(gamma_terms);
    std::uint64_t max_mechanism_terms = 0;
    for (const auto& mechanism : mechanisms)
        max_mechanism_terms = std::max<std::uint64_t>(
            max_mechanism_terms, mechanism.term_count);
    return {
        factorized_interaction_schema, 0,
        options.pattern_id, options.structure_id, options.structure_epoch,
        extents,
        static_cast<std::uint64_t>(mechanisms.size()),
        static_cast<std::uint64_t>(terms.size()),
        static_cast<std::uint64_t>(packets.size()),
        static_cast<std::uint64_t>(packet_inputs.size()),
        static_cast<std::uint64_t>(packet_contexts.size()),
        static_cast<std::uint64_t>(packet_outputs.size()),
        max_inputs, max_contexts, max_mechanisms, max_outputs,
        max_output_mechanisms, max_input_roles, max_context_terms,
        max_alpha_mechanisms, max_bias_mechanisms, max_gamma_terms,
        max_mechanism_terms,
        ptr(mechanisms), ptr(terms), ptr(packets), ptr(packet_mechanisms),
        ptr(mechanism_packets), ptr(packet_inputs), ptr(packet_contexts),
        ptr(packet_outputs), ptr(role_refs), ptr(context_refs),
        {ptr(local_input_roles.offsets), ptr(local_input_roles.entries)},
        {ptr(local_output_mechanisms.offsets), ptr(local_output_mechanisms.entries)},
        {ptr(local_context_terms.offsets), ptr(local_context_terms.entries)},
        {ptr(input_packets.offsets), ptr(input_packets.entries)},
        {ptr(context_packets.offsets), ptr(context_packets.entries)},
        {ptr(output_packets.offsets), ptr(output_packets.entries)},
        {ptr(alpha_mechanisms.offsets), ptr(alpha_mechanisms.entries)},
        {ptr(bias_mechanisms.offsets), ptr(bias_mechanisms.entries)},
        {ptr(gamma_terms.offsets), ptr(gamma_terms.entries)},
        {ptr(context_terms.offsets), ptr(context_terms.entries)}
    };
}

FactorizedInteractionLayout prepare_factorized_interactions(
    const std::vector<FactorizedInteractionMechanism>& mechanisms,
    FactorizedInteractionExtents extents,
    FactorizedInteractionOptions options) {
    return build_layout(mechanisms, extents, options, nullptr);
}

void validate_factorized_interactions(const FactorizedInteractionLayout& layout) {
    const auto mechanisms = reconstruct_input(layout);
    const auto expected = build_layout(mechanisms, layout.extents, layout.options, nullptr);
    if (!same_layout(layout, expected))
        throw std::invalid_argument("factorized packet layout does not reconstruct its logical records");
}

namespace test_support {

FactorizedInteractionLayout prepare_factorized_interactions_counted(
    const std::vector<FactorizedInteractionMechanism>& mechanisms,
    FactorizedInteractionExtents extents,
    FactorizedInteractionOptions options,
    FactorizedInteractionPreparationWork& work) {
    work = {};
    return build_layout(mechanisms, extents, options, &work);
}

} // namespace test_support

} // namespace cellerator::layout
