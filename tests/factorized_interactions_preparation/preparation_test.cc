#include <Cellerator/layout/factorized_interaction.hh>
#include "../../src/layout/factorized_interaction_test_support.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace layout = cellerator::layout;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

layout::FactorizedInteractionMechanism mechanism(
    std::uint64_t id, std::uint32_t arity,
    std::initializer_list<std::uint64_t> inputs, std::uint64_t target,
    std::uint64_t alpha, std::uint64_t bias,
    std::initializer_list<layout::FactorizedInteractionTerm> terms = {}) {
    layout::FactorizedInteractionMechanism result;
    result.logical_id = id;
    result.arity = arity;
    result.target = target;
    result.alpha = alpha;
    result.bias = bias;
    std::copy(inputs.begin(), inputs.end(), result.inputs.begin());
    result.terms.assign(terms.begin(), terms.end());
    return result;
}

std::vector<layout::FactorizedInteractionMechanism> fixture() {
    return {
        mechanism(30, 3, {0, 1, 0}, 5, 0, 0,
                  {{0, 0}, {0, 1}, {1, 0}}),
        mechanism(10, 3, {0, 1, 2}, 5, 0, 0,
                  {{0, 0}, {1, 1}}),
        mechanism(20, 3, {0, 2, 3}, 6, 1, 1,
                  {{2, 1}}),
        mechanism(40, 2, {8, 9}, 5, 0, 0,
                  {{3, 0}}),
        mechanism(50, 3, {90, 91, 92}, 9, 3, 3,
                  {{40, 0}, {41, 1}, {42, 2}, {43, 3}, {44, 0}}),
        mechanism(60, 3, {0, 0, 2}, 6, 1, 1,
                  {{0, 0}, {2, 1}}),
        mechanism(70, 4, {0, 1, 2, 3}, 6, 1, 1)
    };
}

layout::FactorizedInteractionExtents extents() {
    return {128, 64, 16, 4, 4, 4};
}

layout::FactorizedInteractionOptions options() {
    layout::FactorizedInteractionOptions result;
    result.max_mechanisms = 3;
    result.max_inputs = 6;
    result.max_contexts = 4;
    result.pattern_id = 101;
    result.structure_id = 202;
    result.structure_epoch = 303;
    return result;
}

void validate_round_trip_and_postings() {
    const auto input = fixture();
    const auto prepared = layout::prepare_factorized_interactions(input, extents(), options());
    layout::validate_factorized_interactions(prepared);

    require(prepared.mechanisms.size() == input.size(), "all logical mechanisms retained");
    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto& record = prepared.mechanisms[i];
        const auto& original = input[i];
        require(record.logical_id == original.logical_id,
                "canonical logical mechanism order retained");
        require(record.arity == original.arity && record.target == original.target
                    && record.alpha == original.alpha && record.bias == original.bias,
                "canonical output and parameter bindings retained");
        require(record.term_count == original.terms.size(), "condition term count retained");
        for (std::uint32_t role = 0; role < original.arity; ++role)
            require(record.inputs[role] == original.inputs[role],
                    "ordered participant occurrence retained");
        for (std::uint32_t term = 0; term < record.term_count; ++term) {
            const auto& actual = prepared.terms[record.term_begin + term];
            const auto& expected = original.terms[term];
            require(actual.context == expected.context && actual.gamma == expected.gamma
                        && actual.mechanism == i,
                    "ordered context term and canonical gamma identity retained");
        }
    }

    require(prepared.options.pattern_id == 101 && prepared.options.structure_id == 202
                && prepared.options.structure_epoch == 303,
            "the source structure guard is retained");
    const auto view = prepared.view();
    require(view.schema == layout::factorized_interaction_schema
                && view.pattern_id == 101 && view.structure_id == 202
                && view.structure_epoch == 303,
            "borrowed view exposes schema and structural guards");
    require(view.mechanism_count == input.size() && view.term_count == prepared.terms.size(),
            "borrowed view exposes canonical extents");
    std::uint32_t max_inputs = 0, max_contexts = 0, max_mechanisms = 0, max_outputs = 0;
    for (const auto& packet : prepared.packets) {
        max_inputs = std::max(max_inputs, packet.input_count);
        max_contexts = std::max(max_contexts, packet.context_count);
        max_mechanisms = std::max(max_mechanisms, packet.mechanism_count);
        max_outputs = std::max(max_outputs, packet.output_count);
    }
    require(view.max_packet_inputs == max_inputs
                && view.max_packet_contexts == max_contexts
                && view.max_packet_mechanisms == max_mechanisms
                && view.max_packet_outputs == max_outputs,
            "view exposes exact host-computed packet tile maxima");
    std::map<std::uint64_t, std::uint64_t> output_counts, input_role_counts,
        context_counts, alpha_counts, bias_counts, gamma_counts;
    std::uint64_t max_mechanism_terms = 0;
    for (const auto& record : prepared.mechanisms) {
        max_mechanism_terms = std::max<std::uint64_t>(
            max_mechanism_terms, record.term_count);
        ++output_counts[record.target];
        ++alpha_counts[record.alpha];
        ++bias_counts[record.bias];
        for (std::uint32_t role = 0; role < record.arity; ++role)
            ++input_role_counts[record.inputs[role]];
    }
    for (const auto& term : prepared.terms) {
        ++context_counts[term.context];
        ++gamma_counts[term.gamma];
    }
    const auto map_maximum = [](const auto& counts) {
        std::uint64_t result = 0;
        for (const auto& entry : counts) result = std::max(result, entry.second);
        return result;
    };
    require(view.max_output_mechanisms == map_maximum(output_counts)
                && view.max_input_roles == map_maximum(input_role_counts)
                && view.max_context_terms == map_maximum(context_counts)
                && view.max_alpha_mechanisms == map_maximum(alpha_counts)
                && view.max_bias_mechanisms == map_maximum(bias_counts)
                && view.max_gamma_terms == map_maximum(gamma_counts)
                && view.max_mechanism_terms == max_mechanism_terms,
            "view exposes exact canonical fan-in maxima including repeated/tied incidences");

    // Packet records are homogeneous by arity. The source record order remains
    // canonical while packet order is free to follow deterministic overlap.
    std::vector<std::uint8_t> seen(input.size(), 0);
    std::size_t direct_context_packets = 0;
    for (std::size_t p = 0; p < prepared.packets.size(); ++p) {
        const auto& packet = prepared.packets[p];
        require(packet.mechanism_count > 0 && packet.mechanism_count <= options().max_mechanisms,
                "packet mechanism budget");
        require(packet.input_count <= options().max_inputs
                    && packet.context_count <= options().max_contexts,
                "packet dictionary budgets");
        std::uint32_t arity = 0;
        for (std::uint32_t slot = 0; slot < packet.mechanism_count; ++slot) {
            const auto packed = packet.mechanism_begin + slot;
            const auto canonical = prepared.packet_mechanisms[packed];
            require(canonical < input.size() && !seen[canonical],
                    "each logical mechanism appears in exactly one packet");
            seen[canonical] = 1;
            const auto& original = input[canonical];
            if (slot == 0) arity = original.arity;
            require(original.arity == arity, "packet groups one arity");
            require(prepared.mechanism_packets[canonical] == p,
                    "canonical-to-packet identity map round-trips");
            for (std::uint32_t role = 0; role < original.arity; ++role) {
                const auto ref_index = 4 * packet.mechanism_begin
                    + static_cast<std::uint64_t>(role) * packet.mechanism_count + slot;
                const auto ref = prepared.role_refs[ref_index];
                require(ref < packet.input_count, "8-bit role ref is within dictionary");
                require(prepared.packet_inputs[packet.input_begin + ref] == original.inputs[role],
                        "role ref reconstructs ordered input occurrence");
            }
        }
        for (std::uint32_t role = arity; role < layout::factorized_interaction_max_arity; ++role)
            for (std::uint32_t slot = 0; slot < packet.mechanism_count; ++slot) {
                const auto ref_index = 4 * packet.mechanism_begin
                    + static_cast<std::uint64_t>(role) * packet.mechanism_count + slot;
                require(prepared.role_refs[ref_index] == 0xffu,
                        "unused fixed role columns carry the sentinel");
            }
        if (packet.direct_context) {
            ++direct_context_packets;
            require(packet.mechanism_count == 1 && packet.context_count == 0,
                    "oversized context packet uses exact singleton direct route");
            const auto canonical = prepared.packet_mechanisms[packet.mechanism_begin];
            const auto& record = prepared.mechanisms[canonical];
            for (std::uint64_t i = 0; i < record.term_count; ++i)
                require(prepared.context_refs[record.term_begin + i] == 0xffu,
                        "direct context terms have no invalid local dictionary ref");
        }
    }
    require(std::all_of(seen.begin(), seen.end(), [](auto value) { return value == 1; }),
            "packetization preserves every candidate");
    require(direct_context_packets == 1, "only the oversized context singleton uses direct reads");
    const auto canonical_for_id = [&](std::uint64_t id) {
        const auto found = std::find_if(prepared.mechanisms.begin(), prepared.mechanisms.end(),
            [&](const auto& record) { return record.logical_id == id; });
        require(found != prepared.mechanisms.end(), "fixture logical ID exists");
        return static_cast<std::uint64_t>(found - prepared.mechanisms.begin());
    };
    const auto packet_for_10 = prepared.mechanism_packets[canonical_for_id(10)];
    const auto& shared_packet = prepared.packets[packet_for_10];
    bool grouped_shared_output = false;
    for (std::uint32_t slot = 0; slot < shared_packet.mechanism_count; ++slot) {
        const auto canonical = prepared.packet_mechanisms[shared_packet.mechanism_begin + slot];
        if (prepared.mechanisms[canonical].logical_id == 30) grouped_shared_output = true;
    }
    require(grouped_shared_output,
            "greedy affinity favors repeated inputs, context, and shared output locality");

    // Reconstruct every local participant, output, and staged-context reverse
    // posting directly from the packet records.
    const auto posting_entries = [](const layout::FactorizedInteractionPostings& postings,
                                    std::size_t row) {
        return std::vector<std::uint64_t>(
            postings.entries.begin() + static_cast<std::ptrdiff_t>(postings.offsets[row]),
            postings.entries.begin() + static_cast<std::ptrdiff_t>(postings.offsets[row + 1]));
    };
    for (std::size_t row = 0; row < prepared.packet_inputs.size(); ++row) {
        std::vector<std::uint64_t> expected;
        for (const auto& packet : prepared.packets) {
            if (row < packet.input_begin || row >= packet.input_begin + packet.input_count) continue;
            const auto local = row - packet.input_begin;
            for (std::uint32_t slot = 0; slot < packet.mechanism_count; ++slot) {
                const auto packed = packet.mechanism_begin + slot;
                const auto canonical = prepared.packet_mechanisms[packed];
                const auto& record = prepared.mechanisms[canonical];
                for (std::uint32_t role = 0; role < record.arity; ++role) {
                    const auto ref_index = 4 * packet.mechanism_begin
                        + static_cast<std::uint64_t>(role) * packet.mechanism_count + slot;
                    if (prepared.role_refs[ref_index] == local)
                        expected.push_back(4 * packed + role);
                }
            }
        }
        require(posting_entries(prepared.local_input_roles, row) == expected,
                "participant reverse posting preserves every role occurrence");
    }
    for (std::size_t row = 0; row < prepared.packet_outputs.size(); ++row) {
        std::vector<std::uint64_t> expected;
        for (const auto& packet : prepared.packets) {
            if (row < packet.output_begin || row >= packet.output_begin + packet.output_count) continue;
            const auto target = prepared.packet_outputs[row];
            for (std::uint32_t slot = 0; slot < packet.mechanism_count; ++slot) {
                const auto packed = packet.mechanism_begin + slot;
                const auto canonical = prepared.packet_mechanisms[packed];
                if (prepared.mechanisms[canonical].target == target) expected.push_back(packed);
            }
        }
        require(posting_entries(prepared.local_output_mechanisms, row) == expected,
                "overlapping outputs have deterministic local contribution postings");
    }
    for (std::size_t row = 0; row < prepared.packet_contexts.size(); ++row) {
        std::vector<std::uint64_t> expected;
        for (const auto& packet : prepared.packets) {
            if (packet.direct_context || row < packet.context_begin
                || row >= packet.context_begin + packet.context_count) continue;
            const auto context = prepared.packet_contexts[row];
            for (std::size_t term = 0; term < prepared.terms.size(); ++term) {
                if (prepared.terms[term].context == context
                    && prepared.mechanism_packets[prepared.terms[term].mechanism]
                        == static_cast<std::uint64_t>(&packet - prepared.packets.data()))
                    expected.push_back(term);
            }
        }
        require(posting_entries(prepared.local_context_terms, row) == expected,
                "staged context reverse postings preserve every term occurrence");
    }

    // The global route includes every context term, even those read directly
    // by a direct_context singleton. Tied canonical parameter references are
    // likewise represented once per logical occurrence for deterministic VJPs.
    for (std::uint64_t context = 0; context < extents().contexts; ++context) {
        std::vector<std::uint64_t> expected;
        for (std::size_t term = 0; term < prepared.terms.size(); ++term)
            if (prepared.terms[term].context == context) expected.push_back(term);
        require(posting_entries(prepared.context_terms,
                                static_cast<std::size_t>(context)) == expected,
                "global context reverse route covers staged and direct terms");
    }
    for (std::uint64_t gamma = 0; gamma < extents().gammas; ++gamma) {
        std::vector<std::uint64_t> expected;
        for (std::size_t term = 0; term < prepared.terms.size(); ++term)
            if (prepared.terms[term].gamma == gamma) expected.push_back(term);
        require(posting_entries(prepared.gamma_terms,
                                static_cast<std::size_t>(gamma)) == expected,
                "tied gamma identities retain every term gradient destination");
    }

    for (std::uint64_t input_id = 0; input_id < extents().inputs; ++input_id) {
        std::vector<std::uint64_t> expected;
        for (const auto& packet : prepared.packets)
            for (std::uint32_t local = 0; local < packet.input_count; ++local)
                if (prepared.packet_inputs[packet.input_begin + local] == input_id)
                    expected.push_back(packet.input_begin + local);
        require(posting_entries(prepared.input_packets,
                                static_cast<std::size_t>(input_id)) == expected,
                "cross-packet input postings preserve each packet dictionary owner");
    }
    for (std::uint64_t context = 0; context < extents().contexts; ++context) {
        std::vector<std::uint64_t> expected;
        for (const auto& packet : prepared.packets)
            for (std::uint32_t local = 0; local < packet.context_count; ++local)
                if (prepared.packet_contexts[packet.context_begin + local] == context)
                    expected.push_back(packet.context_begin + local);
        require(posting_entries(prepared.context_packets,
                                static_cast<std::size_t>(context)) == expected,
                "cross-packet context postings omit direct dictionaries but retain staged owners");
    }
    for (std::uint64_t target = 0; target < extents().outputs; ++target) {
        std::vector<std::uint64_t> expected;
        for (const auto& packet : prepared.packets)
            for (std::uint32_t local = 0; local < packet.output_count; ++local)
                if (prepared.packet_outputs[packet.output_begin + local] == target)
                    expected.push_back(packet.output_begin + local);
        require(posting_entries(prepared.output_packets,
                                static_cast<std::size_t>(target)) == expected,
                "cross-packet output postings preserve every target dictionary owner");
    }
    for (std::uint64_t alpha = 0; alpha < extents().alpha_rows; ++alpha) {
        std::vector<std::uint64_t> expected;
        for (std::size_t m = 0; m < prepared.mechanisms.size(); ++m)
            if (prepared.mechanisms[m].alpha == alpha) expected.push_back(m);
        require(posting_entries(prepared.alpha_mechanisms,
                                static_cast<std::size_t>(alpha)) == expected,
                "tied alpha rows preserve each mechanism gradient destination");
    }
    for (std::uint64_t bias = 0; bias < extents().biases; ++bias) {
        std::vector<std::uint64_t> expected;
        for (std::size_t m = 0; m < prepared.mechanisms.size(); ++m)
            if (prepared.mechanisms[m].bias == bias) expected.push_back(m);
        require(posting_entries(prepared.bias_mechanisms,
                                static_cast<std::size_t>(bias)) == expected,
                "tied bias scalars preserve each mechanism gradient destination");
    }

    const auto repeated_context_record = prepared.mechanisms[0];
    require(repeated_context_record.term_count == 3
                && prepared.terms[repeated_context_record.term_begin].context
                    == prepared.terms[repeated_context_record.term_begin + 1].context,
            "equal context identities remain repeated ordered terms");
}

void deterministic_and_empty_cases() {
    const auto input = fixture();
    const auto first = layout::prepare_factorized_interactions(input, extents(), options());
    const auto second = layout::prepare_factorized_interactions(input, extents(), options());
    require(first.packet_mechanisms == second.packet_mechanisms
                && first.mechanism_packets == second.mechanism_packets
                && first.packet_inputs == second.packet_inputs
                && first.packet_contexts == second.packet_contexts
                && first.packet_outputs == second.packet_outputs
                && first.role_refs == second.role_refs
                && first.context_refs == second.context_refs,
            "preparation is deterministic for an identical logical snapshot");

    auto reordered = input;
    std::reverse(reordered.begin(), reordered.end());
    const auto permuted = layout::prepare_factorized_interactions(
        reordered, extents(), options());
    const auto logical_packet_sets = [](const layout::FactorizedInteractionLayout& value) {
        std::vector<std::vector<std::uint64_t>> result;
        for (const auto& packet : value.packets) {
            std::vector<std::uint64_t> ids;
            for (std::uint32_t slot = 0; slot < packet.mechanism_count; ++slot) {
                const auto canonical = value.packet_mechanisms[packet.mechanism_begin + slot];
                ids.push_back(value.mechanisms[canonical].logical_id);
            }
            std::sort(ids.begin(), ids.end());
            result.push_back(std::move(ids));
        }
        std::sort(result.begin(), result.end());
        return result;
    };
    require(logical_packet_sets(first) == logical_packet_sets(permuted),
            "stable IDs make packet membership independent of source record order");
    require(permuted.mechanisms.front().logical_id == reordered.front().logical_id,
            "canonical records still retain the supplied order");

    layout::FactorizedInteractionExtents empty_extents{};
    auto empty = layout::prepare_factorized_interactions({}, empty_extents, options());
    layout::validate_factorized_interactions(empty);
    require(empty.packets.empty() && empty.mechanisms.empty() && empty.terms.empty(),
            "empty mechanism family prepares without inventing work");
    require(empty.local_input_roles.offsets.size() == 1
                && empty.context_terms.offsets.size() == 1,
            "empty views still expose valid zero-row posting offsets");
    const auto empty_view = empty.view();
    require(empty_view.max_output_mechanisms == 0 && empty_view.max_input_roles == 0
                && empty_view.max_context_terms == 0
                && empty_view.max_alpha_mechanisms == 0
                && empty_view.max_bias_mechanisms == 0
                && empty_view.max_gamma_terms == 0
                && empty_view.max_mechanism_terms == 0,
            "empty views report zero contribution-count maxima");
}

void large_gate_term_maximum_is_reported_separately() {
    layout::FactorizedInteractionMechanism value;
    value.logical_id = 800;
    value.inputs[0] = 0;
    value.inputs[1] = 1;
    value.target = 0;
    value.alpha = 0;
    value.bias = 0;
    value.arity = 2;
    for (std::uint64_t term = 0; term < 65; ++term)
        value.terms.push_back({term, term});
    const layout::FactorizedInteractionExtents large_gate_extents{
        2, 65, 1, 1, 1, 65};
    const auto prepared = layout::prepare_factorized_interactions(
        {value}, large_gate_extents, options());
    const auto view = prepared.view();
    require(view.max_mechanism_terms == 65
                && view.max_context_terms == 1 && view.max_gamma_terms == 1,
            "one long gate is visible even when every global context/gamma fan-in is one");
}

void malformed_inputs_reject_before_publication() {
    auto input = fixture();
    auto duplicate = input;
    duplicate[1].logical_id = duplicate[0].logical_id;
    bool rejected = false;
    try { (void)layout::prepare_factorized_interactions(duplicate, extents(), options()); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "duplicate logical IDs are rejected");

    auto bad_arity = input;
    bad_arity[0].arity = 1;
    rejected = false;
    try { (void)layout::prepare_factorized_interactions(bad_arity, extents(), options()); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "unsupported arity is rejected");

    auto bad_input = input;
    bad_input[0].inputs[0] = extents().inputs;
    rejected = false;
    try { (void)layout::prepare_factorized_interactions(bad_input, extents(), options()); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "out-of-range occurrence is rejected");

    auto bad_term = input;
    bad_term[0].terms[0].gamma = extents().gammas;
    rejected = false;
    try { (void)layout::prepare_factorized_interactions(bad_term, extents(), options()); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "out-of-range parameter identity is rejected");

    auto bad_options = options();
    bad_options.max_mechanisms = layout::factorized_interaction_max_packet_mechanisms + 1;
    rejected = false;
    try { (void)layout::prepare_factorized_interactions(input, extents(), bad_options); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "packet limit outside schema is rejected");

    bad_options = options();
    bad_options.structure_epoch = 0;
    rejected = false;
    try { (void)layout::prepare_factorized_interactions(input, extents(), bad_options); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "an invalid structural epoch is rejected");

    auto too_small = options();
    too_small.max_inputs = 1;
    rejected = false;
    try { (void)layout::prepare_factorized_interactions(input, extents(), too_small); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "a mechanism that exceeds an explicitly reduced input budget rejects");

    auto huge_extent = extents();
    huge_extent.inputs = UINT64_MAX;
    bool overflowed = false;
    try { (void)layout::prepare_factorized_interactions(input, huge_extent, options()); }
    catch (const std::overflow_error&) { overflowed = true; }
    require(overflowed, "posting offset extent overflow rejects before allocation");

    auto malformed_layout = layout::prepare_factorized_interactions(input, extents(), options());
    malformed_layout.role_refs[0] = 0xffu;
    rejected = false;
    try { layout::validate_factorized_interactions(malformed_layout); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "round-trip validation rejects a corrupted role map");
}

void deterministic_randomized_structures_reconstruct() {
    std::mt19937_64 random(0x4f524445524544ULL);
    std::vector<layout::FactorizedInteractionMechanism> input;
    input.reserve(120);
    for (std::uint64_t m = 0; m < 120; ++m) {
        const auto arity = static_cast<std::uint32_t>(2 + random() % 3);
        layout::FactorizedInteractionMechanism value;
        value.logical_id = 10000 + m;
        value.arity = arity;
        for (std::uint32_t role = 0; role < arity; ++role)
            value.inputs[role] = random() % 96;
        value.target = random() % 8;
        value.alpha = random() % 16;
        value.bias = random() % 16;
        const auto term_count = static_cast<std::size_t>(random() % 20);
        value.terms.reserve(term_count);
        for (std::size_t term = 0; term < term_count; ++term)
            value.terms.push_back({random() % 32, random() % 16});
        input.push_back(std::move(value));
    }
    const layout::FactorizedInteractionExtents random_extents{96, 32, 8, 16, 16, 16};
    auto random_options = options();
    random_options.max_mechanisms = 8;
    random_options.max_inputs = 12;
    random_options.max_contexts = 10;
    const auto first = layout::prepare_factorized_interactions(
        input, random_extents, random_options);
    layout::validate_factorized_interactions(first);
    const auto second = layout::prepare_factorized_interactions(
        input, random_extents, random_options);
    require(first.packet_mechanisms == second.packet_mechanisms
                && first.packet_inputs == second.packet_inputs
                && first.packet_contexts == second.packet_contexts
                && first.packet_outputs == second.packet_outputs
                && first.role_refs == second.role_refs
                && first.context_refs == second.context_refs,
            "varied repeated/overlapping structures group deterministically");
}

void bounded_candidate_work_scales_for_disjoint_and_shared_incidence() {
    constexpr std::uint64_t count = 4096;
    auto scale_options = options();
    scale_options.max_mechanisms = 32;
    scale_options.max_inputs = 64;
    auto make_mechanisms = [](std::uint64_t n, bool shared) {
        std::vector<layout::FactorizedInteractionMechanism> result;
        result.reserve(static_cast<std::size_t>(n));
        for (std::uint64_t m = 0; m < n; ++m) {
            layout::FactorizedInteractionMechanism value;
            value.logical_id = m + 1;
            value.inputs[0] = shared ? 0 : 2 * m;
            value.inputs[1] = shared ? 1 : 2 * m + 1;
            value.target = shared ? 0 : m;
            value.alpha = 0;
            value.bias = 0;
            value.arity = 2;
            result.push_back(std::move(value));
        }
        return result;
    };
    auto measured_work = [&](const auto& mechanisms, bool shared) {
        const auto n = static_cast<std::uint64_t>(mechanisms.size());
        const layout::FactorizedInteractionExtents scale_extents{
            shared ? 2u : 2 * n, 0, shared ? 1u : n, 1, 1, 0};
        layout::test_support::FactorizedInteractionPreparationWork work;
        const auto prepared = layout::test_support::prepare_factorized_interactions_counted(
            mechanisms, scale_extents, scale_options, work);
        require(prepared.mechanisms.size() == mechanisms.size(),
                "large preparation retains every logical mechanism");
        require(prepared.packets.size() == (mechanisms.size() + 31) / 32,
                "disjoint and common-incidence families fill compatible 32-mechanism packets");
        for (const auto& packet : prepared.packets)
            require(packet.mechanism_count <= 32 && packet.input_count <= 64,
                    "bounded scale packets respect mechanism and input dictionaries");
        return work;
    };

    const auto disjoint = make_mechanisms(count, false);
    const auto disjoint_work = measured_work(disjoint, false);
    require(disjoint_work.fallback_entries_visited <= count,
            "disjoint packet filling advances a monotone stable-ID cursor");
    require(disjoint_work.affinity_entries_visited <= count * 64,
            "disjoint affinity work remains bounded by packet-local incidence visits");
    require(disjoint_work.candidates_scored == 0,
            "disjoint records do not manufacture affinity candidates");

    const auto shared = make_mechanisms(count, true);
    const auto shared_work = measured_work(shared, true);
    require(shared_work.fallback_entries_visited <= count,
            "shared high-degree postings do not cause repeated stable fallback scans");
    require(shared_work.affinity_entries_visited <= count * 100,
            "high-degree participant and target postings use bounded candidate windows");
    require(shared_work.candidates_scored <= count * 32,
            "candidate scoring remains bounded for a common-input/common-target family");
}

} // namespace

int main() {
    try {
        validate_round_trip_and_postings();
        deterministic_and_empty_cases();
        large_gate_term_maximum_is_reported_separately();
        malformed_inputs_reject_before_publication();
        deterministic_randomized_structures_reconstruct();
        bounded_candidate_work_scales_for_disjoint_and_shared_incidence();
        std::cout << "factorized interaction preparation checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "factorized interaction preparation check failed: "
                  << error.what() << '\n';
        return 1;
    }
}
