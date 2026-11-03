#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cellerator::layout {

inline constexpr std::uint32_t factorized_interaction_schema = 1;
inline constexpr std::uint32_t factorized_interaction_max_arity = 4;
inline constexpr std::uint32_t factorized_interaction_max_packet_mechanisms = 32;
inline constexpr std::uint32_t factorized_interaction_max_packet_inputs = 64;
inline constexpr std::uint32_t factorized_interaction_max_packet_contexts = 64;

// All coordinates below are operation-local numerical occurrence ordinals.
// They do not infer biological identity from a physical position. The caller
// resolves its existing logical identities and pins the supplied structure
// epoch before preparation; changing that binding requires new preparation.
struct FactorizedInteractionTerm {
    std::uint64_t context = 0;
    std::uint64_t gamma = 0;
};

struct FactorizedInteractionMechanism {
    std::uint64_t logical_id = 0;
    std::array<std::uint64_t, factorized_interaction_max_arity> inputs{};
    std::uint64_t target = 0;
    std::uint64_t alpha = 0;
    std::uint64_t bias = 0;
    std::uint32_t arity = 2;
    std::vector<FactorizedInteractionTerm> terms;
};

struct FactorizedInteractionExtents {
    std::uint64_t inputs = 0;
    std::uint64_t contexts = 0;
    std::uint64_t outputs = 0;
    std::uint64_t alpha_rows = 0;
    std::uint64_t biases = 0;
    std::uint64_t gammas = 0;
};

struct FactorizedInteractionOptions {
    std::uint32_t max_mechanisms = factorized_interaction_max_packet_mechanisms;
    std::uint32_t max_inputs = factorized_interaction_max_packet_inputs;
    std::uint32_t max_contexts = factorized_interaction_max_packet_contexts;
    std::uint64_t pattern_id = 1;
    std::uint64_t structure_id = 1;
    std::uint64_t structure_epoch = 1;
};

// Canonical mechanism records preserve the input record order. Ordered and
// repeated input roles and condition terms are retained, including tied
// alpha-row, bias-scalar and gamma-scalar parameter references.
struct FactorizedInteractionRecord {
    std::uint64_t logical_id = 0;
    std::uint64_t inputs[factorized_interaction_max_arity]{};
    std::uint64_t target = 0, alpha = 0, bias = 0;
    std::uint64_t term_begin = 0;
    std::uint32_t term_count = 0, arity = 0;
};

struct FactorizedInteractionTermRecord {
    std::uint64_t context = 0, gamma = 0, mechanism = 0;
};

// A packet may contain different arities; the initial evaluators specialize
// the bounded law, not arbitrary function pointers. Ref arrays use fixed
// four-role columns: role_refs[4*mechanism_begin + role*mechanism_count + slot].
// Context refs are indexed by canonical term ordinal. For a singleton with
// more distinct context inputs than max_contexts, direct_context is one and
// the gate reads canonical terms directly (no oversized shared dictionary).
struct FactorizedInteractionPacket {
    std::uint64_t mechanism_begin = 0;
    std::uint64_t input_begin = 0, context_begin = 0, output_begin = 0;
    std::uint32_t mechanism_count = 0, input_count = 0;
    std::uint32_t context_count = 0, output_count = 0;
    std::uint32_t direct_context = 0, reserved = 0;
};

// offsets[n+1] delimit entries. Each index is a flat ordinal, never a byte
// pointer or a logical biological ID. The same POD can describe host records
// for inspection or uploaded immutable device records for execution.
struct FactorizedInteractionPostingsView {
    const std::uint64_t* offsets = nullptr;
    const std::uint64_t* entries = nullptr;
};

struct FactorizedInteractionView {
    std::uint32_t schema = factorized_interaction_schema, reserved = 0;
    std::uint64_t pattern_id = 0, structure_id = 0, structure_epoch = 0;
    FactorizedInteractionExtents extents{};
    std::uint64_t mechanism_count = 0, term_count = 0, packet_count = 0;
    std::uint64_t packet_input_count = 0, packet_context_count = 0;
    std::uint64_t packet_output_count = 0;
    // Host-computed maxima size the per-CTA live tiles, without device reads.
    std::uint32_t max_packet_inputs = 0, max_packet_contexts = 0;
    std::uint32_t max_packet_mechanisms = 0, max_packet_outputs = 0;
    // Exact global canonical contribution-count maxima. They select the
    // bounded atomic route or the shared accurate owner-reduction fallback.
    std::uint64_t max_output_mechanisms = 0, max_input_roles = 0;
    std::uint64_t max_context_terms = 0, max_alpha_mechanisms = 0;
    std::uint64_t max_bias_mechanisms = 0, max_gamma_terms = 0;
    std::uint64_t max_mechanism_terms = 0;
    const FactorizedInteractionRecord* mechanisms = nullptr;
    const FactorizedInteractionTermRecord* terms = nullptr;
    const FactorizedInteractionPacket* packets = nullptr;
    const std::uint64_t* packet_mechanisms = nullptr; // canonical mechanism ordinals
    const std::uint64_t* mechanism_packets = nullptr;
    const std::uint64_t* packet_inputs = nullptr;
    const std::uint64_t* packet_contexts = nullptr;
    const std::uint64_t* packet_outputs = nullptr;
    const std::uint8_t* role_refs = nullptr;
    const std::uint8_t* context_refs = nullptr;
    // Local input postings: entry = 4*packed_mechanism_ordinal + ordered role.
    FactorizedInteractionPostingsView local_input_roles{};
    // Local output postings: entry = packed mechanism ordinal.
    FactorizedInteractionPostingsView local_output_mechanisms{};
    // Local context postings: entry = canonical term ordinal.
    FactorizedInteractionPostingsView local_context_terms{};
    // Cross-packet postings map global occurrence -> packet dictionary ordinal.
    FactorizedInteractionPostingsView input_packets{}, context_packets{}, output_packets{};
    // Parameter postings retain every tied occurrence; entries are canonical
    // mechanism ordinals for alpha/bias and canonical term ordinals for gamma.
    FactorizedInteractionPostingsView alpha_mechanisms{}, bias_mechanisms{}, gamma_terms{};
    // Global context -> canonical condition terms, including direct_context
    // singleton packets that have no shared context dictionary entry.
    FactorizedInteractionPostingsView context_terms{};
};

struct FactorizedInteractionPostings {
    std::vector<std::uint64_t> offsets, entries;
};

struct FactorizedInteractionLayout {
    FactorizedInteractionOptions options{};
    FactorizedInteractionExtents extents{};
    std::vector<FactorizedInteractionRecord> mechanisms;
    std::vector<FactorizedInteractionTermRecord> terms;
    std::vector<FactorizedInteractionPacket> packets;
    std::vector<std::uint64_t> packet_mechanisms, mechanism_packets;
    std::vector<std::uint64_t> packet_inputs, packet_contexts, packet_outputs;
    std::vector<std::uint8_t> role_refs, context_refs;
    FactorizedInteractionPostings local_input_roles, local_output_mechanisms, local_context_terms;
    FactorizedInteractionPostings input_packets, context_packets, output_packets;
    FactorizedInteractionPostings alpha_mechanisms, bias_mechanisms, gamma_terms;
    FactorizedInteractionPostings context_terms;

    FactorizedInteractionView view() const noexcept;
};

// Throws invalid_argument/overflow_error before returning malformed structure.
// Empty mechanisms and zero numerical occurrence extents are supported when
// no mechanism references them. Each mechanism has arity 2, 3 or 4. Empty
// condition lists produce sigmoid(bias). Stable logical IDs must be nonzero
// and unique, matching the existing ExplicitPattern identity contract.
FactorizedInteractionLayout prepare_factorized_interactions(
    const std::vector<FactorizedInteractionMechanism>& mechanisms,
    FactorizedInteractionExtents extents,
    FactorizedInteractionOptions options = {});

// Reconstructs all roles, terms, postings, IDs and structural maps. This host
// check does not dereference an uploaded device view or validate numerics.
void validate_factorized_interactions(const FactorizedInteractionLayout& layout);

} // namespace cellerator::layout
