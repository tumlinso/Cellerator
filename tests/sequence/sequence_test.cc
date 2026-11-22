#include "sequence_fixture.hh"
#include <Cellerator/ops/sequence/predicate_plan.hh>
#include <Cellerator/ops/sequence/source.hh>

namespace {
using namespace sequence_test;
namespace biology = cellerator::biology;

static_assert(sizeof(seq::sequence_event) == 8u);
static_assert(seq::sequence_predicate_abi_version == 1u);

void test_packing_and_scan() {
    Fixture f{};
    make_fixture(&f);
    require(f.packed.words == f.words && f.packed.validity == f.validity
        && f.packed.base_count == base_count && f.packed.word_count == word_count
        && f.packed.validity_word_count == 2u, "packed words or borrowed extents");

    for (uint32_t i = 0; i < base_count; ++i) {
        const int b = code(f.bases[i]);
        const uint32_t k = i / 32u, shift = i % 32u;
        const bool valid = ((f.valid[k] >> shift) & 1u) != 0u;
        const uint32_t packed = uint32_t(f.words[k] >> (2u*shift)) & 3u;
        const uint32_t planes = ((f.low[k] >> shift) & 1u) | (((f.high[k] >> shift) & 1u) << 1u);
        require(valid == (b >= 0) && packed == planes && (!valid || packed == uint32_t(b)),
                "independent packed-base, validity, or plane comparison");
    }
    require((f.valid[2] >> 6u) == 0u, "nonzero tail validity");
    uint64_t canary[] = {129u, 456u};
    require(layout::pack_ascii(f.bases, base_count, canary, 1u, canary + 1u, 1u, &f.packed)
                == layout::Status::insufficient_capacity
        && canary[0] == 129u && canary[1] == 456u && f.packed.words == f.words,
        "insufficient pack capacity changed contents or binding");
    uint32_t spare[3] = {9u, 8u, 7u};
    require(layout::pack_planes(f.packed, {spare, spare + 1u, spare + 2u, 1u})
                == layout::Status::insufficient_capacity
        && spare[0] == 9u && spare[1] == 8u && spare[2] == 7u, "short plane output changed");

    uint32_t positions[base_count];
    const uint32_t expected = oracle_hits(f, positions);
    require(expected > 2u, "fixture must exercise short emission");
    uint64_t count;
    require(seq::count_motif(f.packed, f.motif, &count) == seq::Status::ok && count == expected,
            "text oracle motif count");
    const uint32_t capacities[] = {0u, 2u, expected, base_count};
    for (uint32_t capacity : capacities) {
        seq::sequence_event events[base_count + 1u]{};
        events[capacity].local_position = UINT32_MAX;
        seq::sequence_emit_counts counts{};
        require(seq::emit_motif(f.packed, f.motif, capacity ? events : nullptr, capacity, &counts)
                    == seq::Status::ok, "emit motif");
        check_counts(counts, expected, capacity);
        require(events[capacity].local_position == UINT32_MAX, "emit wrote beyond capacity");
        for (uint32_t i = 0; i < counts.stored_records; ++i) {
            const uint32_t p = positions[i];
            require(events[i].local_position == p && events[i].predicate_id == 17u
                && events[i].payload == mismatches(f, p), "stable host event differs from text oracle");
        }
    }

    seq::motif32_exact forged = f.motif;
    forged.active_fields = 0u;
    count = 51u;
    require(seq::count_motif(f.packed, forged, &count) != seq::Status::ok && count == 51u,
            "forged motif changed count");
    tensor::PackedView malformed = f.packed;
    --malformed.word_count;
    require(seq::count_motif(malformed, f.motif, &count) != seq::Status::ok && count == 51u,
            "short input changed count");
    require(seq::count_motif({}, f.motif, &count) == seq::Status::ok && count == 0u,
            "zero-size sequence count");

    // Same frame and backing buffers, smaller input and new values.
    f.bases[0] = 'A'; f.bases[1] = 'C'; f.bases[2] = 'G';
    require(layout::pack_ascii(f.bases, 3u, f.words, word_count, f.validity, 2u, &f.packed)
                == layout::Status::ok
        && seq::count_motif(f.packed, f.motif, &count) == seq::Status::ok && count == 1u
        && f.packed.words == f.words, "smaller input in existing buffers");
}

void test_masks() {
    Fixture f{};
    make_fixture(&f);
    uint32_t cytosine[word_count], gc[word_count], cpg[word_count], actual[word_count + 1u]{};
    oracle_masks(f, cytosine, gc, cpg);
    tensor::BitPlanesView bits{f.low, f.high, f.valid, word_count, base_count};
    require(seq::match_exact_at(bits, 30u, f.motif) && seq::match_exact_at(bits, 62u, f.motif)
        && !seq::match_exact_at(bits, 45u, f.motif) && !seq::match_exact_at(bits, 68u, f.motif),
        "borrowed shifted-window match across a word, validity, or tail boundary");
    actual[word_count] = UINT32_MAX;
    require(seq::base_mask(bits, 1u, actual, word_count) == seq::Status::ok, "base mask");
    for (uint32_t k = 0; k < word_count; ++k) require(actual[k] == cytosine[k], "base-mask oracle");
    require(seq::gc_mask(bits, actual, word_count) == seq::Status::ok, "GC mask");
    for (uint32_t k = 0; k < word_count; ++k) require(actual[k] == gc[k], "GC-mask oracle");
    require(seq::cpg_start_mask(bits, actual, word_count) == seq::Status::ok, "CpG mask");
    for (uint32_t k = 0; k < word_count; ++k) require(actual[k] == cpg[k], "CpG seam and tail oracle");
    require(actual[word_count] == UINT32_MAX, "mask wrote beyond capacity");

    require(seq::base_mask(bits, 0u, actual, word_count) == seq::Status::ok
        && !(actual[1] & (1u << (46u % 32u))), "invalid base decoded as A");
    const uint32_t first = actual[0];
    require(seq::base_mask(bits, 4u, actual, word_count) != seq::Status::ok && actual[0] == first,
            "invalid code changed mask");
    require(seq::gc_mask(bits, actual, 1u) != seq::Status::ok && actual[0] == first,
            "short output changed mask");

    // Borrowed buffers may carry unrelated bits beyond their logical extent.
    f.valid[2] |= UINT32_MAX << 6u;
    f.low[2] |= UINT32_MAX << 6u;
    f.high[2] |= UINT32_MAX << 6u;
    require(seq::base_mask(bits, 3u, actual, word_count) == seq::Status::ok && !(actual[2] >> 6u),
            "borrowed tail bits became bases");
    bits.validity = nullptr;
    require(seq::base_mask(bits, 3u, actual, word_count) == seq::Status::ok && !(actual[2] >> 6u),
            "implicit validity exposed tail bits");
}

void test_motif_extents() {
    Fixture f{};
    make_fixture(&f);
    const tensor::BitPlanesView bits{f.low, f.high, f.valid, word_count, base_count};
    const uint8_t lengths[] = {1u, 8u, 16u, 31u, 32u};
    for (uint8_t length : lengths) {
        seq::motif32_exact motif{};
        require(seq::dna2_normalize_motif32_exact(f.words[0], length, 0u, 19u, &motif),
                "normalize fixed-length motif");
        uint64_t expected = 0u;
        for (uint32_t i = 0; i + length <= base_count; ++i) {
            uint32_t j = 0u;
            while (j < length && code(f.bases[i + j]) == code(f.bases[j])) ++j;
            require(seq::match_exact_at(bits, i, motif) == (j == length), "shifted bit-plane text match");
            expected += j == length;
        }
        uint64_t count;
        require(seq::count_motif(f.packed, motif, &count) == seq::Status::ok && count == expected,
                "packed scan text oracle across fixed motif lengths");
    }
}

void test_predicate_contract() {
    seq::sequence_predicate_program program{};
    program.version = seq::sequence_program_version;
    program.node_count = 3u;
    program.output_count = 1u;
    const uint16_t none = seq::sequence_program_no_input;
    const auto mask = seq::predicate_value_kind::mask;
    program.nodes[0] = {seq::predicate_opcode::base_equal, mask, 0u, 0u, none, none, 2u, 0u};
    program.nodes[1] = {seq::predicate_opcode::shift, mask, seq::predicate_shift_right, 0u, 0u, none, 4u, 0u};
    program.nodes[2] = {seq::predicate_opcode::gc_mask, mask, 0u, 0u, none, none, 0u, 0u};
    program.outputs[0] = {1u, 17u, seq::sequence_output_mode::mask, 0u, 0u, 0u};

    seq::prepared_predicate_plan plan{};
    require(seq::prepare_sequence_predicate_program(program, &plan) == seq::predicate_plan_status::ok,
            "prepare bounded predicate");
    require(plan.live_node_count == 2u && plan.original_to_prepared[2] == none
        && plan.lookahead == 4u && plan.lookbehind == 0u
        && plan.family == seq::predicate_lowering_family::fused_plane_mask,
        "incorrect reach, lowering, or dead-node map");
    require(plan.semantic_hash == seq::sequence_predicate_program_hash(program), "predicate hash mismatch");

    auto variant = program;
    variant.nodes[0].reserved = 19u;
    require(seq::sequence_predicate_program_hash(variant) == plan.semantic_hash, "hash includes reserved byte");
    variant.outputs[0].predicate_id = 7u;
    require(seq::sequence_predicate_program_hash(variant) != plan.semantic_hash, "hash ignores predicate identity");
    variant = program; ++variant.version;
    require(seq::verify_sequence_predicate_program(variant) == seq::predicate_plan_status::unsupported_version,
            "unknown version accepted");
    variant = program; variant.nodes[1].input_a = 1u;
    require(seq::verify_sequence_predicate_program(variant) == seq::predicate_plan_status::invalid_input,
            "self reference accepted");
    variant = program; variant.outputs[0].node = 3u;
    require(seq::verify_sequence_predicate_program(variant) == seq::predicate_plan_status::invalid_output,
            "out of range output node accepted");
    variant = program; variant.node_count = seq::sequence_program_max_nodes + 1u;
    auto rejected = plan;
    require(seq::sequence_predicate_program_hash(variant) == 0u
        && seq::prepare_sequence_predicate_program(variant, &rejected) == seq::predicate_plan_status::invalid_count
        && rejected.semantic_hash == plan.semantic_hash, "oversized predicate read or changed a plan");

    program.node_count = 2u;
    program.exact_motif_count = 2u;
    require(seq::dna2_normalize_motif32_exact(0x24u, 3u, 0u, 17u, &program.exact_motifs[0])
        && seq::dna2_normalize_motif32_exact(0u, 32u, 0u, 18u, &program.exact_motifs[1]),
        "normalize live and dead motifs");
    program.nodes[0] = {seq::predicate_opcode::exact_motif, mask, 0u, 0u, none, none, 0u, 0u};
    program.nodes[1] = {seq::predicate_opcode::exact_motif, mask, 0u, 0u, none, none, 1u, 0u};
    program.outputs[0].node = 0u;
    require(seq::prepare_sequence_predicate_program(program, &plan) == seq::predicate_plan_status::ok
        && plan.lookahead == 2u && plan.lookbehind == 0u && plan.live_node_count == 1u,
        "live motif must declare both trailing bases for halo planning");
    program.nodes[0].opcode = seq::predicate_opcode::cpg_start_mask;
    require(seq::prepare_sequence_predicate_program(program, &plan) == seq::predicate_plan_status::ok
        && plan.lookahead == 1u && plan.lookbehind == 0u, "CpG must declare adjacent base for halo planning");
}

void test_allowed_motif() {
    const uint8_t sets[] = {1u, 6u, 8u};
    seq::motif32_allowed motif{};
    require(seq::dna2_make_motif32_allowed(sets, 3u, 0u, 12u, seq::dna2_strand_mode::both,
                seq::dna2_palindrome_policy::report_both, &motif), "normalize allowed motif");
    for (uint32_t x = 0; x < 64u; ++x) {
        const uint32_t a = x & 3u, b = (x >> 2u) & 3u, c = x >> 4u;
        const uint32_t low = (a & 1u) | ((b & 1u) << 1u) | ((c & 1u) << 2u);
        const uint32_t high = (a >> 1u) | ((b >> 1u) << 1u) | ((c >> 1u) << 2u);
        const bool matches = a == 0u && (b == 1u || b == 2u) && c == 3u;
        const uint32_t validity_cases[] = {7u, 5u};
        for (uint32_t validity : validity_cases) {
            seq::dna2_allowed_match_result result{};
            require(seq::dna2_match_allowed_window(low, high, validity, motif, &result),
                    "evaluate allowed window");
            const bool hit = matches && validity == 7u;
            require(result.strand_hits == (hit ? 3u : 0u), "allowed subsets, reverse complement, or validity");
        }
    }
    motif.active_mask = 0u;
    require(!seq::dna2_valid_motif32_allowed(motif), "forged allowed motif accepted");
}

void test_borrowed_detail() {
    Fixture f{};
    make_fixture(&f);
    constexpr uint64_t origin = 1000u;
    const seq::source_view source{f.packed, {{17u, 3u}, 1u, 4u, origin, base_count,
                                             4u, base_count - 2u, 4u, 2u}, {8u}, 5u};
    const biology::SequenceStrand strands[] = {biology::SequenceStrand::forward, biology::SequenceStrand::reverse};
    const seq::support_domain domains[] = {seq::support_domain::owned, seq::support_domain::include_halos};
    for (auto strand : strands) {
        for (auto domain : domains) {
            const seq::sequence_question question{seq::stamp(source), strand, domain, 0u, base_count};
            seq::detail_record detail[base_count + 1u]{};
            const uint32_t capacities[] = {0u, 3u, base_count};
            for (uint32_t capacity : capacities) {
                detail[capacity].global_position = UINT64_MAX;
                const auto result = seq::emit_exact_detail(source, question,
                                                           capacity ? detail : nullptr, capacity);
                require(result.status == seq::question_status::ok && result.examined == base_count
                    && result.source.value_generation == source.value_generation,
                    "source detail status or provenance");
                uint32_t hits = 0u, invalid = 0u, excluded = 0u;
                const bool reverse = strand == biology::SequenceStrand::reverse;
                for (uint32_t i = 0; i < base_count; ++i) {
                    const uint32_t p = reverse ? base_count - i - 1u : i;
                    const int b = code(f.bases[p]);
                    const bool owned = p >= 4u && p < base_count - 2u;
                    if (b < 0) { ++invalid; continue; }
                    if (!owned && domain == seq::support_domain::owned) { ++excluded; continue; }
                    if (hits < capacity) {
                        const auto &d = detail[hits];
                        require(d.global_position == origin + p && d.logical_position == i
                            && d.physical_position == p && d.code == (reverse ? b ^ 3 : b)
                            && d.strand == strand && d.owned == owned,
                            "text oracle source coordinates, strand, or halo ownership");
                    }
                    ++hits;
                }
                check_counts(result.counts, hits, capacity);
                require(result.invalid_positions == invalid && result.excluded_halos == excluded
                    && detail[capacity].global_position == UINT64_MAX, "detail accounting or capacity overwrite");
                uint64_t count = 0u;
                require(seq::count_valid_source(source, question, &count) && count == hits,
                        "exact detail count differs from emitted total");
            }
            auto stale = question;
            ++stale.expected_source.value_generation;
            require(seq::emit_exact_detail(source, stale, detail, base_count).status
                        == seq::question_status::stale_source, "stale source accepted");
            auto overflow = question; overflow.begin = UINT32_MAX;
            require(seq::emit_exact_detail(source, overflow, detail, base_count).status
                        == seq::question_status::invalid_support, "overflowing support accepted");
        }
    }
}
} // namespace

int main() {
    test_packing_and_scan();
    test_masks();
    test_motif_extents();
    test_predicate_contract();
    test_allowed_motif();
    test_borrowed_detail();
    puts("sequence host contracts passed");
    return 0;
}
