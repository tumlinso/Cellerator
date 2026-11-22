#pragma once

#include <Cellerator/biology/sequence.hh>
#include <Cellerator/ops/sequence/sequence_event.hh>
#include <Cellerator/tensor/sequence/dna2.hh>

namespace cellerator::ops::sequence {

struct source_stamp {
    structure::domain_id genome;
    uint64_t contig, chunk;
    structure::structure_epoch structure_epoch;
    uint64_t value_generation;
};

// Borrowed host data in original genomic order. The source owner retains the
// arrays and advances the epoch for changes to coordinates or ownership, and
// the value generation for edits to the bases or their validity.
struct source_view {
    tensor::sequence::PackedView sequence;
    biology::SequenceChunk coordinates;
    structure::structure_epoch structure_epoch;
    uint64_t value_generation;
};

constexpr source_stamp stamp(const source_view &source) noexcept {
    const auto &chunk = source.coordinates;
    return {chunk.genome, chunk.contig, chunk.chunk, source.structure_epoch,
        source.value_generation};
}

enum class support_domain : uint8_t { owned = 1, include_halos = 2 };

struct sequence_question {
    source_stamp expected_source;
    biology::SequenceStrand strand;
    support_domain domain;
    uint32_t begin, length;
};

enum class question_status : uint8_t {
    ok, invalid_source, stale_source, invalid_support, invalid_output
};

struct detail_record {
    uint64_t global_position;
    uint32_t logical_position, physical_position;
    uint8_t code;
    biology::SequenceStrand strand;
    bool owned;
};

struct detail_result {
    question_status status;
    source_stamp source;
    sequence_emit_counts counts;
    uint64_t examined, invalid_positions, excluded_halos;
};

// Support is half-open in strand traversal order. Invalid bases are skipped.
// Output follows logical order, and capacity zero only counts. Output storage
// must be disjoint from the input arrays.
detail_result emit_exact_detail(const source_view &source,
    const sequence_question &question, detail_record *output, uint64_t capacity) noexcept;

bool count_valid_source(const source_view &source,
    const sequence_question &question, uint64_t *count) noexcept;

} // namespace cellerator::ops::sequence
