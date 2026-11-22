#include <Cellerator/ops/sequence/source.hh>

namespace cellerator::ops::sequence {
namespace packed = tensor::sequence;
using biology::SequenceStrand;

detail_result emit_exact_detail(const source_view &source,
    const sequence_question &question, detail_record *output, uint64_t capacity) noexcept {
    const auto &chunk = source.coordinates;
    const auto current = stamp(source);
    detail_result result{question_status::ok, current, {}, 0u, 0u, 0u};
    if (!biology::valid_sequence_chunk(chunk) || !packed::valid_view(source.sequence)
        || source.sequence.base_count != chunk.base_count) {
        result.status = question_status::invalid_source;
        return result;
    }

    const auto &expected = question.expected_source;
    if (!structure::same_identity(expected.genome, current.genome)
        || expected.contig != current.contig || expected.chunk != current.chunk
        || !structure::same_epoch(expected.structure_epoch, current.structure_epoch)
        || expected.value_generation != current.value_generation) {
        result.status = question_status::stale_source;
        return result;
    }

    const bool reverse = question.strand == SequenceStrand::reverse;
    if ((!reverse && question.strand != SequenceStrand::forward)
        || (question.domain != support_domain::owned
            && question.domain != support_domain::include_halos)
        || question.begin > chunk.base_count
        || question.length > chunk.base_count - question.begin) {
        result.status = question_status::invalid_support;
        return result;
    }
    if (capacity && !output) {
        result.status = question_status::invalid_output;
        return result;
    }

    result.examined = question.length;
    for (uint32_t i = question.begin; i < question.begin + question.length; ++i) {
        const uint32_t p = reverse ? chunk.base_count - 1u - i : i;
        if (!packed::valid_window(source.sequence, p, 1u)) {
            ++result.invalid_positions;
            continue;
        }
        const bool owned = p >= chunk.owned_begin && p < chunk.owned_end;
        if (!owned && question.domain == support_domain::owned) {
            ++result.excluded_halos;
            continue;
        }
        ++result.counts.total_matches;
        if (result.counts.stored_records < capacity) {
            const uint8_t code = packed::base_code(source.sequence, p) ^ (reverse ? 3u : 0u);
            output[result.counts.stored_records++] = {
                chunk.global_base_begin + p, i, p, code, question.strand, owned};
        }
    }
    result.counts.required_capacity = result.counts.total_matches;
    result.counts.dropped_records = result.counts.total_matches - result.counts.stored_records;
    return result;
}

bool count_valid_source(const source_view &source,
    const sequence_question &question, uint64_t *count) noexcept {
    if (!count) return false;
    const auto result = emit_exact_detail(source, question, nullptr, 0u);
    if (result.status != question_status::ok) return false;
    *count = result.counts.total_matches;
    return true;
}

} // namespace cellerator::ops::sequence
