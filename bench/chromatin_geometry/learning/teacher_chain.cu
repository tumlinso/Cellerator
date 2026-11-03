#include "learning/teacher_chain.hh"

#include <cmath>
#include <limits>

namespace cgs::teacher {
namespace {
bool events_valid(const PhaseEvents& p) { return bool(p.begin) == bool(p.end); }
cudaError_t record(cudaEvent_t event, cudaStream_t stream) {
    return event ? cudaEventRecord(event, stream) : cudaSuccess;
}
bool size_fits(std::uint64_t n, std::size_t width) {
    return n <= std::numeric_limits<std::size_t>::max() / width;
}
bool matrix_fits(std::uint64_t a, std::uint64_t b, std::size_t width) {
    const auto limit = std::numeric_limits<std::size_t>::max() / width;
    return !a || b <= limit / a;
}
} // namespace

cudaError_t enqueue_chain(const ChainSpec& s, const ChainBuffers& b,
                          const ChainEvents& e, ChainReceipt* receipt,
                          cudaStream_t stream) noexcept {
    if (!receipt) return cudaErrorInvalidValue;
    *receipt = ChainReceipt{};
    if (!s.program || !s.program->valid || !b.status ||
        s.program->table_count != s.table_count ||
        !same_revisions(s.program->revisions, s.after) ||
        !events_valid(e.training) || !events_valid(e.hardening) ||
        !events_valid(e.soft_observation) || !events_valid(e.circuit) ||
        !size_fits(s.training_sample_count, sizeof(float)) ||
        !size_fits(s.soft_sample_count, sizeof(std::uint32_t)) ||
        !size_fits(std::uint64_t(s.table_count) * 8, sizeof(float)) ||
        !size_fits(std::uint64_t(s.table_count) * 8, sizeof(RowStatistics)) ||
        s.batch > std::numeric_limits<std::uint64_t>::max() - 31)
        return cudaErrorInvalidValue;
    if (s.epochs && !s.table_count) return cudaErrorInvalidValue;

    // Fit changes teacher values; a fresh revision must be supplied even if
    // SGD happens to produce the same bit pattern. No unrelated revision may
    // change under this single chain invocation.
    auto allowed = s.before;
    allowed.teacher_revision = s.after.teacher_revision;
    allowed.parameter_revision = s.after.parameter_revision;
    if (!same_revisions(allowed, s.after)) return cudaErrorInvalidValue;
    if (s.epochs && s.table_count) {
        if (s.after.teacher_revision <= s.before.teacher_revision ||
            s.after.parameter_revision < s.before.parameter_revision || !s.fits ||
            !b.logits || !b.statistics ||
            b.capacity.statistics < std::uint64_t(s.table_count) * 8 ||
            b.capacity.training_samples < s.training_sample_count)
            return cudaErrorInvalidValue;
        for (std::uint32_t t = 0; t < s.table_count; ++t) {
            const auto f = s.fits[t];
            if (f.first > s.training_sample_count || f.count > s.training_sample_count - f.first ||
                !std::isfinite(f.learning_rate) || f.learning_rate <= 0.0f ||
                (f.count && (!b.training_rows || !b.training_targets))) return cudaErrorInvalidValue;
        }
    } else if (!same_revisions(s.before, s.after)) {
        return cudaErrorInvalidValue;
    }
    if (s.table_count && (!b.logits || !b.luts ||
                          b.capacity.logits < std::uint64_t(s.table_count) * 8 ||
                          b.capacity.tables < s.table_count)) return cudaErrorInvalidValue;
    if (s.soft_sample_count && (!s.table_count || !b.soft_tables || !b.soft_rows ||
                                b.capacity.soft_samples < s.soft_sample_count ||
                                (!b.soft_probability && !b.soft_thresholded && !b.soft_absolute_change)))
        return cudaErrorInvalidValue;
    const auto words = (s.batch + 31) / 32;
    if (!matrix_fits(s.program->node_count, words, sizeof(std::uint32_t)) ||
        !matrix_fits(s.program->predicate_count, words, sizeof(std::uint32_t)) ||
        !matrix_fits(s.program->root_count, words, sizeof(std::uint32_t)) ||
        !matrix_fits(s.batch, s.program->root_count, sizeof(std::uint8_t)))
        return cudaErrorInvalidValue;
    if (words && s.program->node_count &&
        (!b.nodes || !b.node_words || (s.program->root_count && (!b.roots || !b.gate_words)) ||
         (s.program->predicate_count && !b.predicates) ||
         b.capacity.nodes < s.program->node_count || b.capacity.roots < s.program->root_count ||
         b.capacity.predicate_words < std::uint64_t(s.program->predicate_count) * words ||
         b.capacity.node_words < std::uint64_t(s.program->node_count) * words ||
         b.capacity.gate_words < std::uint64_t(s.program->root_count) * words ||
         (b.admission && b.capacity.admission < s.batch * s.program->root_count)))
        return cudaErrorInvalidValue;

    // Reuse the evaluator's complete revision/circuit guard before allowing
    // any updates. The empty query makes no CUDA call and touches no device.
    CircuitStamp anticipated{s.after, s.table_count, b.luts, true};
    const auto validity = evaluate(*s.program, anticipated, s.after, nullptr,
                                   nullptr, b.luts, nullptr, 0, nullptr, nullptr,
                                   nullptr, b.status, stream);
    if (validity != cudaSuccess) return validity;

    auto error = record(e.training.begin, stream);
    if (error != cudaSuccess) return error;
    for (std::uint32_t epoch = 0; epoch < s.epochs; ++epoch) {
        for (std::uint32_t table = 0; table < s.table_count; ++table) {
            const auto f = s.fits[table];
            const auto rows = f.count ? b.training_rows + f.first : nullptr;
            const auto targets = f.count ? b.training_targets + f.first : nullptr;
            error = train_step(b.logits + std::uint64_t(table) * 8, rows, targets,
                               f.count, f.learning_rate,
                               b.statistics + std::uint64_t(table) * 8, b.status, stream);
            if (error != cudaSuccess) return error;
        }
    }
    if ((error = record(e.training.end, stream)) != cudaSuccess) return error;
    if ((error = record(e.hardening.begin, stream)) != cudaSuccess) return error;
    CircuitStamp circuit;
    error = harden(b.logits, s.table_count, b.luts, b.table_mean_change,
                   s.after, &circuit, b.status, stream);
    if (error != cudaSuccess) return error;
    if ((error = record(e.hardening.end, stream)) != cudaSuccess) return error;
    if ((error = record(e.soft_observation.begin, stream)) != cudaSuccess) return error;
    error = predict_soft(b.logits, s.table_count, b.soft_tables, b.soft_rows,
                         s.soft_sample_count, b.soft_probability,
                         b.soft_thresholded, b.soft_absolute_change, b.status, stream);
    if (error != cudaSuccess) return error;
    if ((error = record(e.soft_observation.end, stream)) != cudaSuccess) return error;
    if ((error = record(e.circuit.begin, stream)) != cudaSuccess) return error;
    error = evaluate(*s.program, circuit, s.after, b.nodes, b.roots, b.luts,
                     b.predicates, s.batch, b.node_words, b.gate_words,
                     b.admission, b.status, stream);
    if (error != cudaSuccess) return error;
    if ((error = record(e.circuit.end, stream)) != cudaSuccess) return error;

    receipt->before = s.before;
    receipt->after = s.after;
    receipt->circuit = circuit;
    receipt->status = b.status;
    receipt->gate.profile = GateProfile::teacher;
    receipt->gate.probabilities = nullptr;
    receipt->gate.admitted = b.admission;
    receipt->gate.primal_words = b.gate_words;
    receipt->gate.derivative_words = b.gate_words;
    receipt->gate.words_per_condition = words;
    receipt->gate.revisions = s.after;
    receipt->enqueued = true;
    return cudaSuccess;
}

} // namespace cgs::teacher
