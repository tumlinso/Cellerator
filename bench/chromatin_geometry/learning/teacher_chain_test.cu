#include "learning/teacher_chain.hh"
#include "reference/teacher_reference.hh"

#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>

namespace {
struct Checks {
    int errors = 0;
    void check(bool ok, const char* label) {
        if (!ok) { std::fprintf(stderr, "teacher chain: %s\n", label); ++errors; }
    }
    bool cuda(cudaError_t e, const char* label) {
        if (e == cudaSuccess) return true;
        std::fprintf(stderr, "teacher chain: %s: %s\n", label, cudaGetErrorString(e));
        ++errors; return false;
    }
};
template<class T, std::size_t N> struct Buffer {
    T* p = nullptr;
    explicit Buffer(Checks& c) { c.cuda(cudaMalloc(reinterpret_cast<void**>(&p), N * sizeof(T)), "allocate"); }
    ~Buffer() { if (p) cudaFree(p); }
    bool upload(const std::array<T, N>& a, cudaStream_t s, Checks& c) {
        return c.cuda(cudaMemcpyAsync(p, a.data(), N * sizeof(T), cudaMemcpyHostToDevice, s), "upload");
    }
    bool download(std::array<T, N>& a, cudaStream_t s, Checks& c) {
        return c.cuda(cudaMemcpyAsync(a.data(), p, N * sizeof(T), cudaMemcpyDeviceToHost, s), "download");
    }
};
bool close(float a, float b) { return std::fabs(a - b) <= 2e-5f * (1.0f + std::fabs(b)); }
} // namespace

int run_teacher_chain_tests(cudaStream_t stream) {
    using namespace cgs::teacher;
    Checks c;
    constexpr std::uint32_t batch = 35, tables = 2, words = 2;
    constexpr std::array<std::uint8_t, tables> truth{0x96, 0xe8};
    std::array<float, 16> logits{}, targets{};
    std::array<std::uint8_t, 16> rows{};
    std::array<std::uint32_t, 16> soft_tables{};
    for (unsigned t = 0; t < tables; ++t) for (unsigned r = 0; r < 8; ++r) {
        rows[t * 8 + r] = r;
        soft_tables[t * 8 + r] = t;
        targets[t * 8 + r] = (truth[t] >> r) & 1u;
    }
    std::array<Node, 5> nodes{};
    for (unsigned i = 0; i < 3; ++i) { nodes[i].op = Op::predicate; nodes[i].x = i; }
    for (unsigned i = 0; i < 2; ++i) {
        nodes[i + 3].op = Op::lut3; nodes[i + 3].x = 0; nodes[i + 3].y = 1;
        nodes[i + 3].z = 2; nodes[i + 3].table = i;
    }
    const std::array<std::uint32_t, 3> levels{0, 3, 5};
    const std::array<std::uint32_t, 2> roots{3, 4};
    std::array<std::uint32_t, 6> predicate{};
    for (unsigned cell = 0; cell < batch; ++cell) {
        unsigned row = cell % 8;
        for (unsigned x = 0; x < 3; ++x)
            if (row & (4u >> x)) predicate[x * words + cell / 32] |= 1u << (cell % 32);
    }
    cgs::GateRevisions before{};
    before.structure_epoch = 1; before.mapping_revision = 2;
    before.accessibility_generation = 3; before.teacher_revision = 4;
    auto after = before; ++after.teacher_revision;
    Program program;
    c.check(prepare_program(nodes.data(), nodes.size(), levels.data(), levels.size(),
                            roots.data(), roots.size(), 3, tables, after, &program), "prepare");
    std::array<FitSpan, tables> fits{{{0, 8, 0.5f}, {8, 8, 0.5f}}};
    ChainSpec spec;
    spec.program = &program; spec.fits = fits.data(); spec.training_sample_count = 16;
    spec.soft_sample_count = 16; spec.batch = batch; spec.table_count = tables;
    spec.epochs = 1; spec.before = before; spec.after = after;
    Buffer<float, 16> d_logits(c), d_targets(c), d_prob(c), d_diff(c);
    Buffer<std::uint8_t, 16> d_rows(c), d_softbits(c);
    Buffer<std::uint32_t, 16> d_soft_tables(c);
    Buffer<RowStatistics, 16> d_stats(c);
    Buffer<std::uint8_t, 2> d_luts(c);
    Buffer<float, 2> d_table_diff(c);
    Buffer<Node, 5> d_nodes(c);
    Buffer<std::uint32_t, 2> d_roots(c);
    Buffer<std::uint32_t, 6> d_pred(c);
    Buffer<std::uint32_t, 10> d_scratch(c);
    Buffer<std::uint32_t, 4> d_gates(c);
    Buffer<std::uint8_t, batch * tables> d_admission(c);
    Buffer<std::uint32_t, 1> d_status(c);
    if (c.errors) return 1;
    d_logits.upload(logits, stream, c); d_targets.upload(targets, stream, c);
    d_rows.upload(rows, stream, c); d_soft_tables.upload(soft_tables, stream, c);
    d_nodes.upload(nodes, stream, c); d_roots.upload(roots, stream, c); d_pred.upload(predicate, stream, c);
    c.cuda(cudaMemsetAsync(d_status.p, 0, sizeof(std::uint32_t), stream), "clear status");
    ChainBuffers b;
    b.capacity = {16, 16, 16, 2, 16, 5, 2, 6, 10, 4, batch * tables};
    b.logits = d_logits.p; b.training_rows = d_rows.p; b.training_targets = d_targets.p;
    b.statistics = d_stats.p; b.luts = d_luts.p; b.table_mean_change = d_table_diff.p;
    b.soft_tables = d_soft_tables.p; b.soft_rows = d_rows.p; b.soft_probability = d_prob.p;
    b.soft_thresholded = d_softbits.p; b.soft_absolute_change = d_diff.p;
    b.nodes = d_nodes.p; b.roots = d_roots.p; b.predicates = d_pred.p;
    b.node_words = d_scratch.p; b.gate_words = d_gates.p; b.admission = d_admission.p;
    b.status = d_status.p;

    ChainReceipt receipt;
    auto bad = spec; bad.after = before;
    c.check(enqueue_chain(bad, b, {}, &receipt, stream) == cudaErrorInvalidValue && !receipt.enqueued,
            "fit without teacher revision rejected");
    auto short_buffer = b; short_buffer.capacity.gate_words = 3;
    c.check(enqueue_chain(spec, short_buffer, {}, &receipt, stream) == cudaErrorInvalidValue && !receipt.enqueued,
            "insufficient gate capacity rejected");

    std::array<cudaEvent_t, 8> event{};
    for (auto& value : event) c.cuda(cudaEventCreate(&value), "create phase event");
    if (c.errors) { for (auto value : event) if (value) cudaEventDestroy(value); return 1; }
    ChainEvents timing{{event[0],event[1]}, {event[2],event[3]},
                       {event[4],event[5]}, {event[6],event[7]}};
    c.cuda(enqueue_chain(spec, b, timing, &receipt, stream), "enqueue");
    c.check(receipt.enqueued && receipt.status == d_status.p &&
            receipt.gate.profile == cgs::GateProfile::teacher &&
            receipt.gate.primal_words == d_gates.p && receipt.gate.derivative_words == d_gates.p &&
            receipt.gate.admitted == d_admission.p && receipt.gate.probabilities == nullptr &&
            cgs::same_revisions(receipt.gate.revisions, after), "gate and revision receipt");

    std::array<std::uint8_t, 2> luts{};
    std::array<float, 16> actual_logits{}, probabilities{}, differences{};
    std::array<std::uint8_t, 16> decisions{};
    std::array<std::uint32_t, 4> gate_words{};
    std::array<std::uint8_t, batch * tables> admission{};
    std::array<std::uint32_t, 1> status{};
    d_logits.download(actual_logits, stream, c); d_luts.download(luts, stream, c);
    d_prob.download(probabilities, stream, c); d_diff.download(differences, stream, c);
    d_softbits.download(decisions, stream, c); d_gates.download(gate_words, stream, c);
    d_admission.download(admission, stream, c); d_status.download(status, stream, c);
    c.cuda(cudaStreamSynchronize(stream), "synchronize");
    c.check(status[0] == 0 && luts == truth, "fit/harden table and status");
    for (unsigned i = 0; i < 16; ++i) {
        const auto initial = cgs::reference::teacher_example_reference(0.0f, targets[i]);
        const float expected_logit = -0.5f * initial.dlogit;
        const auto trained = cgs::reference::teacher_example_reference(expected_logit, targets[i]);
        c.check(close(actual_logits[i], expected_logit), "fitted logit");
        c.check(close(probabilities[i], trained.probability), "soft observation");
        c.check(decisions[i] == ((truth[i / 8] >> (i % 8)) & 1u), "soft decision");
        c.check(close(differences[i], std::fabs(trained.probability - float(decisions[i]))), "soft-hard change");
    }
    for (unsigned cell = 0; cell < batch; ++cell) for (unsigned t = 0; t < tables; ++t) {
        unsigned bit = (truth[t] >> (cell % 8)) & 1u;
        c.check(admission[cell * tables + t] == bit, "chain root admission");
        c.check(((gate_words[t * words + cell / 32] >> (cell % 32)) & 1u) == bit, "chain root bitplane");
    }
    for (unsigned t = 0; t < tables; ++t) c.check((gate_words[t * words + 1] & ~7u) == 0, "chain tail bits");
    for (unsigned phase = 0; phase < 4; ++phase) {
        float elapsed = -1;
        c.cuda(cudaEventElapsedTime(&elapsed, event[phase * 2], event[phase * 2 + 1]), "phase event elapsed");
        c.check(elapsed >= 0, "phase events ordered");
    }
    for (auto value : event) c.cuda(cudaEventDestroy(value), "destroy phase event");
    if (!c.errors) std::puts("teacher fit-to-circuit chain CUDA tests: PASS");
    return c.errors ? 1 : 0;
}
