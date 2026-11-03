#include "learning/teacher.hh"
#include "reference/teacher_reference.hh"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {
using namespace cgs::teacher;

struct Checks {
    int failures = 0;
    void check(bool ok, const char* what) {
        if (!ok) { std::fprintf(stderr, "teacher test: %s\n", what); ++failures; }
    }
    bool cuda(cudaError_t result, const char* what) {
        if (result == cudaSuccess) return true;
        std::fprintf(stderr, "teacher test: %s: %s\n", what, cudaGetErrorString(result));
        ++failures;
        return false;
    }
};

template<class T> struct Device {
    T* p = nullptr;
    explicit Device(std::size_t n, Checks& c) {
        if (n) c.cuda(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)), "cudaMalloc");
    }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

template<class T>
bool upload(T* to, const T* from, std::size_t n, cudaStream_t stream, Checks& c) {
    return !n || c.cuda(cudaMemcpyAsync(to, from, n * sizeof(T), cudaMemcpyHostToDevice, stream), "upload");
}
template<class T>
bool download(T* to, const T* from, std::size_t n, cudaStream_t stream, Checks& c) {
    return !n || c.cuda(cudaMemcpyAsync(to, from, n * sizeof(T), cudaMemcpyDeviceToHost, stream), "download");
}
bool close(float a, float b) { return std::fabs(a - b) <= 2e-5f * (1.0f + std::fabs(b)); }

bool train_checks(cudaStream_t stream, Checks& c) {
    std::array<float, 8> original{-3.0f, -1.0f, 0.0f, 0.5f, 1.25f, 2.0f, 4.0f, -0.25f};
    // Unequal sample counts and one unobserved row expose incorrect global or
    // missing-row normalization. Row 2 has an exact threshold-equality logit.
    const std::vector<std::uint8_t> rows{0,0,0,1,1,2,3,3,3,3,4,5,6};
    const std::vector<float> labels{0,1,0.25f,1,0.5f,0,1,0,0.25f,1,0,1,0.75f};
    const float rate = 0.3f;
    std::array<float, 8> grad{}, loss{};
    std::array<unsigned, 8> count{};
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto ref = cgs::reference::teacher_example_reference(original[rows[i]], labels[i]);
        grad[rows[i]] += ref.dlogit;
        loss[rows[i]] += ref.bce;
        ++count[rows[i]];
    }
    for (unsigned i = 0; i < 8; ++i) if (count[i]) { grad[i] /= count[i]; loss[i] /= count[i]; }

    Device<float> logits(8, c), dev_labels(labels.size(), c);
    Device<std::uint8_t> dev_rows(rows.size(), c);
    Device<RowStatistics> stats(8, c);
    Device<std::uint32_t> status(1, c);
    if (!logits.p || !dev_labels.p || !dev_rows.p || !stats.p || !status.p) return false;
    upload(logits.p, original.data(), 8, stream, c);
    upload(dev_rows.p, rows.data(), rows.size(), stream, c);
    upload(dev_labels.p, labels.data(), labels.size(), stream, c);
    c.cuda(cudaMemsetAsync(status.p, 0, sizeof(std::uint32_t), stream), "clear train status");
    c.cuda(train_step(logits.p, dev_rows.p, dev_labels.p, rows.size(), rate, stats.p, status.p, stream), "train step");
    std::array<float, 8> trained{};
    std::array<RowStatistics, 8> got{};
    std::uint32_t flags = 0;
    download(trained.data(), logits.p, 8, stream, c);
    download(got.data(), stats.p, 8, stream, c);
    download(&flags, status.p, 1, stream, c);
    if (!c.cuda(cudaStreamSynchronize(stream), "train synchronize")) return false;
    c.check(flags == 0, "training status");
    for (unsigned i = 0; i < 8; ++i) {
        c.check(got[i].count == count[i], "training row count");
        c.check(close(got[i].gradient, grad[i]), "training analytic gradient");
        c.check(close(got[i].loss, loss[i]), "training BCE");
        c.check(close(trained[i], original[i] - rate * grad[i]), "training SGD update");
    }
    Device<std::uint8_t> crossing_lut(1, c);
    if (!crossing_lut.p) return false;
    cgs::GateRevisions crossing_revision{};
    CircuitStamp crossing_stamp;
    std::uint8_t crossing_value = 0;
    c.cuda(harden(logits.p, 1, crossing_lut.p, nullptr, crossing_revision,
                  &crossing_stamp, status.p, stream), "harden after threshold crossing");
    download(&crossing_value, crossing_lut.p, 1, stream, c);
    if (!c.cuda(cudaStreamSynchronize(stream), "crossing synchronize")) return false;
    c.check(original[2] == 0.0f && trained[2] < 0.0f && ((crossing_value >> 2) & 1u) == 0,
            "training crossing updates hardened decision");

    // Invalid targets or row ids must not partially update another row.
    std::vector<float> bad_labels = labels;
    bad_labels[1] = std::numeric_limits<float>::quiet_NaN();
    upload(logits.p, original.data(), 8, stream, c);
    upload(dev_labels.p, bad_labels.data(), bad_labels.size(), stream, c);
    c.cuda(cudaMemsetAsync(status.p, 0, sizeof(std::uint32_t), stream), "clear invalid status");
    c.cuda(train_step(logits.p, dev_rows.p, dev_labels.p, rows.size(), rate, stats.p, status.p, stream), "invalid training enqueue");
    download(trained.data(), logits.p, 8, stream, c);
    download(&flags, status.p, 1, stream, c);
    if (!c.cuda(cudaStreamSynchronize(stream), "invalid synchronize")) return false;
    c.check((flags & cgs::status_invalid_descriptor) != 0, "invalid sample rejected");
    c.check(trained == original, "invalid sample never partially commits logits");

    // Empty dataset is valid, with zero gradient and unchanged logits.
    upload(logits.p, original.data(), 8, stream, c);
    c.cuda(cudaMemsetAsync(status.p, 0, sizeof(std::uint32_t), stream), "clear empty status");
    c.cuda(train_step(logits.p, nullptr, nullptr, 0, rate, stats.p, status.p, stream), "empty training");
    download(trained.data(), logits.p, 8, stream, c);
    download(got.data(), stats.p, 8, stream, c);
    download(&flags, status.p, 1, stream, c);
    if (!c.cuda(cudaStreamSynchronize(stream), "empty synchronize")) return false;
    c.check(flags == 0 && trained == original, "empty training preserves logits");
    for (auto s : got) c.check(s.count == 0 && s.gradient == 0 && s.loss == 0, "empty statistics");
    return true;
}

bool circuit_checks(cudaStream_t stream, Checks& c) {
    constexpr std::uint32_t tables = 257, batch = 35, words = 2;
    std::vector<float> logits(tables * 8);
    for (unsigned t = 0; t < 256; ++t)
        for (unsigned r = 0; r < 8; ++r) logits[t * 8 + r] = (t & (1u << r)) ? 5.0f : -5.0f;
    for (unsigned r = 0; r < 8; ++r) logits[256 * 8 + r] = 0.0f;
    std::vector<std::uint8_t> expected_luts(tables), actual_luts(tables);
    std::vector<float> expected_change(tables), actual_change(tables);
    for (unsigned t = 0; t < tables; ++t) {
        std::array<float, 8> z{}, labels{};
        for (unsigned r = 0; r < 8; ++r) z[r] = logits[t * 8 + r];
        auto ref = cgs::reference::teacher_reference(z, labels);
        expected_luts[t] = ref.table;
        for (unsigned r = 0; r < 8; ++r)
            expected_change[t] += std::fabs(ref.probability[r] - float(ref.decision[r])) / 8.0f;
    }
    c.check(expected_luts[256] == 255, "threshold equality admits all rows");

    std::vector<Node> nodes(262);
    for (unsigned i = 0; i < 3; ++i) { nodes[i].op = Op::predicate; nodes[i].x = i; }
    for (unsigned t = 0; t < 256; ++t) {
        nodes[3 + t].op = Op::lut3;
        nodes[3 + t].x = 0; nodes[3 + t].y = 1; nodes[3 + t].z = 2; nodes[3 + t].table = t;
    }
    nodes[259].op = Op::conjunction; nodes[259].x = 3 + 0xaa; nodes[259].y = 3 + 0xf0;
    nodes[260].op = Op::disjunction; nodes[260].x = 3 + 0xcc; nodes[260].y = 3 + 0xf0;
    nodes[261].op = Op::negate; nodes[261].x = 3 + 0xaa;
    std::vector<std::uint32_t> roots(259);
    for (unsigned i = 0; i < 256; ++i) roots[i] = i + 3;
    roots[256] = 259; roots[257] = 260; roots[258] = 261;
    const std::uint32_t levels[] = {0, 3, 259, 262};
    cgs::GateRevisions revisions{};
    revisions.structure_epoch = 1; revisions.mapping_revision = 2;
    revisions.accessibility_generation = 3; revisions.parameter_revision = 4;
    revisions.teacher_revision = 5;
    Program program;
    c.check(prepare_program(nodes.data(), nodes.size(), levels, 4, roots.data(), roots.size(), 3, tables, revisions, &program), "prepare DAG");
    auto bad_nodes = nodes;
    bad_nodes[3].x = 3; // same-level dependency cannot be evaluated safely.
    Program rejected;
    c.check(!prepare_program(bad_nodes.data(), bad_nodes.size(), levels, 4, roots.data(), roots.size(), 3, tables, revisions, &rejected), "reject same-level dependency");
    auto bad_roots = roots; bad_roots[0] = nodes.size();
    c.check(!prepare_program(nodes.data(), nodes.size(), levels, 4, bad_roots.data(), bad_roots.size(), 3, tables, revisions, &rejected), "reject out-of-range root");

    std::vector<std::uint32_t> predicate(3 * words, 0);
    for (unsigned cell = 0; cell < batch; ++cell) {
        unsigned row = cell % 8;
        for (unsigned input = 0; input < 3; ++input)
            if (row & (4u >> input)) predicate[input * words + cell / 32] |= 1u << (cell % 32);
    }
    // Deliberately poison predicate tail bits: output tails must still be zero.
    for (unsigned input = 0; input < 3; ++input) predicate[input * words + 1] |= 0xfffffff8u;
    std::vector<std::uint32_t> sample_tables(tables * 8);
    std::vector<std::uint8_t> sample_rows(tables * 8);
    for (unsigned t = 0; t < tables; ++t)
        for (unsigned r = 0; r < 8; ++r) { sample_tables[t * 8 + r] = t; sample_rows[t * 8 + r] = r; }
    Device<float> d_logits(logits.size(), c), d_change(tables, c);
    Device<std::uint8_t> d_luts(tables, c), d_bytes(batch * roots.size(), c);
    Device<std::uint32_t> d_status(1, c), d_pred(predicate.size(), c),
        d_roots(roots.size(), c), d_scratch(nodes.size() * words, c), d_out(roots.size() * words, c);
    Device<std::uint32_t> d_sample_tables(sample_tables.size(), c);
    Device<std::uint8_t> d_sample_rows(sample_rows.size(), c), d_soft_decisions(sample_rows.size(), c);
    Device<float> d_probabilities(sample_rows.size(), c), d_soft_changes(sample_rows.size(), c);
    Device<Node> d_nodes(nodes.size(), c);
    if (!d_logits.p || !d_change.p || !d_luts.p || !d_bytes.p || !d_status.p || !d_pred.p ||
        !d_roots.p || !d_scratch.p || !d_out.p || !d_nodes.p || !d_sample_tables.p ||
        !d_sample_rows.p || !d_soft_decisions.p || !d_probabilities.p || !d_soft_changes.p) return false;
    upload(d_logits.p, logits.data(), logits.size(), stream, c);
    upload(d_nodes.p, nodes.data(), nodes.size(), stream, c);
    upload(d_roots.p, roots.data(), roots.size(), stream, c);
    upload(d_pred.p, predicate.data(), predicate.size(), stream, c);
    upload(d_sample_tables.p, sample_tables.data(), sample_tables.size(), stream, c);
    upload(d_sample_rows.p, sample_rows.data(), sample_rows.size(), stream, c);
    c.cuda(cudaMemsetAsync(d_status.p, 0, sizeof(std::uint32_t), stream), "clear circuit status");
    CircuitStamp stamp;
    c.cuda(harden(d_logits.p, tables, d_luts.p, d_change.p, revisions, &stamp, d_status.p, stream), "harden tables");
    c.cuda(predict_soft(d_logits.p, tables, d_sample_tables.p, d_sample_rows.p, sample_rows.size(),
                        d_probabilities.p, d_soft_decisions.p, d_soft_changes.p, d_status.p, stream), "soft teacher");
    c.cuda(evaluate(program, stamp, revisions, d_nodes.p, d_roots.p, d_luts.p, d_pred.p, batch,
                    d_scratch.p, d_out.p, d_bytes.p, d_status.p, stream), "evaluate DAG");
    std::vector<std::uint32_t> out(roots.size() * words);
    std::vector<std::uint8_t> bytes(batch * roots.size());
    std::vector<float> probabilities(sample_rows.size()), changes(sample_rows.size());
    std::vector<std::uint8_t> decisions(sample_rows.size());
    std::uint32_t flags = 0;
    download(actual_luts.data(), d_luts.p, tables, stream, c);
    download(actual_change.data(), d_change.p, tables, stream, c);
    download(probabilities.data(), d_probabilities.p, probabilities.size(), stream, c);
    download(decisions.data(), d_soft_decisions.p, decisions.size(), stream, c);
    download(changes.data(), d_soft_changes.p, changes.size(), stream, c);
    download(out.data(), d_out.p, out.size(), stream, c);
    download(bytes.data(), d_bytes.p, bytes.size(), stream, c);
    download(&flags, d_status.p, 1, stream, c);
    if (!c.cuda(cudaStreamSynchronize(stream), "circuit synchronize")) return false;
    c.check(flags == 0 && actual_luts == expected_luts, "all hardened LUT decisions equal teacher");
    for (unsigned t = 0; t < tables; ++t) c.check(close(actual_change[t], expected_change[t]), "soft-to-hard discrepancy");
    for (std::size_t i = 0; i < sample_rows.size(); ++i) {
        const auto ref = cgs::reference::teacher_example_reference(logits[i], 0.0f);
        c.check(close(probabilities[i], ref.probability), "soft teacher probability");
        c.check(decisions[i] == ((actual_luts[sample_tables[i]] >> sample_rows[i]) & 1u), "soft teacher threshold equals LUT");
        c.check(close(changes[i], std::fabs(ref.probability - float(decisions[i]))), "sample soft-to-hard discrepancy");
    }
    for (unsigned cell = 0; cell < batch; ++cell) {
        const unsigned row = cell % 8;
        for (unsigned t = 0; t < 256; ++t) {
            bool expected = cgs::reference::teacher_lookup(static_cast<std::uint8_t>(t),
                row & 4u, row & 2u, row & 1u);
            c.check(bytes[cell * roots.size() + t] == unsigned(expected), "DAG truth table byte");
            c.check(((out[t * words + cell / 32] >> (cell % 32)) & 1u) == unsigned(expected), "DAG truth table plane");
        }
        c.check(bytes[cell * roots.size() + 256] == unsigned(bool((row & 1u) && (row & 4u))), "AND composition");
        c.check(bytes[cell * roots.size() + 257] == unsigned(bool((row & 2u) || (row & 4u))), "OR composition");
        c.check(bytes[cell * roots.size() + 258] == unsigned(!(row & 1u)), "NOT composition");
    }
    for (std::size_t root = 0; root < roots.size(); ++root) c.check((out[root * words + 1] & ~7u) == 0, "tail bits masked");

    auto changed = revisions; ++changed.accessibility_generation;
    c.check(evaluate(program, stamp, changed, nullptr, nullptr, d_luts.p, nullptr, 0, nullptr, nullptr, nullptr, d_status.p, stream) == cudaErrorInvalidValue, "stale accessibility rejected");
    changed = revisions; ++changed.teacher_revision;
    Program updated;
    c.check(prepare_program(nodes.data(), nodes.size(), levels, 4, roots.data(), roots.size(), 3, tables, changed, &updated), "new teacher descriptor");
    c.check(evaluate(updated, stamp, changed, nullptr, nullptr, d_luts.p, nullptr, 0, nullptr, nullptr, nullptr, d_status.p, stream) == cudaErrorInvalidValue, "stale hardened teacher rejected after reprepare");
    c.check(evaluate(program, stamp, revisions, nullptr, nullptr, d_luts.p, nullptr, 0, nullptr, nullptr, nullptr, d_status.p, stream) == cudaSuccess, "empty batch");

    // A device-side invalid teacher must not propagate partially hardened LUTs
    // through the DAG. All result words remain their poison sentinel.
    logits[0] = std::numeric_limits<float>::quiet_NaN();
    upload(d_logits.p, logits.data(), logits.size(), stream, c);
    c.cuda(cudaMemsetAsync(d_status.p, 0, sizeof(std::uint32_t), stream), "clear nonfinite status");
    c.cuda(cudaMemsetAsync(d_out.p, 0xa5, out.size() * sizeof(std::uint32_t), stream), "poison DAG output");
    c.cuda(harden(d_logits.p, tables, d_luts.p, d_change.p, revisions, &stamp, d_status.p, stream), "nonfinite harden enqueue");
    c.cuda(evaluate(program, stamp, revisions, d_nodes.p, d_roots.p, d_luts.p, d_pred.p, batch,
                    d_scratch.p, d_out.p, nullptr, d_status.p, stream), "guarded DAG enqueue");
    download(out.data(), d_out.p, out.size(), stream, c);
    download(&flags, d_status.p, 1, stream, c);
    if (!c.cuda(cudaStreamSynchronize(stream), "nonfinite synchronize")) return false;
    c.check((flags & cgs::status_nonfinite) != 0, "nonfinite teacher rejected");
    for (auto word : out) c.check(word == 0xa5a5a5a5u, "failed teacher cannot publish circuit output");
    return true;
}
} // namespace

int run_teacher_tests(cudaStream_t stream) {
    Checks checks;
    train_checks(stream, checks);
    circuit_checks(stream, checks);
    if (!checks.failures) std::puts("teacher CUDA tests: PASS");
    return checks.failures ? 1 : 0;
}
