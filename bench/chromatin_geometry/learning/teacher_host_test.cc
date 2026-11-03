#include "learning/teacher.hh"
#include "learning/teacher_chain.hh"

#include <cstdint>
#include <cstdio>
#include <vector>

// Host-only structural and revision checks; no CUDA runtime call is made for
// the zero-batch queries below and this test must not create a device context.
int main() {
    using namespace cgs::teacher;
    int failures = 0;
    const auto check = [&](bool ok, const char* message) {
        if (!ok) { std::fprintf(stderr, "teacher host: %s\n", message); ++failures; }
    };
    Node nodes[2]{};
    nodes[0].op = Op::predicate;
    nodes[1].op = Op::negate; nodes[1].x = 0;
    const std::uint32_t levels[]{0, 1, 2};
    const std::uint32_t roots[]{1};
    cgs::GateRevisions version{};
    version.structure_epoch = 1;
    version.mapping_revision = 2;
    version.accessibility_generation = 3;
    version.score_generation = 4;
    version.parameter_revision = 5;
    version.teacher_revision = 6;
    version.temporal_generation = 7;
    version.primal_generation = 8;
    version.derivative_generation = 9;
    version.temperature_bits = 10;
    version.threshold_bits = 11;
    Program p;
    check(prepare_program(nodes, 2, levels, 3, roots, 1, 1, 0, version, &p), "valid program rejected");
    CircuitStamp stamp{version, 0, nullptr, true};
    std::uint32_t placeholder = 0; // never dereferenced: batch is zero
    const auto call = [&](const Program& q, const CircuitStamp& s, const cgs::GateRevisions& r) {
        return evaluate(q, s, r, nullptr, nullptr, nullptr, nullptr, 0,
                        nullptr, nullptr, nullptr, &placeholder, nullptr);
    };
    check(call(p, stamp, version) == cudaSuccess, "valid empty query rejected");
    std::vector<cgs::GateRevisions> stale(11, version);
    ++stale[0].structure_epoch; ++stale[1].mapping_revision;
    ++stale[2].accessibility_generation; ++stale[3].score_generation;
    ++stale[4].parameter_revision; ++stale[5].teacher_revision;
    ++stale[6].temporal_generation; ++stale[7].primal_generation;
    ++stale[8].derivative_generation; ++stale[9].temperature_bits;
    ++stale[10].threshold_bits;
    for (const auto& r : stale)
        check(call(p, stamp, r) == cudaErrorInvalidValue, "stale program accepted");
    for (unsigned index : {0u, 1u, 4u, 5u}) {
        Program updated;
        check(prepare_program(nodes, 2, levels, 3, roots, 1, 1, 0, stale[index], &updated), "updated program rejected");
        check(call(updated, stamp, stale[index]) == cudaErrorInvalidValue, "stale circuit accepted");
    }
    Program invalid;
    Node bad[2]{nodes[0], nodes[1]}; bad[1].x = 1;
    check(!prepare_program(bad, 2, levels, 3, roots, 1, 1, 0, version, &invalid), "cycle accepted");
    const std::uint32_t bad_levels[]{0, 2, 1};
    check(!prepare_program(nodes, 2, bad_levels, 3, roots, 1, 1, 0, version, &invalid), "bad levels accepted");
    const std::uint32_t bad_roots[]{2};
    check(!prepare_program(nodes, 2, levels, 3, bad_roots, 1, 1, 0, version, &invalid), "bad root accepted");

    // Chain preflight failures must occur before any GPU operation or partial
    // fitting, including invalid revisions, event pairs and buffer capacities.
    Program with_table;
    check(prepare_program(nodes, 2, levels, 3, roots, 1, 1, 1, version, &with_table), "table program rejected");
    ChainSpec chain;
    chain.program = &with_table; chain.table_count = 1;
    chain.before = version; chain.after = version;
    ChainBuffers buffers;
    buffers.status = &placeholder;
    buffers.logits = reinterpret_cast<float*>(std::uintptr_t(1));
    buffers.luts = reinterpret_cast<std::uint8_t*>(std::uintptr_t(1));
    buffers.capacity.logits = 7; buffers.capacity.tables = 1;
    ChainReceipt receipt;
    check(enqueue_chain(chain, buffers, {}, &receipt, nullptr) == cudaErrorInvalidValue && !receipt.enqueued,
          "short logit buffer accepted");
    buffers.capacity.logits = 8;
    ChainEvents broken_events;
    broken_events.training.begin = reinterpret_cast<cudaEvent_t>(std::uintptr_t(1));
    check(enqueue_chain(chain, buffers, broken_events, &receipt, nullptr) == cudaErrorInvalidValue && !receipt.enqueued,
          "unpaired phase event accepted");
    chain.epochs = 1;
    check(enqueue_chain(chain, buffers, {}, &receipt, nullptr) == cudaErrorInvalidValue && !receipt.enqueued,
          "fit without revision/fit buffers accepted");
    if (!failures) std::puts("teacher host structural/revision checks: PASS");
    return failures ? 1 : 0;
}
