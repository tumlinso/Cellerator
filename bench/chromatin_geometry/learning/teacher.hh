#pragma once

// Study-private native CUDA Bernoulli-table teacher and Boolean circuit.
// All pointer arguments below are borrowed device allocations, except where a
// pointer is explicitly described as host. Calls enqueue on the given stream,
// do not allocate device memory, and do not synchronize.
#include "contract.hh"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cgs::teacher {

// One independent Bernoulli logit per row. A sample has row 4*a+2*b+c and a
// target in [0,1]. For each represented row this computes its mean BCE and
// mean gradient, then applies logit -= learning_rate * gradient. Empty rows
// have zero loss and gradient and are unchanged. One sample per row therefore
// reproduces the existing eight-target independent-table training contract.
// Scratch must contain eight RowStatistics. Status is a caller-cleared device
// word. An invalid/nonfinite sample or logit prevents *all* updates. Loss and
// gradient are the pre-update quantities, and outputs must not be used when
// status is nonzero. Input and output arrays may not overlap other arguments.
struct RowStatistics {
    float gradient;
    float loss;
    std::uint64_t count;
};

cudaError_t train_step(float* logits, const std::uint8_t* rows,
                       const float* targets, std::uint64_t sample_count,
                       float learning_rate, RowStatistics* scratch,
                       std::uint32_t* status, cudaStream_t stream) noexcept;

// Harden each eight-logit table with sigmoid(logit) >= 0.5, bit r = row r.
// mean_abs_change measures the mean |p - hard(p)| over its eight rows; it is
// a soft-to-hard discrepancy, not circuit error or biological accuracy.
// Output is only publishable when status remains zero. Hardened gates stop
// task gradients: this API has no backward operation from the Boolean output.
// Host stamp identifies the teacher and structural meaning of LUT storage.
// It does not certify successful device execution: status still must be checked.
struct CircuitStamp {
    GateRevisions revisions{};
    std::uint64_t table_count = 0;
    const std::uint8_t* luts = nullptr;
    bool valid = false;
};
cudaError_t harden(const float* logits, std::uint64_t table_count,
                   std::uint8_t* luts, float* mean_abs_change,
                   GateRevisions revisions, CircuitStamp* stamp,
                   std::uint32_t* status, cudaStream_t stream) noexcept;

// Evaluate the numerical teacher on explicit (table,row) observations. The
// optional per-sample absolute changes let the caller report distribution-
// weighted soft-to-hard differences instead of assuming uniform row use.
// This is the teacher's local numerical value, before any downstream DAG
// composition; soft semantics for arbitrary Boolean compositions are not
// implied. At least one output pointer must be supplied for nonempty input.
cudaError_t predict_soft(const float* logits, std::uint64_t table_count,
                         const std::uint32_t* table_indices,
                         const std::uint8_t* rows, std::uint64_t count,
                         float* probabilities, std::uint8_t* thresholded,
                         float* absolute_changes, std::uint32_t* status,
                         cudaStream_t stream) noexcept;

enum class Op : std::uint8_t { predicate, lut3, conjunction, disjunction, negate };
struct Node {
    Op op = Op::predicate;
    std::uint8_t reserved[3]{};
    // predicate: x is predicate index. lut3: x,y,z are node indices and
    // table is LUT index. Binary ops use x,y; negate uses x.
    std::uint32_t x = 0, y = 0, z = 0, table = 0;
};

// The host owns this immutable descriptor; it caches structure and version
// stamps, not numerical predicate values, circuits, or output planes. Each
// node must occur exactly once in consecutive levels; dependencies must be in
// earlier levels. Device nodes/roots must be byte-identical to validated host
// nodes/roots and remain valid on stream until evaluation completes.
struct Program {
    std::vector<std::uint32_t> levels; // host level offsets, first 0, last N
    std::uint32_t node_count = 0, predicate_count = 0, table_count = 0;
    std::uint32_t root_count = 0;
    GateRevisions revisions{};
    bool valid = false;
};

// Returns false for malformed opcodes, dependencies, roots, or level offsets.
bool prepare_program(const Node* host_nodes, std::uint32_t node_count,
                     const std::uint32_t* host_levels, std::size_t level_count,
                     const std::uint32_t* host_roots, std::uint32_t root_count,
                     std::uint32_t predicate_count, std::uint32_t table_count,
                     GateRevisions revisions, Program* out);

// Predicate and output planes are condition-major [index, ceil(batch/32)].
// scratch_words is [node_count,ceil(batch/32)] and must not overlap inputs or
// outputs. Output words are [root_count,ceil(batch/32)], optional bytes are
// [batch,root_count]. Tail bits are always zero. The current revisions must
// exactly match those used for preparation: accessibility, score, threshold,
// temperature, teacher, model, and mapping changes cannot reuse stale results.
// Repreparation only validates/caches descriptors; caller accounts for this,
// all table hardening, output materialization and each launch in total cost.
cudaError_t evaluate(const Program& program, const CircuitStamp& circuit,
                     const GateRevisions& current,
                     const Node* device_nodes, const std::uint32_t* device_roots,
                     const std::uint8_t* device_luts,
                     const std::uint32_t* predicates, std::uint64_t batch,
                     std::uint32_t* scratch_words, std::uint32_t* output_words,
                     std::uint8_t* output_bytes, std::uint32_t* status,
                     cudaStream_t stream) noexcept;

} // namespace cgs::teacher

// Intended for an admitted Scheduler job, not an unscheduled host run.
int run_teacher_tests(cudaStream_t stream);
