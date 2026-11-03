#pragma once

#include "learning/teacher.hh"

namespace cgs::teacher {

// Host-resident immutable sample partitions for one independent teacher table.
// Training inputs are device-resident flattened arrays. Each table sees
// [first, first+count), using per-observed-row mean BCE and gradient.
struct FitSpan {
    std::uint64_t first = 0, count = 0;
    float learning_rate = 0.5f;
};

struct ChainSpec {
    const Program* program = nullptr;       // host descriptor for after revision
    const FitSpan* fits = nullptr;          // host [table_count]
    std::uint64_t training_sample_count = 0;
    std::uint64_t soft_sample_count = 0;
    std::uint64_t batch = 0;
    std::uint32_t table_count = 0;
    std::uint32_t epochs = 0;               // zero reuses frozen teacher logits
    GateRevisions before{};
    GateRevisions after{};                  // increments teacher_revision on fit
};

// Counts are elements, not bytes. A shared capacity applies to every
// non-null pointer in its named group. Status is always one uint32 word.
struct ChainCapacities {
    std::uint64_t logits = 0, training_samples = 0, statistics = 0, tables = 0;
    std::uint64_t soft_samples = 0, nodes = 0, roots = 0, predicate_words = 0;
    std::uint64_t node_words = 0, gate_words = 0, admission = 0;
};

// Every field is a borrowed device pointer; caller initializes status to zero
// and owns all storage through stream completion.
struct ChainBuffers {
    ChainCapacities capacity{};
    float* logits = nullptr;                      // [tables,8], updated in place
    const std::uint8_t* training_rows = nullptr;  // [training_sample_count]
    const float* training_targets = nullptr;      // [training_sample_count]
    RowStatistics* statistics = nullptr;          // [tables,8]
    std::uint8_t* luts = nullptr;                 // [tables]
    float* table_mean_change = nullptr;           // optional [tables]
    const std::uint32_t* soft_tables = nullptr;  // [soft_sample_count]
    const std::uint8_t* soft_rows = nullptr;     // [soft_sample_count]
    float* soft_probability = nullptr;            // optional [soft_sample_count]
    std::uint8_t* soft_thresholded = nullptr;     // optional [soft_sample_count]
    float* soft_absolute_change = nullptr;        // optional [soft_sample_count]
    const Node* nodes = nullptr;                  // [program.node_count]
    const std::uint32_t* roots = nullptr;         // [program.root_count]
    const std::uint32_t* predicates = nullptr;    // [predicates,ceil(batch/32)]
    std::uint32_t* node_words = nullptr;           // [nodes,ceil(batch/32)]
    std::uint32_t* gate_words = nullptr;           // [roots,ceil(batch/32)]
    std::uint8_t* admission = nullptr;             // optional [batch,roots]
    std::uint32_t* status = nullptr;               // one device word
};

struct PhaseEvents { cudaEvent_t begin = nullptr, end = nullptr; };
struct ChainEvents {
    PhaseEvents training{}, hardening{}, soft_observation{}, circuit{};
};

struct ChainReceipt {
    bool enqueued = false;
    GateRevisions before{}, after{};
    CircuitStamp circuit{};
    GateView gate{};
    std::uint32_t* status = nullptr;
};

// Enqueues complete fitting, hardening, optional soft observations, and circuit
// evaluation in that order on stream. No allocation or synchronization occurs.
// Events, if supplied in begin/end pairs, are recorded on the same stream;
// all fitting, hardening, observation and level/gather launches are included.
// A receipt certifies enqueue only: it and every output are publishable only
// after completion and confirmation that *status == 0. A same-stream consumer
// may be enqueued before that check if it observes the same device status and
// suppresses all work on failure. It describes a Boolean
// gate: no task gradient flows into the teacher, but admitted numerical
// mechanisms still receive their ordinary participant/parameter gradients.
// A failure before or during enqueue invalidates the receipt; an in-flight
// failure may have modified earlier outputs/logits and requires caller recovery.
cudaError_t enqueue_chain(const ChainSpec&, const ChainBuffers&,
                          const ChainEvents&, ChainReceipt*, cudaStream_t) noexcept;

} // namespace cgs::teacher

int run_teacher_chain_tests(cudaStream_t stream);
