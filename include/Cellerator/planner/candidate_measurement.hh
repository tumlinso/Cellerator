#pragma once

#include <Cellerator/planner/end_to_end_planner.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::planner {

inline constexpr std::uint32_t candidate_measurement_schema_version = 1u;
inline constexpr std::uint32_t maximum_measurement_warmups = 4u;
inline constexpr std::uint32_t maximum_measurement_samples = 9u;

// Optional per-run work enqueued on the same caller/session stream as the
// prepared operation. These callbacks let the measurement path include
// candidate-dependent packing, epilogue, order, or communication without
// forcing absent phases to run or synchronizing between phases.
using enqueue_phase_function = bool (*)(
    void *context,
    execution::stream_context stream) noexcept;

struct enqueue_phase {
    void *context = nullptr;
    enqueue_phase_function enqueue = nullptr;
};

// Invoked only after a complete untimed pipeline and a stream-local
// synchronization. It must independently validate semantics, order, effects,
// and numerical output before any samples are accepted.
using candidate_referee_function = bool (*)(
    void *context,
    const execution::launch_bindings &private_launch) noexcept;

struct candidate_measurement_entry {
    std::uint32_t schema_version = candidate_measurement_schema_version;
    operation_core::stable_id candidate{};
    operation_core::projection_key projection{};
    const operation_core::prepared_operation *prepared = nullptr;
    execution::launch_bindings private_launch{};
    const void *caller_visible_output = nullptr;
    phase_costs premeasured{};
    enqueue_phase dynamic_input_pack{};
    enqueue_phase epilogue{};
    enqueue_phase output_order{};
    enqueue_phase communication{};
    void *referee_context = nullptr;
    candidate_referee_function referee = nullptr;
    std::uint32_t warmup_count = 1u;
    std::uint32_t sample_count = 5u;
};

struct candidate_measurement_session {
    const candidate_measurement_entry *entries = nullptr;
    std::uint32_t entry_count = 0u;
};

// Real planner measurement hook. It uses CUDA events on the launch stream,
// never device-wide synchronization, performs bounded warmup/repeats, uses a
// median plus median-absolute-deviation spread, and allocates no output or
// scratch. The caller owns candidate-private outputs and all phase state.
bool measure_prepared_candidate(
    void *context,
    const planner_candidate &candidate,
    measured_candidate *measurement) noexcept;

static_assert(std::is_trivially_copyable<enqueue_phase>::value,
    "measurement phase binding must remain pointer-copyable");
static_assert(std::is_trivially_copyable<candidate_measurement_entry>::value,
    "measurement entry must remain pointer-copyable");

} // namespace cellerator::planner
