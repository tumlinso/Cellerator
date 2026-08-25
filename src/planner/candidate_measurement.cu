#include <Cellerator/planner/candidate_measurement.hh>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cellerator::planner {
namespace {

enum event_index : std::uint32_t {
    pipeline_begin = 0u,
    after_dynamic_pack = 1u,
    after_kernel = 2u,
    after_epilogue = 3u,
    after_output_order = 4u,
    pipeline_end = 5u,
    event_count = 6u
};

bool same_candidate(
    const candidate_measurement_entry &entry,
    const planner_candidate &candidate) noexcept {
    return operation_core::same_stable_id(entry.candidate, candidate.identity)
        && execution::same_identity(
            entry.projection.persistent, candidate.projection.persistent)
        && execution::same_handle(
            entry.projection.runtime, candidate.projection.runtime)
        && entry.projection.kind == candidate.projection.kind
        && entry.projection.schema_version == candidate.projection.schema_version
        && entry.projection.variant == candidate.projection.variant;
}

bool enqueue_optional(
    const enqueue_phase &phase,
    execution::stream_context stream) noexcept {
    return phase.enqueue == nullptr || phase.enqueue(phase.context, stream);
}

bool run_pipeline(const candidate_measurement_entry &entry) noexcept {
    return enqueue_optional(entry.dynamic_input_pack, entry.private_launch.stream)
        && static_cast<bool>(operation_core::run_prepared_operation(
            *entry.prepared, entry.private_launch))
        && enqueue_optional(entry.epilogue, entry.private_launch.stream)
        && enqueue_optional(entry.output_order, entry.private_launch.stream)
        && enqueue_optional(entry.communication, entry.private_launch.stream);
}

bool valid_private_output(
    const candidate_measurement_entry &entry) noexcept {
    if (entry.private_launch.output_count != 1u
        || entry.private_launch.outputs == nullptr
        || entry.prepared->binding_contract.output_effect_count != 1u
        || entry.prepared->binding_contract.output_effects == nullptr
        || entry.prepared->binding_contract.output_effects[0].update
            != execution::output_update_kind::overwrite)
        return false;
    const void *private_output = execution::operand_data_address(
        entry.private_launch.outputs[0]);
    return private_output != nullptr && entry.caller_visible_output != nullptr
        && private_output != entry.caller_visible_output;
}

double median(double *values, std::uint32_t count) noexcept {
    std::sort(values, values + count);
    if ((count & 1u) != 0u) return values[count / 2u];
    return 0.5 * (values[count / 2u - 1u] + values[count / 2u]);
}

double robust_spread_percent(
    const double *totals,
    std::uint32_t count,
    double center) noexcept {
    double deviations[maximum_measurement_samples]{};
    for (std::uint32_t index = 0u; index < count; ++index)
        deviations[index] = std::fabs(totals[index] - center);
    const double mad = median(deviations, count);
    return center == 0.0 ? (mad == 0.0 ? 0.0 : 100.0)
        : mad * 100.0 / center;
}

bool elapsed_ns(
    cudaEvent_t begin,
    cudaEvent_t end,
    double *nanoseconds) noexcept {
    float milliseconds = 0.0f;
    if (cudaEventElapsedTime(&milliseconds, begin, end) != cudaSuccess
        || !std::isfinite(milliseconds) || milliseconds < 0.0f)
        return false;
    *nanoseconds = static_cast<double>(milliseconds) * 1.0e6;
    return true;
}

bool record_sample(
    const candidate_measurement_entry &entry,
    cudaEvent_t *events,
    double *dynamic_ns,
    double *kernel_ns,
    double *epilogue_ns,
    double *order_ns,
    double *communication_ns) noexcept {
    const cudaStream_t stream = static_cast<cudaStream_t>(
        entry.private_launch.stream.stream);
    if (cudaEventRecord(events[pipeline_begin], stream) != cudaSuccess
        || !enqueue_optional(entry.dynamic_input_pack,
            entry.private_launch.stream)
        || cudaEventRecord(events[after_dynamic_pack], stream) != cudaSuccess)
        return false;
    if (!operation_core::run_prepared_operation(
            *entry.prepared, entry.private_launch)
        || cudaEventRecord(events[after_kernel], stream) != cudaSuccess)
        return false;
    if (!enqueue_optional(entry.epilogue, entry.private_launch.stream)
        || cudaEventRecord(events[after_epilogue], stream) != cudaSuccess
        || !enqueue_optional(entry.output_order, entry.private_launch.stream)
        || cudaEventRecord(events[after_output_order], stream) != cudaSuccess
        || !enqueue_optional(entry.communication, entry.private_launch.stream)
        || cudaEventRecord(events[pipeline_end], stream) != cudaSuccess
        || cudaEventSynchronize(events[pipeline_end]) != cudaSuccess)
        return false;
    *dynamic_ns = 0.0;
    *epilogue_ns = 0.0;
    *order_ns = 0.0;
    *communication_ns = 0.0;
    return (entry.dynamic_input_pack.enqueue == nullptr
            || elapsed_ns(events[pipeline_begin], events[after_dynamic_pack],
                dynamic_ns))
        && elapsed_ns(events[after_dynamic_pack], events[after_kernel], kernel_ns)
        && (entry.epilogue.enqueue == nullptr
            || elapsed_ns(events[after_kernel], events[after_epilogue],
                epilogue_ns))
        && (entry.output_order.enqueue == nullptr
            || elapsed_ns(events[after_epilogue], events[after_output_order],
                order_ns))
        && (entry.communication.enqueue == nullptr
            || elapsed_ns(events[after_output_order], events[pipeline_end],
                communication_ns));
}

} // namespace

bool measure_prepared_candidate(
    void *context,
    const planner_candidate &candidate,
    measured_candidate *measurement) noexcept {
    if (context == nullptr || measurement == nullptr) return false;
    *measurement = {};
    const auto &session = *static_cast<const candidate_measurement_session *>(
        context);
    if (session.entries == nullptr || session.entry_count == 0u) return false;
    const candidate_measurement_entry *entry = nullptr;
    for (std::uint32_t index = 0u; index < session.entry_count; ++index)
        if (same_candidate(session.entries[index], candidate)) {
            if (entry != nullptr) return false;
            entry = &session.entries[index];
        }
    if (entry == nullptr
        || entry->schema_version != candidate_measurement_schema_version
        || entry->prepared == nullptr || entry->referee == nullptr
        || entry->warmup_count > maximum_measurement_warmups
        || entry->sample_count < 3u
        || entry->sample_count > maximum_measurement_samples
        || entry->private_launch.stream.device_ordinal < 0
        || !operation_core::same_stable_id(
            entry->prepared->kernel, candidate.identity)
        || !execution::same_identity(entry->projection.persistent,
            entry->prepared->projection.persistent)
        || !execution::same_handle(entry->projection.runtime,
            entry->prepared->projection.runtime)
        || entry->projection.kind != entry->prepared->projection.kind
        || entry->projection.schema_version
            != entry->prepared->projection.schema_version
        || entry->projection.variant != entry->prepared->projection.variant
        || !operation_core::validate_prepared_operation(*entry->prepared)
        || !valid_private_output(*entry))
        return false;

    const cudaStream_t stream = static_cast<cudaStream_t>(
        entry->private_launch.stream.stream);
    if (cudaSetDevice(entry->private_launch.stream.device_ordinal) != cudaSuccess
        || !run_pipeline(*entry) || cudaStreamSynchronize(stream) != cudaSuccess
        || !entry->referee(entry->referee_context, entry->private_launch))
        return false;

    for (std::uint32_t warmup = 0u; warmup < entry->warmup_count; ++warmup)
        if (!run_pipeline(*entry)) return false;
    if (entry->warmup_count != 0u
        && cudaStreamSynchronize(stream) != cudaSuccess)
        return false;

    cudaEvent_t events[event_count]{};
    for (std::uint32_t index = 0u; index < event_count; ++index)
        if (cudaEventCreate(&events[index]) != cudaSuccess) {
            for (std::uint32_t prior = 0u; prior < index; ++prior)
                (void)cudaEventDestroy(events[prior]);
            return false;
        }

    double dynamic_samples[maximum_measurement_samples]{};
    double kernel_samples[maximum_measurement_samples]{};
    double epilogue_samples[maximum_measurement_samples]{};
    double order_samples[maximum_measurement_samples]{};
    double communication_samples[maximum_measurement_samples]{};
    double total_samples[maximum_measurement_samples]{};
    bool ok = true;
    for (std::uint32_t sample = 0u; sample < entry->sample_count; ++sample) {
        ok = record_sample(*entry, events,
            &dynamic_samples[sample], &kernel_samples[sample],
            &epilogue_samples[sample], &order_samples[sample],
            &communication_samples[sample]);
        if (!ok) break;
        total_samples[sample] = dynamic_samples[sample] + kernel_samples[sample]
            + epilogue_samples[sample] + order_samples[sample]
            + communication_samples[sample];
    }
    for (cudaEvent_t event : events) (void)cudaEventDestroy(event);
    if (!ok) return false;

    double spread_samples[maximum_measurement_samples]{};
    for (std::uint32_t index = 0u; index < entry->sample_count; ++index)
        spread_samples[index] = total_samples[index];
    const double total_median = median(spread_samples, entry->sample_count);
    measurement->correct = true;
    measurement->sample_count = entry->sample_count;
    measurement->phases = entry->premeasured;
    measurement->phases.persistent_bytes = std::max(
        measurement->phases.persistent_bytes,
        candidate.operation->persistent_bytes);
    measurement->phases.transient_bytes = std::max(
        measurement->phases.transient_bytes,
        candidate.operation->transient_bytes);
    measurement->phases.dynamic_input_pack_ns += median(
        dynamic_samples, entry->sample_count);
    measurement->phases.kernel_ns += median(
        kernel_samples, entry->sample_count);
    measurement->phases.epilogue_ns += median(
        epilogue_samples, entry->sample_count);
    measurement->phases.order_transform_ns += median(
        order_samples, entry->sample_count);
    measurement->phases.communication_ns += median(
        communication_samples, entry->sample_count);
    measurement->spread_percent = robust_spread_percent(
        total_samples, entry->sample_count, total_median);
    measurement->contaminated = !std::isfinite(measurement->spread_percent);
    return !measurement->contaminated;
}

} // namespace cellerator::planner
