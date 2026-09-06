#pragma once
// Internal single-device/single-stream native contract. See
// docs/semantic_spine_v1/native_contract.md for ownership and failure semantics.
#include <Cellerator/compute/operation/relation_semantics.hh>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace cellerator::compute::relation {
// Destination rows, source columns, offsets in declared logical edge order.
// Preparation consumes host arrays synchronously; never borrows them after return.
struct csr_host_view {
    const std::uint32_t* row_offsets = nullptr;
    std::uint64_t row_offset_count = 0;
    const std::uint32_t* source_indices = nullptr;
    std::uint64_t source_index_count = 0;
};
struct preparation_options {
    int device_ordinal = 0;
    std::uint64_t persistent_byte_limit = 0; // zero: no extra user limit
};
struct device_values_binding {
    const void* f16_data = nullptr; // IEEE binary16 bytes in logical edge order
    std::uint64_t count = 0;
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    execution::order_id logical_edge_order{};
    execution::value_generation generation{};
    int device_ordinal = 0;
};
struct device_state_view {
    const void* data = nullptr;
    std::uint64_t count = 0; // f32 elements in the supported N1 path
    axis_descriptor axis{};
    int device_ordinal = 0;
};
struct device_result_view {
    void* data = nullptr;
    std::uint64_t count = 0;
    axis_descriptor axis{};
    int device_ordinal = 0;
};
struct prepared_relation_pair; // owns cold preparation; no general-purpose container API
struct preparation_report {
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    execution::projection_id forward_projection{};
    execution::projection_id transpose_projection{};
    execution::value_generation latest_enqueued_generation{};
    std::uint64_t topology_preparations = 0;
    std::uint64_t value_refreshes = 0;
    std::uint64_t accepted_forward_launches = 0;
    std::uint64_t accepted_transpose_launches = 0;
    const char* forward_candidate = nullptr; // actual bound implementation
    const char* transpose_candidate = nullptr;
};
// Failure clears *out (when out is nonnull) and releases partial preparation.
// Input *out must be null: replacing an existing owned handle is not supported.
status prepare_relation_pair(const operation_descriptor& forward,
                             const operation_descriptor& transpose,
                             const csr_host_view& topology,
                             const preparation_options& options,
                             cudaStream_t stream,
                             prepared_relation_pair** out) noexcept;
// Strictly increasing nonzero generation; borrowed values live through stream completion.
// Successful submission publishes the enqueued generation, not GPU completion.
status publish_values(prepared_relation_pair&, const device_values_binding&,
                      cudaStream_t stream) noexcept;
// Validate all metadata/capacities/overlap before submission. Metadata rejection
// preserves output and counters. CUDA partial-submission failure poisons the pair.
status enqueue(prepared_relation_pair&, const operation_descriptor&,
               const device_state_view&, const device_result_view&,
               execution::value_generation expected_values,
               cudaStream_t stream) noexcept;
status inspect(const prepared_relation_pair&, preparation_report*) noexcept;
// Host calls on one pair are externally serialized. inspect does not synchronize.
// Teardown may fence the pair's stream; repeated enqueue/refresh must not globally synchronize.
void destroy(prepared_relation_pair*) noexcept;
} // namespace cellerator::compute::relation
