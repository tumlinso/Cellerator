#pragma once
// Internal integration contract. Runtime implementations land in the native and
// readiness lanes; no installed SDK or device capability is implied by this header.
#include <Cellerator/compute/operation/prepared_relation.hh>
#include <Cellerator/compute/operation/relation_calculus.hh>
#include <cstdint>
namespace cellerator::compute::relation {
enum class gradient_route : std::uint8_t { automatic, force_sparse, force_hybrid };
struct gradient_preparation_options {
    gradient_route route = gradient_route::automatic;
    std::uint64_t scratch_byte_limit = 0;
};
struct operand_version { std::uint64_t identity = 0, version = 0; };
struct edge_layout_view {
    execution::order_id order{};
    std::uint64_t count = 0;
    const std::uint32_t* logical_to_physical = nullptr; // host; immutable through pair lifetime
};
struct edge_plane_view {
    void* f32_data = nullptr;
    std::uint64_t count = 0;
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    execution::order_id order{};
    std::int32_t device_ordinal = 0;
};
struct gradient_stamp {
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    execution::order_id order{};
    execution::value_generation forward_generation{};
    operand_version input{}, cotangent{};
    std::uint64_t producer_serial = 0;
    std::uint64_t pair_incarnation = 0; // process-unique lifetime, not a pointer
    gradient_arithmetic arithmetic = gradient_arithmetic::full_f32;
};
struct value_update_request {
    value_update_kind kind = value_update_kind::delta_add;
    edge_plane_view operand; // read-only during update despite mutable view representation
    execution::value_generation expected{}, next{};
    float alpha = 0;
    gradient_stamp gradient{}; // required for gradient_step, not for independent caller delta
};
struct value_read_lease {
    const void* physical_f16_values = nullptr;
    std::uint64_t count = 0;
    execution::order_id order{};
    execution::value_generation generation{};
    std::uint64_t nonce = 0; // opaque token; never user-authored
    std::uint64_t pair_incarnation = 0; // rejects cross-pair and address-reuse replay
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    std::int32_t device_ordinal = -1;
};
struct relation_update_report {
    preparation_report relation;
    std::uint64_t gradient_preparations = 0, gradient_launches = 0;
    std::uint64_t physical_updates = 0, wmma_launches = 0, residual_launches = 0;
    std::uint64_t sparse_launches = 0, operand_pack_refreshes = 0;
    std::uint64_t implicit_canonicalizations = 0;
    std::uint64_t persistent_bytes = 0, scratch_bytes = 0;
    std::uint64_t ready_records = 0, reader_returns = 0;
};
// All entrypoints are externally host-serialized and owner-stream-only except
// begin/end read. Operands must not overlap pair-owned values, immutable
// storage or internal scratch. Gradient output must not overlap either supplied
// dense input; dense-operation input/output ranges must be disjoint. A delta may
// alias caller buffers read-only during its update. No earlier caller-buffer
// lifetime is retained merely to forbid safe read/read aliasing.
// Counts are element capacities checked with overflow-safe byte arithmetic.
// Gradient stamps bind pair lifetime, generation, input versions and producer
// serial; stamp replay after another gradient or update is rejected.
// Cold extension of the existing pair, not a separate prepared-training object.
status prepare_relation_gradient(prepared_relation_pair&,const relation_calculus_descriptor&,
    const gradient_preparation_options&,cudaStream_t) noexcept;
status inspect_edge_layout(const prepared_relation_pair&,edge_layout_view*) noexcept;
status enqueue_edge_gradient(prepared_relation_pair&,const relation_calculus_descriptor&,
    const device_state_view& input,const device_state_view& cotangent,
    operand_version input_version,operand_version cotangent_version,
    execution::value_generation expected,const edge_plane_view& output,
    gradient_stamp* produced,cudaStream_t) noexcept;
status enqueue_value_update(prepared_relation_pair&,const value_update_request&,cudaStream_t) noexcept;
// New mutable/lease operations reject stream capture before side effects.
// One outstanding external const lease at a time. Execution remains owner-stream-only.
status begin_value_read(prepared_relation_pair&,execution::value_generation,cudaStream_t consumer,
    value_read_lease*) noexcept;
// Records consumer done, enqueues owner wait, invalidates lease. No host-global fence.
status end_value_read(prepared_relation_pair&,value_read_lease&,cudaStream_t consumer) noexcept;
status inspect_updates(const prepared_relation_pair&,relation_update_report*) noexcept;
// Reject outstanding unreturned leases; drain valid owned dependencies; destroy on success.
status close_relation_pair(prepared_relation_pair**) noexcept;
} // namespace cellerator::compute::relation
