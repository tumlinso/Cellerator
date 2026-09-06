#pragma once
// PLANNING CONTRACT SKETCH, not installed Cellerator implementation.
// C01 may refine this with all consumers before I02 publishes the internal interface.
#include <Cellerator/execution/identity.hh>
#include <Cellerator/execution/operands.hh>
#include <cstdint>

namespace cellerator::compute::relation {
struct axis_descriptor {
    execution::persistent_axis_identity identity{};
    std::uint64_t extent = 0;
};
struct topology_descriptor {
    execution::structure_id identity{};
    execution::structure_epoch epoch{};
    axis_descriptor source{};
    axis_descriptor destination{};
    execution::order_id logical_edge_order{};
    std::uint64_t edge_count = 0;
};
enum class orientation : std::uint8_t { forward, transpose };
enum class output_update : std::uint8_t { overwrite, accumulate, affine_accumulate };
enum class nonfinite_policy : std::uint8_t { propagate, reject };
struct arithmetic_policy {
    execution::numeric_type relation_storage = execution::numeric_type::f16;
    execution::numeric_type input_storage = execution::numeric_type::f32;
    execution::numeric_type multiply = execution::numeric_type::f32;
    execution::numeric_type accumulation = execution::numeric_type::f32;
    execution::numeric_type output_storage = execution::numeric_type::f32;
    bool permit_fma = true;
    bool permit_reassociation = true;
    nonfinite_policy nonfinite = nonfinite_policy::propagate;
};
struct operation_descriptor {
    topology_descriptor topology{};
    orientation direction = orientation::forward;
    arithmetic_policy arithmetic{};
    std::uint32_t dense_width = 1;
    output_update update = output_update::overwrite;
    bool input_output_aliasing_legal = false;
};
enum class status_code : std::uint8_t {
    ok, invalid_argument, invalid_identity, invalid_shape, invalid_axis,
    unsupported_semantics, unsupported_numeric_policy, unsupported_width,
    stale_structure, stale_generation, incompatible_order, incompatible_stream,
    incompatible_device, insufficient_capacity, cuda_failure, invalid_state
};
struct status {
    status_code code = status_code::ok;
    const char* message = "ok";
    constexpr explicit operator bool() const noexcept { return code == status_code::ok; }
};
// Definition belongs in relation_semantics.cc. Comparisons are fieldwise, not memcmp.
status validate(const operation_descriptor&) noexcept;
bool equivalent(const operation_descriptor&, const operation_descriptor&) noexcept;
} // namespace cellerator::compute::relation
