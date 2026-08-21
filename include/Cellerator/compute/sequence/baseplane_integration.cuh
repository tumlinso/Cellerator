#pragma once

#include <Baseplane/seq/predicate_plan.hh>
#include <Cellerator/compute/math/operation_core/operation_core.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compute::sequence {

namespace operation_core = cellerator::compute::math::core;

inline constexpr std::uint16_t baseplane_integration_schema_version = 1u;

enum class sequence_strategy : std::uint8_t {
    automatic = 0u,
    materialize_mask = 1u,
    fuse_predicate = 2u
};

// Sorted, non-overlapping local coordinate intervals form one typed physical
// projection from a Baseplane predicate to regulatory-element execution order.
struct regulatory_interval {
    std::uint32_t begin;
    std::uint32_t end;
    std::uint32_t regulatory_element;
    std::uint16_t predicate_id;
    std::uint16_t reserved;
};

// element_offsets and gene_indices describe immutable relation structure.
// Mutable numerical edge values arrive through launch_bindings::values.
struct regulatory_projection_view {
    const regulatory_interval *intervals;
    const std::uint32_t *element_offsets;
    const std::uint32_t *gene_indices;
    std::uint32_t interval_count;
    std::uint32_t regulatory_element_count;
    std::uint32_t gene_count;
    std::uint32_t edge_count;
    execution::device_location location;
};

struct sequence_prepare_policy {
    sequence_strategy requested = sequence_strategy::automatic;
    std::uint32_t expected_predicate_reuse = 1u;
    bool allow_materialization = true;
    bool allow_fusion = true;
    std::uint8_t reserved[2]{};
};

struct sequence_prepare_request {
    const baseplane::seq::sequence_predicate_program *program = nullptr;
    const baseplane::seq::prepared_predicate_plan *baseplane_plan = nullptr;
    execution::structure_id persistent_structure{};
    execution::projection_id persistent_projection{};
    execution::projection_handle projection{};
    execution::relation_structure relation{};
    execution::sequence_domain source_domain{};
    execution::axis_identity predicate_mask_axis{};
    regulatory_projection_view regulatory{};
};

// Caller-owned state outlives the prepared operation. It contains immutable
// structural/projection state and binding contracts, never launch pointers.
struct prepared_sequence_state {
    std::uint16_t schema_version = baseplane_integration_schema_version;
    sequence_strategy strategy = sequence_strategy::automatic;
    std::uint8_t reserved = 0u;
    std::uint64_t predicate_semantic_hash = 0u;
    baseplane::seq::motif32_exact motif{};
    std::uint16_t predicate_id = 0u;
    std::uint16_t output_flags = 0u;
    execution::sequence_domain source_domain{};
    regulatory_projection_view regulatory{};
    execution::operand_axis_contract input_contracts[1]{};
    execution::operand_axis_contract output_contracts[2]{};
    execution::output_axis_contract output_orders[2]{};
};

struct sequence_execution_accounting {
    std::uint64_t packed_sequence_bytes = 0u;
    std::uint64_t plane_and_validity_bytes = 0u;
    std::uint64_t materialized_mask_bytes = 0u;
    std::uint64_t immutable_relation_bytes = 0u;
    std::uint64_t mutable_value_bytes = 0u;
    std::uint64_t output_bytes = 0u;
    std::uint32_t launch_count = 0u;
    std::uint32_t reserved = 0u;
};

bool adapt_baseplane_chunk(
    const baseplane::seq::dna2_chunk_coordinates &source,
    execution::domain_handle genome_domain,
    execution::sequence_domain *destination) noexcept;

bool adapt_baseplane_planes(
    const baseplane::seq::dna2_planes32_valid_stream_view &source,
    execution::axis_identity coordinate_axis,
    baseplane::seq::sequence_buffer_residency residency,
    std::int32_t device_ordinal,
    execution::biological_operand_view *destination) noexcept;

bool validate_regulatory_projection_host(
    const regulatory_projection_view &projection,
    std::uint32_t local_base_count) noexcept;

sequence_strategy select_sequence_strategy(
    const sequence_prepare_policy &policy) noexcept;

operation_core::operation_status prepare_sequence_regulatory_operation(
    const sequence_prepare_request &request,
    const sequence_prepare_policy &policy,
    prepared_sequence_state *state,
    operation_core::prepared_operation *prepared) noexcept;

operation_core::operation_status run_sequence_regulatory_operation(
    const operation_core::prepared_operation &prepared,
    const execution::launch_bindings &launch) noexcept;

sequence_execution_accounting sequence_accounting(
    const prepared_sequence_state &state,
    std::uint32_t base_count) noexcept;

static_assert(sizeof(regulatory_interval) == 16u,
    "regulatory interval projection ABI must remain compact");
static_assert(std::is_trivially_copyable<regulatory_interval>::value,
    "regulatory interval must remain device-copyable");
static_assert(std::is_trivially_copyable<regulatory_projection_view>::value,
    "regulatory projection must remain device-copyable");
static_assert(std::is_trivially_copyable<prepared_sequence_state>::value,
    "prepared sequence state must remain directly bindable");

} // namespace cellerator::compute::sequence
