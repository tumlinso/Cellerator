#pragma once

#include <Baseplane/seq/predicate_plan.hh>
#include <Cellerator/compute/math/operation_core/operation_core.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compute::sequence {

namespace operation_core = cellerator::compute::math::core;

inline constexpr std::uint16_t baseplane_integration_schema_version = 2u;
inline constexpr std::uint32_t required_baseplane_sequence_predicate_abi = 1u;
static_assert(baseplane::seq::sequence_predicate_abi_version
        == required_baseplane_sequence_predicate_abi,
    "Cellerator requires Baseplane sequence predicate ABI version 1");
#if defined(BASEPLANE_SEQUENCE_PREDICATE_ABI_VERSION)
static_assert(BASEPLANE_SEQUENCE_PREDICATE_ABI_VERSION
        == required_baseplane_sequence_predicate_abi,
    "Baseplane target and public header report different predicate ABI versions");
#endif

enum class sequence_strategy : std::uint8_t {
    automatic = 0u,
    materialize_mask = 1u,
    fuse_predicate = 2u
};

// Performance evidence is reusable only for the exact predicate, coordinate
// order, regulatory projection, device performance class, and runtime build
// that produced it. Sequence value generation is deliberately absent: it is a
// launch/cache identity, not a performance-model dimension.
struct sequence_measurement_key {
    std::uint64_t predicate_semantic_hash = 0u;
    execution::order_id coordinate_order{};
    execution::projection_id regulatory_projection{};
    execution::device_performance_class device{};
    std::uint64_t runtime_build_identity = 0u;
    std::uint32_t local_base_count = 0u;
    std::uint16_t predicate_id = 0u;
    std::uint16_t output_flags = 0u;
};

struct sequence_strategy_evidence {
    sequence_measurement_key key{};
    double fused_per_use_ns = 0.0;
    double first_materialized_use_ns = 0.0;
    double cached_materialized_use_ns = 0.0;
    double fused_spread_percent = 0.0;
    double materialized_spread_percent = 0.0;
    std::uint32_t sample_count = 0u;
    std::uint32_t reserved = 0u;
};

struct sequence_strategy_decision {
    sequence_strategy strategy = sequence_strategy::automatic;
    bool empirical_measurement_required = true;
    std::uint8_t reserved[6]{};
    double fused_total_ns = 0.0;
    double materialized_total_ns = 0.0;
    const char *reason = nullptr;
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
    const sequence_strategy_evidence *evidence = nullptr;
    execution::device_performance_class device{};
    std::uint64_t runtime_build_identity = 0u;
    double practical_tolerance_percent = 2.0;
    double maximum_spread_percent = 10.0;
};

struct sequence_prepare_request {
    const baseplane::seq::sequence_predicate_program *program = nullptr;
    const baseplane::seq::prepared_predicate_plan *baseplane_plan = nullptr;
    execution::structure_id persistent_coordinate_structure{};
    execution::structure_id persistent_regulatory_structure{};
    execution::order_id persistent_coordinate_order{};
    execution::projection_id persistent_projection{};
    execution::projection_handle projection{};
    execution::relation_structure coordinate_to_regulatory{};
    execution::relation_structure regulatory_to_gene{};
    execution::sequence_domain source_domain{};
    execution::axis_identity regulatory_axis{};
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
    execution::structure_id persistent_coordinate_structure{};
    execution::order_id persistent_coordinate_order{};
    execution::sequence_domain source_domain{};
    regulatory_projection_view regulatory{};
    execution::operand_axis_contract input_contracts[1]{};
    execution::operand_axis_contract output_contracts[2]{};
    execution::output_axis_contract output_orders[2]{};
    execution::output_effect_contract output_effects[2]{};
};

// A cache entry is mutable session state, never part of prepared_sequence_state.
// The key determines semantic reuse. words and ready_event are physical resource
// bindings supplied by the caller before first use; changing either requires a
// fresh entry. Callers serialize mutation of an entry.
struct predicate_materialization_key {
    execution::value_generation sequence_generation{};
    std::uint64_t predicate_semantic_hash = 0u;
    execution::structure_id coordinate_structure{};
    execution::order_id coordinate_order{};
    std::uint16_t predicate_id = 0u;
    std::uint16_t output_flags = 0u;
    std::uint32_t reserved = 0u;
};

struct predicate_mask_cache_entry {
    predicate_materialization_key key{};
    std::uint32_t *words = nullptr;
    void *ready_event = nullptr;
    std::uint32_t word_capacity = 0u;
    execution::device_location location{};
    bool occupied = false;
    std::uint8_t reserved[7]{};
};

struct predicate_cache_run_result {
    bool cache_hit = false;
    std::uint8_t reserved[3]{};
    std::uint32_t launches_enqueued = 0u;
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

sequence_strategy_decision select_sequence_strategy(
    const sequence_measurement_key &key,
    const sequence_prepare_policy &policy) noexcept;

operation_core::operation_status prepare_sequence_regulatory_operation(
    const sequence_prepare_request &request,
    const sequence_prepare_policy &policy,
    prepared_sequence_state *state,
    operation_core::prepared_operation *prepared) noexcept;

operation_core::operation_status run_sequence_regulatory_operation(
    const operation_core::prepared_operation &prepared,
    const execution::launch_bindings &launch) noexcept;

operation_core::operation_status run_sequence_regulatory_operation_cached(
    const operation_core::prepared_operation &prepared,
    const execution::launch_bindings &launch,
    execution::value_generation sequence_generation,
    predicate_mask_cache_entry *cache,
    predicate_cache_run_result *result) noexcept;

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
static_assert(std::is_trivially_copyable<predicate_materialization_key>::value,
    "predicate cache keys must remain pointer-free values");
static_assert(std::is_trivially_copyable<sequence_strategy_evidence>::value,
    "sequence strategy evidence must remain replaceable data");

} // namespace cellerator::compute::sequence
