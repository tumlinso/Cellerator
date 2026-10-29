#pragma once

#include <Cellerator/compute/indexing/reverse_postings.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::gating {

enum class condition_opcode_v1 : std::uint8_t {
    predicate = 0u,
    all_of = 1u,
    any_of = 2u,
    logical_not = 3u,
};

struct condition_node_v1 {
    condition_opcode_v1 opcode = condition_opcode_v1::predicate;
    std::uint8_t expected_value = 1u;
    std::uint16_t reserved = 0u;
    std::uint32_t first_child = 0u;
    std::uint32_t child_count = 0u;
    std::uint32_t predicate_index = 0u;
    std::uint32_t reserved2 = 0u;
};

struct edge_condition_binding_v1 {
    std::uint64_t logical_edge_ordinal = 0u;
    std::uint32_t root_node = 0u;
    std::uint32_t reserved = 0u;
};

struct condition_program_view_v1 {
    std::uint64_t structure_epoch = 0u;
    const condition_node_v1 *nodes = nullptr;
    std::uint64_t node_count = 0u;
    const edge_condition_binding_v1 *edges = nullptr;
    std::uint64_t binding_count = 0u;
    std::uint64_t edge_count = 0u; // total logical edges in the snapshot
    std::uint32_t predicate_count = 0u;
    std::uint32_t reserved = 0u;
};

enum class condition_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_program,
    epoch_mismatch,
    insufficient_bitmap_capacity,
    invalid_workset,
    insufficient_offsets_capacity,
    primal_workset_overflow,
    derivative_workset_overflow,
    insufficient_root_map_capacity,
};

// Dense edge-ordinal maps use this value for an edge with no condition
// binding (such edges are eligible whenever evaluated).
static constexpr std::uint32_t unbound_condition_root_v1 = 0xffffffffu;

condition_status_v1 validate_condition_program_v1(
    const condition_program_view_v1 &program) noexcept;

// Compile the sparse sorted edge bindings into caller-owned dense storage.
// `required_capacity` is always set after the program validates. On a short
// buffer no entries are written. The result has one root node per logical
// edge ordinal and is suitable for copying into a device-resident prepared
// CUDA condition descriptor.
condition_status_v1 prepare_condition_root_map_v1(
    const condition_program_view_v1 &program,
    std::uint32_t *root_node_by_edge_ordinal,
    std::uint64_t root_map_capacity,
    std::uint64_t *required_capacity) noexcept;

// Predicate bytes are row-major [cell_count, predicate_count]. Bitmap words
// are row-major [cell_count, ceil(edge_count/64)].
condition_status_v1 evaluate_condition_bitmaps_v1(
    const condition_program_view_v1 &program, std::uint64_t cell_count,
    const std::uint8_t *predicate_values,
    std::uint64_t predicate_value_count,
    std::uint64_t *bitmap_words, std::uint64_t bitmap_word_capacity,
    std::uint8_t *node_scratch, std::uint64_t node_scratch_capacity) noexcept;

// Sparse update for dirty edge ordinals in one cell. Existing unrelated bitmap
// bits are preserved. The predicate vector is one cell's predicate row.
condition_status_v1 update_condition_bitmap_v1(
    const condition_program_view_v1 &program,
    const std::uint8_t *predicate_values, std::uint64_t predicate_value_count,
    const std::uint64_t *dirty_edge_ordinals, std::uint64_t dirty_edge_count,
    std::uint64_t *bitmap_words, std::uint64_t bitmap_word_capacity,
    std::uint8_t *node_scratch, std::uint64_t node_scratch_capacity) noexcept;

struct per_cell_candidate_view_v1 {
    const std::uint64_t *offsets = nullptr; // cell_count + 1
    const indexing::indexed_work_item_v1 *items = nullptr;
    std::uint64_t item_count = 0u;
    // Each cell slice must be strictly ordered by logical_edge_id and unique.
};

struct per_cell_workset_v1 {
    std::uint64_t *offsets = nullptr; // cell_count + 1
    std::uint64_t offsets_capacity = 0u;
    indexing::indexed_work_item_v1 *items = nullptr;
    std::uint64_t item_capacity = 0u;
    std::uint64_t written = 0u;
    std::uint64_t required = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t state_generation = 0u;
};

struct gate_workset_pair_v1 {
    per_cell_workset_v1 primal;
    per_cell_workset_v1 derivative;
};

// Condition-filter both candidate streams. Derivative work is the unique union
// of condition-eligible primal candidates and condition-eligible derivative
// proposals. Capacity is checked before either output is populated.
condition_status_v1 build_condition_worksets_v1(
    const condition_program_view_v1 &program, std::uint64_t cell_count,
    std::uint64_t state_generation,
    const std::uint64_t *bitmap_words, std::uint64_t bitmap_word_count,
    const per_cell_candidate_view_v1 &primal_candidates,
    const per_cell_candidate_view_v1 &derivative_candidates,
    gate_workset_pair_v1 *output) noexcept;

static_assert(std::is_trivially_copyable<condition_program_view_v1>::value,
              "condition programs are borrowed immutable views");
static_assert(std::is_trivially_copyable<per_cell_candidate_view_v1>::value,
              "candidate lists are compact borrowed views");

} // namespace cellerator::compute::gating
