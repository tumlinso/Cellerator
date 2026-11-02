#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::ops::condition {

enum class OpcodeV1 : std::uint8_t {
    predicate = 0u,
    all_of = 1u,
    any_of = 2u,
    logical_not = 3u,
};

struct NodeV1 {
    OpcodeV1 opcode = OpcodeV1::predicate;
    std::uint8_t expected_value = 1u;
    std::uint16_t reserved = 0u;
    std::uint32_t first_child = 0u;
    std::uint32_t child_count = 0u;
    std::uint32_t predicate_index = 0u;
    std::uint32_t reserved2 = 0u;
};

struct EdgeBindingV1 {
    std::uint64_t logical_edge_ordinal = 0u;
    std::uint32_t root_node = 0u;
    std::uint32_t reserved = 0u;
};

struct ProgramViewV1 {
    std::uint64_t structure_epoch = 0u;
    const NodeV1* nodes = nullptr;
    std::uint64_t node_count = 0u;
    const EdgeBindingV1* edges = nullptr;
    std::uint64_t binding_count = 0u;
    std::uint64_t edge_count = 0u;
    std::uint32_t predicate_count = 0u;
    std::uint32_t reserved = 0u;
};

enum class StatusV1 : std::uint8_t {
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

inline constexpr std::uint32_t unbound_root_v1 = 0xffffffffu;

StatusV1 validate(const ProgramViewV1& program) noexcept;
StatusV1 prepare_root_map(const ProgramViewV1& program,
                          std::uint32_t* root_node_by_edge_ordinal,
                          std::uint64_t root_map_capacity,
                          std::uint64_t* required_capacity) noexcept;

// Predicate bytes are row-major [batch_count, predicate_count]; bitmap words
// are row-major [batch_count, ceil(edge_count / 64)].
StatusV1 evaluate_bitmaps(const ProgramViewV1& program,
                          std::uint64_t batch_count,
                          const std::uint8_t* predicate_values,
                          std::uint64_t predicate_value_count,
                          std::uint64_t* bitmap_words,
                          std::uint64_t bitmap_word_capacity,
                          std::uint8_t* node_scratch,
                          std::uint64_t node_scratch_capacity) noexcept;

// Sparse update for dirty edge ordinals in one batch row. Existing unrelated
// bitmap bits are preserved.
StatusV1 update_bitmap(const ProgramViewV1& program,
                      const std::uint8_t* predicate_values,
                      std::uint64_t predicate_value_count,
                      const std::uint64_t* dirty_edge_ordinals,
                      std::uint64_t dirty_edge_count,
                      std::uint64_t* bitmap_words,
                      std::uint64_t bitmap_word_capacity,
                      std::uint8_t* node_scratch,
                      std::uint64_t node_scratch_capacity) noexcept;

static_assert(std::is_trivially_copyable<ProgramViewV1>::value,
              "condition programs are borrowed immutable views");

} // namespace cellerator::ops::condition
