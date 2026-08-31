#pragma once

#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::operation::v2 {

enum class sparse_update_operation : std::uint8_t {
    assign = 1,
    add = 2,
    subtract = 3,
    multiply = 4,
    maximum = 5
};

enum class sparse_index_type : std::uint8_t {
    u32 = 1,
    u64 = 2
};

struct sparse_axis_update_descriptor {
    sparse_update_operation update = sparse_update_operation::assign;
    execution::persistent_axis_identity target_axis{};
    std::uint64_t target_operand = 0;
    std::uint64_t index_operand = 0;
    std::uint64_t update_operand = 0;
    sparse_index_type index_type = sparse_index_type::u32;
    execution::numeric_type value_type = execution::numeric_type::invalid;
    bool indices_are_unique = false;
    bool indices_are_in_persistent_order = false;
    bool preserve_canonical_identity = true;
    std::uint8_t reserved = 0;
};

enum class composition_kind : std::uint8_t {
    value_generation_to_pack = 1,
    value_pack_to_relation_apply,
    mma_to_residual,
    relation_apply_to_epilogue,
    contraction_to_edge_map,
    contraction_to_segment,
    normalization_to_relation_apply,
    sparse_exchange,
    bundle_to_shared_destination,
    relation_moments_pair
};

struct composition_stage {
    stable_id identity{};
    operation_kind operation = operation_kind::relation_apply;
    std::uint64_t problem_index = 0;
};

struct composition_dependency {
    std::uint64_t producer_stage = 0;
    std::uint64_t consumer_stage = 0;
};

struct composition_descriptor {
    stable_id identity{};
    composition_kind kind = composition_kind::value_generation_to_pack;
    bool experimental = true;
    bool requires_measurement = true;
    bool explicitly_selectable = true;
    bool unfused_stages_available = true;
    const composition_stage *stages = nullptr;
    std::uint64_t stage_count = 0;
    const composition_dependency *dependencies = nullptr;
    std::uint64_t dependency_count = 0;
};

schema_status validate_sparse_axis_update(
    const sparse_axis_update_descriptor &descriptor) noexcept;
schema_status validate_composition(
    const composition_descriptor &descriptor) noexcept;

static_assert(std::is_trivially_copyable_v<sparse_axis_update_descriptor>);
static_assert(std::is_trivially_copyable_v<composition_stage>);
static_assert(std::is_trivially_copyable_v<composition_dependency>);
static_assert(std::is_trivially_copyable_v<composition_descriptor>);

}  // namespace cellerator::compute::operation::v2
