#pragma once

#include <Cellerator/compiler/sema/implement_axis_semantics_v1.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>

namespace cellerator::compiler::sema::v1 {

enum class relation_orientation : std::uint8_t { forward = 1, transpose };
enum class relation_mutation_policy : std::uint8_t {
    immutable_structure = 1,
    replace_structure_epoch,
    mutable_active_support
};

struct relation_endpoint_semantics {
    axis_type source{};
    axis_type destination{};
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    semantic_identity logical_edge_identity{};
    semantic_identity exact_support{};
    execution::order_id logical_edge_order{};
    relation_orientation orientation = relation_orientation::forward;
    semantic_identity value_plane{};
    relation_mutation_policy mutation = relation_mutation_policy::immutable_structure;
    std::uint64_t logical_edge_count = 0;
};

compute::operation::v2::typed_relation to_runtime_relation(
    const relation_endpoint_semantics &relation) noexcept;
bool agrees_with_runtime_relation(
    const relation_endpoint_semantics &source,
    const compute::operation::v2::typed_relation &runtime) noexcept;

}  // namespace cellerator::compiler::sema::v1
