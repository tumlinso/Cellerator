#pragma once

#include <Cellerator/compiler/ir/semantic/implement_domain_and_axis_ir_types_v1.hh>
#include <Cellerator/compute/operation/relation_algebra_v2/composition.hh>

#include <cstdint>
#include <optional>
#include <vector>

namespace Cellerator::compiler::ir::semantic {

enum class semantic_graph_operation_kind_v1 : std::uint8_t {
    relation_bundle = 1,
    relation_chain,
    paired_moments,
    incidence_pool,
    incidence_broadcast,
    typed_exchange,
};

enum semantic_graph_effect_v1 : std::uint32_t {
    graph_reads_inputs_v1 = 1u << 0,
    graph_writes_outputs_v1 = 1u << 1,
    graph_advances_generation_v1 = 1u << 2,
    graph_communicates_v1 = 1u << 3,
};

struct semantic_graph_node_v1 {
    semantic_identity_v1 identity{};
    semantic_graph_operation_kind_v1 kind = semantic_graph_operation_kind_v1::relation_chain;
    std::vector<semantic_identity_v1> input_axes;
    std::vector<semantic_identity_v1> output_axes;
    std::vector<semantic_identity_v1> intermediate_axes;
    std::uint32_t effects = graph_reads_inputs_v1 | graph_writes_outputs_v1 |
        graph_advances_generation_v1;
};

struct semantic_graph_dependency_v1 {
    std::uint64_t producer = 0;
    std::uint64_t consumer = 0;
    semantic_identity_v1 exchanged_axis{};
};

struct semantic_operation_graph_v1 {
    semantic_identity_v1 identity{};
    std::vector<semantic_graph_node_v1> nodes;
    std::vector<semantic_graph_dependency_v1> dependencies;
};

enum class semantic_graph_validation_code_v1 : std::uint8_t {
    success = 0,
    invalid_identity,
    invalid_node,
    invalid_axis,
    invalid_effects,
    invalid_dependency,
    axis_mismatch,
    cycle,
};

[[nodiscard]] semantic_graph_validation_code_v1
validate_semantic_operation_graph_v1(const semantic_operation_graph_v1& graph) noexcept;

[[nodiscard]] std::optional<semantic_operation_graph_v1>
round_trip_operation_portfolio_graph_v1(const semantic_operation_graph_v1& graph) noexcept;

[[nodiscard]] cellerator::compute::operation::v2::composition_kind
lower_semantic_graph_kind_v1(semantic_graph_operation_kind_v1 kind) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
