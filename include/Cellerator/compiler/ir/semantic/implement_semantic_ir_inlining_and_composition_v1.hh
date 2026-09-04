#pragma once

#include <Cellerator/compiler/ir/semantic/implement_domain_and_axis_ir_types_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Cellerator::compiler::ir::semantic {

struct semantic_inline_node_v1 {
    std::uint64_t identity = 0;
    std::string operation;
    std::vector<std::uint64_t> operands;
    std::vector<std::uint64_t> results;
    semantic_identity_v1 profile{};
    std::uint64_t generation = 0;
    std::vector<std::uint64_t> provenance;
};

struct semantic_inline_graph_v1 {
    std::uint64_t identity = 0;
    std::vector<std::uint64_t> captures;
    std::vector<semantic_inline_node_v1> nodes;
};

struct semantic_capture_substitution_v1 {
    std::uint64_t capture = 0;
    std::uint64_t replacement = 0;
};

enum class semantic_inline_identity_policy_v1 : std::uint8_t {
    preserve = 1,
    rebase,
};

struct semantic_inline_request_v1 {
    std::uint64_t caller_identity = 0;
    std::vector<semantic_capture_substitution_v1> captures;
    semantic_identity_v1 profile_replacement{};
    std::uint64_t minimum_generation = 0;
    semantic_inline_identity_policy_v1 identity_policy =
        semantic_inline_identity_policy_v1::rebase;
    std::uint64_t identity_seed = 0;
};

enum class semantic_inline_status_v1 : std::uint8_t {
    success = 0,
    invalid_graph,
    invalid_request,
    missing_capture,
    duplicate_substitution,
    identity_collision,
};

[[nodiscard]] semantic_inline_status_v1
inline_semantic_graph_v1(
    const semantic_inline_graph_v1& callee,
    const semantic_inline_request_v1& request,
    semantic_inline_graph_v1* result) noexcept;

[[nodiscard]] std::optional<std::vector<std::string>>
canonicalize_semantic_inline_graph_v1(const semantic_inline_graph_v1& graph);

}  // namespace Cellerator::compiler::ir::semantic
