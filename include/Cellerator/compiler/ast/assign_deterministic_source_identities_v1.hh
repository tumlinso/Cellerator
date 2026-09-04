#pragma once

#include <Cellerator/compiler/ast/freeze_ast_node_ownership_and_lifetime_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Cellerator::compiler::ast {

struct compilation_source_identity_v1 {
    std::uint64_t high = 0;
    std::uint64_t low = 0;
};

[[nodiscard]] constexpr bool operator==(compilation_source_identity_v1 left,
                                        compilation_source_identity_v1 right) noexcept {
    return left.high == right.high && left.low == right.low;
}

// This identity is user-owned and suitable for persistent artifacts. It is
// intentionally neither derived from nor interchangeable with an AST handle.
struct persistent_user_identity_v1 {
    std::uint64_t high = 0;
    std::uint64_t low = 0;
};

struct source_identity_input_v1 {
    std::uint64_t semantic_owner = 0;
    std::uint64_t canonical_file = 0;
    std::uint64_t canonical_offset = 0;
    std::uint64_t declaration_identity = 0;
    std::uint32_t language_revision = 0;
};

struct source_identity_record_v1 {
    compilation_source_identity_v1 source_identity{};
    // A transient arena-qualified handle valid only for the associated AST
    // snapshot. It never participates in source identity derivation.
    ast_node_handle_v1 transient_node{};
    std::optional<persistent_user_identity_v1> persistent_identity;
};

[[nodiscard]] compilation_source_identity_v1
derive_source_identity_v1(const source_identity_input_v1& input) noexcept;

[[nodiscard]] std::optional<std::vector<source_identity_record_v1>>
assign_source_identities_v1(const std::vector<source_identity_input_v1>& inputs,
                            const std::vector<ast_node_handle_v1>& transient_nodes,
                            const std::vector<std::optional<persistent_user_identity_v1>>&
                                persistent_identities,
                            std::string* error = nullptr);

} // namespace Cellerator::compiler::ast
