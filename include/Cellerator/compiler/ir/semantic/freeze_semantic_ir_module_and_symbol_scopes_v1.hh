#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::ir::semantic {

inline constexpr std::uint32_t semantic_scope_schema_version_v1 = 1;
using semantic_scope_id_v1 = std::uint32_t;
inline constexpr semantic_scope_id_v1 invalid_semantic_scope_id_v1 = UINT32_MAX;

enum class semantic_scope_kind_v1 : std::uint8_t {
    program = 1,
    module,
    translation_unit,
    function,
    named_field,
    anonymous_field,
};

enum class semantic_symbol_kind_v1 : std::uint8_t {
    domain = 1,
    axis,
    relation,
    state,
    profile,
    function,
    named_field,
    semantic_object,
};

struct semantic_scope_definition_v1 {
    semantic_scope_id_v1 id = invalid_semantic_scope_id_v1;
    semantic_scope_id_v1 parent = invalid_semantic_scope_id_v1;
    semantic_scope_kind_v1 kind = semantic_scope_kind_v1::program;
    std::uint64_t stable_identity = 0;
    std::string name;
};

struct semantic_symbol_definition_v1 {
    std::uint64_t stable_identity = 0;
    semantic_symbol_kind_v1 kind = semantic_symbol_kind_v1::semantic_object;
    semantic_scope_id_v1 owner_scope = invalid_semantic_scope_id_v1;
    std::string name;
};

// Cross-translation-unit visibility is closed by default. Each imported symbol
// requires one exact exporter/importer authorization; module membership alone
// never grants access.
struct semantic_symbol_export_authorization_v1 {
    std::uint64_t symbol_identity = 0;
    semantic_scope_id_v1 exporting_translation_unit = invalid_semantic_scope_id_v1;
    semantic_scope_id_v1 importing_translation_unit = invalid_semantic_scope_id_v1;
};

struct imported_semantic_symbol_v1 {
    std::uint64_t symbol_identity = 0;
    semantic_scope_id_v1 importing_scope = invalid_semantic_scope_id_v1;
};

struct semantic_scope_module_definition_v1 {
    std::uint32_t schema_version = semantic_scope_schema_version_v1;
    std::vector<semantic_scope_definition_v1> scopes;
    std::vector<semantic_symbol_definition_v1> symbols;
    std::vector<semantic_symbol_export_authorization_v1> export_authorizations;
    std::vector<imported_semantic_symbol_v1> imports;
};

enum class semantic_scope_diagnostic_code_v1 : std::uint8_t {
    success = 0,
    unsupported_schema,
    missing_program_scope,
    invalid_scope_identity,
    invalid_scope_parent,
    invalid_scope_nesting,
    duplicate_scope,
    invalid_symbol,
    duplicate_symbol,
    invalid_export_authorization,
    duplicate_export_authorization,
    unauthorized_import,
    duplicate_import,
};

struct semantic_scope_diagnostic_v1 {
    semantic_scope_diagnostic_code_v1 code = semantic_scope_diagnostic_code_v1::success;
    semantic_scope_id_v1 scope = invalid_semantic_scope_id_v1;
    std::uint64_t symbol_identity = 0;
    std::string message;

    [[nodiscard]] explicit operator bool() const noexcept {
        return code == semantic_scope_diagnostic_code_v1::success;
    }
};

class frozen_semantic_scope_module_v1 {
public:
    [[nodiscard]] const semantic_scope_definition_v1* scope(
        semantic_scope_id_v1 id) const noexcept;
    [[nodiscard]] const semantic_symbol_definition_v1* symbol(
        std::uint64_t stable_identity) const noexcept;
    [[nodiscard]] const semantic_symbol_definition_v1* resolve_local(
        semantic_scope_id_v1 scope, std::string_view name) const noexcept;
    [[nodiscard]] const semantic_symbol_definition_v1* resolve_imported(
        semantic_scope_id_v1 scope, std::string_view name) const noexcept;
    [[nodiscard]] semantic_scope_id_v1 owning_translation_unit(
        semantic_scope_id_v1 scope) const noexcept;
    [[nodiscard]] const std::vector<semantic_scope_definition_v1>& scopes() const noexcept;
    [[nodiscard]] const std::vector<semantic_symbol_definition_v1>& symbols() const noexcept;

private:
    std::vector<semantic_scope_definition_v1> scopes_;
    std::vector<semantic_symbol_definition_v1> symbols_;
    std::vector<semantic_symbol_export_authorization_v1> export_authorizations_;
    std::vector<imported_semantic_symbol_v1> imports_;

    friend std::optional<frozen_semantic_scope_module_v1>
    freeze_semantic_ir_module_and_symbol_scopes_v1(
        semantic_scope_module_definition_v1, semantic_scope_diagnostic_v1*) noexcept;
};

[[nodiscard]] std::optional<frozen_semantic_scope_module_v1>
freeze_semantic_ir_module_and_symbol_scopes_v1(
    semantic_scope_module_definition_v1 definition,
    semantic_scope_diagnostic_v1* diagnostic = nullptr) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
