#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::ast {

enum class symbol_kind_v1 : std::uint8_t {
    domain = 1,
    axis,
    relation,
    field,
    profile,
    candidate,
    compiler_pass,
    ir_name,
    imported_program,
};

using symbol_scope_id_v1 = std::uint32_t;
inline constexpr symbol_scope_id_v1 invalid_symbol_scope_v1 = UINT32_MAX;

struct symbol_declaration_v1 {
    std::uint64_t identity = 0;
    symbol_kind_v1 kind = symbol_kind_v1::domain;
    std::string name;
    // Zero denotes a non-overloaded declaration. Otherwise this is a stable
    // signature identity supplied by the owning semantic adapter.
    std::uint64_t signature_identity = 0;
    std::uint64_t source_file_identity = 0;
};

struct symbol_scope_v1 {
    symbol_scope_id_v1 id = invalid_symbol_scope_v1;
    symbol_scope_id_v1 parent = invalid_symbol_scope_v1;
    std::string cxx_namespace;
    std::vector<symbol_scope_id_v1> imports;
    std::vector<symbol_declaration_v1> declarations;
};

enum class symbol_lookup_status_v1 : std::uint8_t {
    resolved = 1,
    overload_set,
    not_found,
    ambiguous,
    invalid_request,
};

struct symbol_lookup_request_v1 {
    symbol_scope_id_v1 scope = invalid_symbol_scope_v1;
    std::string_view name;
    std::optional<symbol_kind_v1> kind;
    std::optional<std::uint64_t> signature_identity;
    // Qualified lookup searches exactly scope; unqualified lookup follows
    // lexical parents and their explicitly imported program scopes.
    bool qualified = false;
};

struct symbol_lookup_result_v1 {
    symbol_lookup_status_v1 status = symbol_lookup_status_v1::invalid_request;
    std::vector<const symbol_declaration_v1*> candidates;
    symbol_scope_id_v1 declaring_scope = invalid_symbol_scope_v1;
};

class symbol_table_v1 {
public:
    [[nodiscard]] const symbol_scope_v1* scope(symbol_scope_id_v1 id) const noexcept;
    [[nodiscard]] std::size_t scope_count() const noexcept;
    [[nodiscard]] symbol_lookup_result_v1 lookup(symbol_lookup_request_v1 request) const;

private:
    std::vector<symbol_scope_v1> scopes_;
    friend std::optional<symbol_table_v1>
    freeze_symbol_table_v1(std::vector<symbol_scope_v1>, std::string*);
};

[[nodiscard]] std::optional<symbol_table_v1>
freeze_symbol_table_v1(std::vector<symbol_scope_v1> scopes, std::string* error = nullptr);

} // namespace Cellerator::compiler::ast
