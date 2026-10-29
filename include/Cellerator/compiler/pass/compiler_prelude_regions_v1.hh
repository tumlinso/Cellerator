#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::pass::v1 {

enum class prelude_declaration_kind_v1 : std::uint8_t {
    transform = 0,
    extension_schema,
    pipeline_configuration,
    ordinary,
};

struct prelude_declaration_v1 {
    prelude_declaration_kind_v1 kind = prelude_declaration_kind_v1::ordinary;
    std::string name;
    std::vector<std::string> references;
};

struct source_region_v1 {
    bool compiler_prelude = false;
    std::vector<prelude_declaration_v1> declarations;
};

enum class prelude_resolution_status_v1 : std::uint8_t {
    success = 0,
    invalid_declaration,
    duplicate_prelude_symbol,
    unresolved_prelude_reference,
};

struct prelude_resolution_receipt_v1 {
    prelude_resolution_status_v1 status = prelude_resolution_status_v1::success;
    std::vector<std::string> prelude_symbols;
    std::vector<prelude_declaration_v1> ordinary_declarations;
    std::string diagnostic;
};

[[nodiscard]] prelude_resolution_receipt_v1 resolve_compiler_preludes_v1(
    const std::vector<source_region_v1>& regions);

}  // namespace cellerator::compiler::pass::v1
