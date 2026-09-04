#pragma once

#include <Cellerator/compiler/ast/create_structured_frontend_diagnostic_records_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::ast {

enum class source_fix_kind_v1 : std::uint8_t {
    missing_pragma = 1,
    malformed_field_delimiter,
    relation_endpoint_mismatch,
    absent_profile_binding,
    effect_contract_omission,
    deprecated_syntax,
};

struct source_fix_request_v1 {
    source_fix_kind_v1 kind = source_fix_kind_v1::missing_pragma;
    frontend::source::source_span_v1 source{};
    std::string replacement_hint;
    bool physical_source = true;
    bool macro_expanded = false;
    bool promises_recompile = true;
};

struct source_fix_v1 {
    source_fix_kind_v1 kind = source_fix_kind_v1::missing_pragma;
    diagnostic_fix_it_v1 edit;
    bool promises_recompile = true;
};

using repaired_source_validator_v1 = bool (*)(std::string_view, void*) noexcept;

[[nodiscard]] std::optional<source_fix_v1>
generate_source_fix_v1(const source_fix_request_v1& request, std::uint64_t source_size,
                       std::string* error = nullptr);

// Edits are applied only to their physical file, in reverse byte order. When
// any edit promises recompilation, the caller's actual frontend validator must
// accept the repaired buffer before it is returned.
[[nodiscard]] std::optional<std::string>
apply_source_fixes_v1(std::string_view source,
                      frontend::source::source_space_id_v1 physical_file,
                      std::vector<source_fix_v1> fixes,
                      repaired_source_validator_v1 validator,
                      void* validator_context,
                      std::string* error = nullptr);

} // namespace Cellerator::compiler::ast
