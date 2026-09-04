#pragma once

#include <cstdint>
#include <string_view>

namespace cellerator::compiler::ir {

enum class lexical_kind : std::uint8_t {
    identifier,
    persistent_identity,
    ssa_value,
    type_name,
    attribute_name,
    region_label,
    profile_reference,
    native_payload,
    extension_name,
    abstraction_level,
    comment,
    invalid
};

struct lexical_token {
    lexical_kind kind{lexical_kind::invalid};
    std::string_view spelling{};
    std::string_view payload{};
};

// Sigils are disjoint from C++ identifiers. Words used by the textual CEIR
// remain contextual so embedding CEIR does not reserve additional C++ words.
bool is_ceir_identifier(std::string_view spelling) noexcept;
bool is_contextual_keyword(std::string_view spelling) noexcept;
lexical_token classify_ceir_token(std::string_view spelling) noexcept;
bool is_valid_extension_name(std::string_view spelling) noexcept;
bool is_valid_abstraction_level(std::string_view spelling) noexcept;

} // namespace cellerator::compiler::ir
