#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

struct language_revision {
    std::uint16_t major = 0;
    std::uint16_t minor = 1;
};

inline constexpr language_revision executable_language_revision_v1{};

enum class token_kind : std::uint16_t {
    end_of_file,
    cxx_token,
    identifier,
    revision,
    pragma_cellerator,
    field_open,
    field_close,
    relation_open,
    relation_close,
    directive_separator,
    kw_domain,
    kw_field,
    kw_given,
    kw_prefer,
    kw_require,
    kw_offer,
    kw_force,
    kw_inspect,
    kw_expect,
    kw_verify,
    kw_effects,
    kw_ir,
    kw_transform,
    kw_on,
    kw_where,
    kw_matches,
    kw_in,
    extension_name
};

enum class token_role : std::uint8_t {
    delegated_cxx,
    contextual_keyword,
    cellerator_punctuator,
    preprocessor_directive,
    extension_point
};

struct token_definition {
    token_kind kind;
    std::string_view spelling;
    token_role role;
};

enum class associativity : std::uint8_t { none, left, right };

struct precedence_definition {
    token_kind kind;
    std::uint8_t rank;
    associativity association;
};

enum class extension_point : std::uint8_t {
    pragma_extension,
    offered_kind,
    forced_kind,
    compiler_semantic_type,
    operation_family,
    ir_level,
    effect_name
};

enum class grammar_production : std::uint8_t {
    cellerator_pragma,
    domain_declaration,
    execution_field,
    planning_directive,
    named_field_definition,
    relation_transfer_expression,
    program_point_directive,
    effect_specifier,
    ir_type,
    transform_definition
};

enum class vocabulary_issue_kind : std::uint8_t {
    contextual_identifier_use,
    reserved_token_collision,
    malformed_field_delimiter,
    malformed_relation_delimiter
};

struct vocabulary_issue {
    vocabulary_issue_kind kind;
    std::string spelling;
    std::size_t offset = 0;
};

struct vocabulary_report {
    std::size_t cxx_token_count = 0;
    std::size_t contextual_identifier_uses = 0;
    std::vector<vocabulary_issue> issues;

    bool has_reserved_collision() const noexcept;
};

const std::array<token_definition, 24> &token_vocabulary_v1() noexcept;
const std::array<precedence_definition, 3> &precedence_table_v1() noexcept;
const std::array<extension_point, 7> &extension_points_v1() noexcept;
const std::array<grammar_production, 10> &grammar_v1() noexcept;
bool is_contextual_keyword(std::string_view spelling) noexcept;
bool is_cellerator_punctuator(std::string_view spelling) noexcept;
vocabulary_report analyze_cxx_token_collisions(std::string_view source,
                                               bool cellerator_enabled);

} // namespace Cellerator::compiler::frontend::parser
