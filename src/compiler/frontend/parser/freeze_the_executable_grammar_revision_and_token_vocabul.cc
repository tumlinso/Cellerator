#include <Cellerator/compiler/frontend/parser/freeze_the_executable_grammar_revision_and_token_vocabul_v1.hh>

#include <cctype>

namespace Cellerator::compiler::frontend::parser {
namespace {

constexpr std::array<token_definition, 24> vocabulary{{
    {token_kind::pragma_cellerator, "cellerator", token_role::preprocessor_directive},
    {token_kind::field_open, "<[", token_role::cellerator_punctuator},
    {token_kind::field_close, "]>", token_role::cellerator_punctuator},
    {token_kind::relation_open, "-[", token_role::cellerator_punctuator},
    {token_kind::relation_close, "]->", token_role::cellerator_punctuator},
    {token_kind::directive_separator, "::", token_role::cellerator_punctuator},
    {token_kind::kw_domain, "domain", token_role::contextual_keyword},
    {token_kind::kw_field, "field", token_role::contextual_keyword},
    {token_kind::kw_given, "given", token_role::contextual_keyword},
    {token_kind::kw_prefer, "prefer", token_role::contextual_keyword},
    {token_kind::kw_require, "require", token_role::contextual_keyword},
    {token_kind::kw_offer, "offer", token_role::contextual_keyword},
    {token_kind::kw_force, "force", token_role::contextual_keyword},
    {token_kind::kw_inspect, "inspect", token_role::contextual_keyword},
    {token_kind::kw_expect, "expect", token_role::contextual_keyword},
    {token_kind::kw_verify, "verify", token_role::contextual_keyword},
    {token_kind::kw_effects, "effects", token_role::contextual_keyword},
    {token_kind::kw_ir, "ir", token_role::contextual_keyword},
    {token_kind::kw_transform, "transform", token_role::contextual_keyword},
    {token_kind::kw_on, "on", token_role::contextual_keyword},
    {token_kind::kw_where, "where", token_role::contextual_keyword},
    {token_kind::kw_matches, "matches", token_role::contextual_keyword},
    {token_kind::kw_in, "in", token_role::contextual_keyword},
    {token_kind::extension_name, "extension", token_role::extension_point},
}};

constexpr std::array<precedence_definition, 3> precedence{{
    {token_kind::relation_open, 3, associativity::left},
    {token_kind::kw_matches, 2, associativity::none},
    {token_kind::kw_in, 2, associativity::none},
}};

constexpr std::array<extension_point, 7> extensions{{
    extension_point::pragma_extension,
    extension_point::offered_kind,
    extension_point::forced_kind,
    extension_point::compiler_semantic_type,
    extension_point::operation_family,
    extension_point::ir_level,
    extension_point::effect_name,
}};

constexpr std::array<grammar_production, 10> productions{{
    grammar_production::cellerator_pragma,
    grammar_production::domain_declaration,
    grammar_production::execution_field,
    grammar_production::planning_directive,
    grammar_production::named_field_definition,
    grammar_production::relation_transfer_expression,
    grammar_production::program_point_directive,
    grammar_production::effect_specifier,
    grammar_production::ir_type,
    grammar_production::transform_definition,
}};

bool identifier_start(char value) noexcept {
    return std::isalpha(static_cast<unsigned char>(value)) != 0 || value == '_';
}

bool identifier_continue(char value) noexcept {
    return std::isalnum(static_cast<unsigned char>(value)) != 0 || value == '_';
}

} // namespace

bool vocabulary_report::has_reserved_collision() const noexcept {
    for (const auto &issue : issues) {
        if (issue.kind == vocabulary_issue_kind::reserved_token_collision)
            return true;
    }
    return false;
}

const std::array<token_definition, 24> &token_vocabulary_v1() noexcept {
    return vocabulary;
}

const std::array<precedence_definition, 3> &precedence_table_v1() noexcept {
    return precedence;
}

const std::array<extension_point, 7> &extension_points_v1() noexcept {
    return extensions;
}

const std::array<grammar_production, 10> &grammar_v1() noexcept {
    return productions;
}

bool is_contextual_keyword(std::string_view spelling) noexcept {
    for (const auto &entry : vocabulary) {
        if (entry.role == token_role::contextual_keyword && entry.spelling == spelling)
            return true;
    }
    return false;
}

bool is_cellerator_punctuator(std::string_view spelling) noexcept {
    for (const auto &entry : vocabulary) {
        if (entry.role == token_role::cellerator_punctuator && entry.spelling == spelling)
            return true;
    }
    return false;
}

vocabulary_report analyze_cxx_token_collisions(std::string_view source,
                                               bool cellerator_enabled) {
    vocabulary_report report{};
    for (std::size_t offset = 0; offset < source.size();) {
        if (identifier_start(source[offset])) {
            const auto begin = offset++;
            while (offset < source.size() && identifier_continue(source[offset]))
                ++offset;
            ++report.cxx_token_count;
            const auto word = source.substr(begin, offset - begin);
            if (is_contextual_keyword(word)) {
                ++report.contextual_identifier_uses;
                report.issues.push_back({
                    vocabulary_issue_kind::contextual_identifier_use,
                    std::string(word), begin});
            }
            continue;
        }
        if (offset + 1 < source.size()) {
            const auto pair = source.substr(offset, 2);
            if ((pair == "<[" || pair == "]>") && !cellerator_enabled) {
                report.issues.push_back({vocabulary_issue_kind::reserved_token_collision,
                                         std::string(pair), offset});
            }
        }
        ++offset;
    }
    return report;
}

} // namespace Cellerator::compiler::frontend::parser
