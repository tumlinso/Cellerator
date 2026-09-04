#include <Cellerator/compiler/ir/common/freeze_ceir_common_lexical_conventions_v1.hh>

#include <array>
#include <cctype>

namespace cellerator::compiler::ir {
namespace {

constexpr std::array<std::string_view, 12> contextual_keywords{{
    "attr", "block", "effects", "extension", "native", "profile",
    "region", "source", "semantic", "projection", "schedule", "executable"}};

bool identifier_body(std::string_view spelling) noexcept {
    if (spelling.empty())
        return false;
    const auto first = static_cast<unsigned char>(spelling.front());
    if (!(std::isalpha(first) || spelling.front() == '_'))
        return false;
    for (const char character : spelling) {
        const auto value = static_cast<unsigned char>(character);
        if (!(std::isalnum(value) || character == '_'))
            return false;
    }
    return true;
}

bool dotted_identifier(std::string_view spelling) noexcept {
    if (spelling.empty() || spelling.front() == '.' || spelling.back() == '.')
        return false;
    while (!spelling.empty()) {
        const auto separator = spelling.find('.');
        const auto component = spelling.substr(0, separator);
        if (!identifier_body(component))
            return false;
        if (separator == std::string_view::npos)
            return true;
        spelling.remove_prefix(separator + 1u);
    }
    return false;
}

lexical_token sigiled(
    lexical_kind kind, std::string_view spelling) noexcept {
    const auto payload = spelling.substr(1u);
    return {identifier_body(payload) ? kind : lexical_kind::invalid,
        spelling, payload};
}

} // namespace

bool is_ceir_identifier(std::string_view spelling) noexcept {
    return identifier_body(spelling);
}

bool is_contextual_keyword(std::string_view spelling) noexcept {
    for (const auto keyword : contextual_keywords) {
        if (keyword == spelling)
            return true;
    }
    return false;
}

bool is_valid_extension_name(std::string_view spelling) noexcept {
    return spelling.size() > 2u && spelling.substr(0u, 2u) == "x."
        && dotted_identifier(spelling.substr(2u));
}

bool is_valid_abstraction_level(std::string_view spelling) noexcept {
    constexpr std::array<std::string_view, 5> levels{{
        "ceir.source", "ceir.semantic", "ceir.projection",
        "ceir.schedule", "ceir.executable"}};
    for (const auto level : levels) {
        if (level == spelling)
            return true;
    }
    return false;
}

lexical_token classify_ceir_token(std::string_view spelling) noexcept {
    if (spelling.size() >= 2u && spelling.substr(0u, 2u) == "//")
        return {lexical_kind::comment, spelling, spelling.substr(2u)};
    if (spelling.size() >= 2u && spelling.front() == '`'
        && spelling.back() == '`')
        return {lexical_kind::native_payload, spelling,
            spelling.substr(1u, spelling.size() - 2u)};
    if (is_valid_abstraction_level(spelling))
        return {lexical_kind::abstraction_level, spelling, spelling};
    if (is_valid_extension_name(spelling))
        return {lexical_kind::extension_name, spelling, spelling.substr(2u)};
    if (spelling.empty())
        return {lexical_kind::invalid, spelling, {}};
    switch (spelling.front()) {
    case '@': return sigiled(lexical_kind::persistent_identity, spelling);
    case '%': return sigiled(lexical_kind::ssa_value, spelling);
    case '!': return sigiled(lexical_kind::type_name, spelling);
    case '#': return sigiled(lexical_kind::attribute_name, spelling);
    case '^': return sigiled(lexical_kind::region_label, spelling);
    case '$': return sigiled(lexical_kind::profile_reference, spelling);
    default:
        return {identifier_body(spelling) ? lexical_kind::identifier
                                         : lexical_kind::invalid,
            spelling, spelling};
    }
}

} // namespace cellerator::compiler::ir
