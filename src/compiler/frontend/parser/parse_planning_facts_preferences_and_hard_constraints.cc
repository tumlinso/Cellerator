#include <Cellerator/compiler/frontend/parser/parse_planning_facts_preferences_and_hard_constraints_v1.hh>

#include <cctype>
#include <optional>

namespace Cellerator::compiler::frontend::parser {
namespace {

std::string trim_copy(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
        value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
        value.remove_suffix(1);
    return std::string(value);
}

std::optional<planning_authority_v1> authority_of(std::string_view statement,
                                                  std::size_t &length) {
    constexpr std::pair<std::string_view, planning_authority_v1> entries[] = {
        {"given", planning_authority_v1::given_fact},
        {"prefer", planning_authority_v1::preference},
        {"offer", planning_authority_v1::offered_alternative},
        {"force", planning_authority_v1::forced_selection},
        {"require", planning_authority_v1::hard_requirement},
    };
    for (const auto &[name, authority] : entries) {
        if (statement.compare(0, name.size(), name) == 0) {
            length = name.size();
            return authority;
        }
    }
    return std::nullopt;
}

planning_subject_v1 subject_of(std::string_view expression) {
    constexpr std::pair<std::string_view, planning_subject_v1> entries[] = {
        {"profile", planning_subject_v1::profile},
        {"reuse", planning_subject_v1::reuse},
        {"persist", planning_subject_v1::persistence},
        {"budget", planning_subject_v1::budget},
        {"latency", planning_subject_v1::objective},
        {"throughput", planning_subject_v1::objective},
        {"target", planning_subject_v1::target_class},
        {"candidate", planning_subject_v1::candidate},
        {"realization", planning_subject_v1::realization},
    };
    for (const auto &[needle, subject] : entries)
        if (expression.find(needle) != std::string_view::npos)
            return subject;
    return planning_subject_v1::other;
}

std::string selected_name(std::string_view expression, std::string_view category) {
    const auto at = expression.find(category);
    if (at == std::string_view::npos)
        return {};
    auto begin = at + category.size();
    while (begin < expression.size()
           && std::isspace(static_cast<unsigned char>(expression[begin])))
        ++begin;
    auto end = begin;
    while (end < expression.size()
           && (std::isalnum(static_cast<unsigned char>(expression[end]))
               || expression[end] == '_' || expression[end] == ':'))
        ++end;
    return std::string(expression.substr(begin, end - begin));
}

} // namespace

planning_parse_v1 parse_planning_directives_v1(std::string_view prologue) {
    planning_parse_v1 result;
    std::size_t begin = 0;
    while (begin < prologue.size()) {
        const auto semicolon = prologue.find(';', begin);
        const auto end = semicolon == std::string_view::npos ? prologue.size() : semicolon;
        auto statement = std::string_view(prologue).substr(begin, end - begin);
        while (!statement.empty()
               && std::isspace(static_cast<unsigned char>(statement.front()))) {
            statement.remove_prefix(1);
            ++begin;
        }
        if (!statement.empty()) {
            std::size_t keyword_length = 0;
            const auto authority = authority_of(statement, keyword_length);
            if (!authority) {
                result.diagnostics.push_back({"unknown planning directive", {begin, end}});
            } else {
                auto expression = trim_copy(statement.substr(keyword_length));
                std::string operation_scope;
                if (expression.compare(0, 10, "operation(") == 0) {
                    const auto close = expression.find(')');
                    const auto colon = close == std::string::npos
                        ? std::string::npos : expression.find(':', close);
                    if (colon != std::string::npos) {
                        operation_scope = expression.substr(10, close - 10);
                        expression = trim_copy(std::string_view(expression).substr(colon + 1));
                    }
                }
                result.directives.push_back({*authority, subject_of(expression),
                                             std::move(operation_scope), std::move(expression),
                                             {begin, semicolon == std::string_view::npos ? end : end + 1}});
            }
        }
        if (semicolon == std::string_view::npos)
            break;
        begin = semicolon + 1;
    }

    std::string forced_candidate;
    std::string forced_realization;
    for (const auto &directive : result.directives) {
        if (directive.authority == planning_authority_v1::forced_selection) {
            const auto category = directive.subject == planning_subject_v1::candidate
                ? std::string_view("candidate") : std::string_view("realization");
            auto &selected = directive.subject == planning_subject_v1::candidate
                ? forced_candidate : forced_realization;
            const auto name = selected_name(directive.expression, category);
            if (!selected.empty() && !name.empty() && selected != name)
                result.diagnostics.push_back({"conflicting forced selections", directive.range});
            else if (!name.empty())
                selected = name;
        }
        if (directive.authority == planning_authority_v1::hard_requirement
            && directive.expression.find("exclude candidate " + forced_candidate)
                != std::string::npos && !forced_candidate.empty())
            result.diagnostics.push_back({"hard requirement excludes forced candidate",
                                          directive.range});
    }
    return result;
}

} // namespace Cellerator::compiler::frontend::parser
