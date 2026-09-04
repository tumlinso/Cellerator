#pragma once

#include <Cellerator/compiler/frontend/parser/parse_compiler_semantic_declarations_v1.hh>

#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

enum class planning_authority_v1 : unsigned char {
    given_fact = 1,
    preference = 2,
    offered_alternative = 3,
    forced_selection = 4,
    hard_requirement = 5
};

enum class planning_subject_v1 {
    profile,
    reuse,
    persistence,
    budget,
    objective,
    target_class,
    candidate,
    realization,
    other
};

struct planning_directive_v1 {
    planning_authority_v1 authority = planning_authority_v1::given_fact;
    planning_subject_v1 subject = planning_subject_v1::other;
    std::string operation_scope;
    std::string expression;
    parser_source_range_v1 range{};
};

struct planning_parse_v1 {
    std::vector<planning_directive_v1> directives;
    std::vector<declaration_diagnostic_v1> diagnostics;
    [[nodiscard]] bool accepted() const noexcept { return diagnostics.empty(); }
};

[[nodiscard]] planning_parse_v1 parse_planning_directives_v1(
    std::string_view prologue);

} // namespace Cellerator::compiler::frontend::parser
