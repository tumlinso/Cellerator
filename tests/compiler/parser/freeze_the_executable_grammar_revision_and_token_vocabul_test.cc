#include <Cellerator/compiler/frontend/parser/freeze_the_executable_grammar_revision_and_token_vocabul_v1.hh>

#include <cassert>
#include <string_view>

using namespace Cellerator::compiler::frontend::parser;

int main() {
    static_assert(executable_language_revision_v1.major == 0);
    static_assert(executable_language_revision_v1.minor == 1);
    assert(token_vocabulary_v1().size() == 24);
    assert(grammar_v1().size() == 10);
    assert(extension_points_v1().size() == 7);
    assert(is_contextual_keyword("domain"));
    assert(is_contextual_keyword("field"));
    assert(!is_contextual_keyword("template"));
    assert(is_cellerator_punctuator("<["));
    assert(precedence_table_v1()[0].rank > precedence_table_v1()[1].rank);

    constexpr std::string_view representative_cxx = R"cpp(
        template<class field> struct domain {
            int given = 0;
            void inspect() { auto require = given; }
        };
    )cpp";
    const auto cxx_report = analyze_cxx_token_collisions(representative_cxx, false);
    assert(cxx_report.contextual_identifier_uses == 6);
    assert(!cxx_report.has_reserved_collision());

    const auto delimiter_report = analyze_cxx_token_collisions("x <[ y ]>", false);
    assert(delimiter_report.has_reserved_collision());
    const auto enabled_report = analyze_cxx_token_collisions("x <[ y ]>", true);
    assert(!enabled_report.has_reserved_collision());
}
