#include <Cellerator/compiler/frontend/parser/deliver_full_grammar_conformance_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_non_relation_operation_families_v1.hh>

#include <cassert>
#include <fstream>
#include <iterator>
#include <string>

using namespace Cellerator::compiler::frontend::parser;

namespace {
std::string read_file(const char *path) {
    std::ifstream input(path);
    if (!input.good()) {
        input.close();
        input.open(std::string("../") + path);
    }
    assert(input.good());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
} // namespace

int main() {
    const auto specification = audit_language_examples_v1(
        "cellerator-language-specification.md",
        read_file("docs/language/cellerator-language-specification.md"));
    const auto guide = audit_language_examples_v1(
        "cellerator-programming-guide.md",
        read_file("docs/language/cellerator-programming-guide.md"));

    assert(specification.fully_disposed());
    assert(guide.fully_disposed());
    assert(specification.examples.size() > 100);
    assert(guide.examples.size() > 100);
    assert(specification.unresolved_examples == 0);
    assert(guide.unresolved_examples == 0);
    assert(specification.unresolved_operation_kinds == 0);
    assert(guide.unresolved_operation_kinds == 0);

    std::size_t parsed = 0;
    std::size_t delegated = 0;
    std::size_t references = 0;
    for (const auto &example : specification.examples) {
        parsed += example.disposition == grammar_example_disposition_v1::parsed_cellerator;
        delegated += example.disposition == grammar_example_disposition_v1::delegated_cxx;
        references += example.disposition == grammar_example_disposition_v1::grammar_reference;
    }
    assert(parsed != 0);
    assert(delegated != 0);
    assert(references != 0);

    const auto &matrix = grammar_coverage_matrix_v1();
    assert(matrix.size() == 10 + operation_family_table_v1().size());
    std::size_t operation_entries = 0;
    std::size_t intentional_revisions = 0;
    for (const auto &entry : matrix) {
        operation_entries += entry.category == "operation";
        intentional_revisions +=
            entry.status == grammar_coverage_status_v1::intentional_revision;
    }
    assert(operation_entries == operation_family_table_v1().size());
    assert(intentional_revisions == 2);

    const auto synthetic_revision = audit_language_examples_v1("revision.md", R"(
```cpp
ceir<semantic> captures(x) results(y) validation(checked) { y = x }
```
)" );
    assert(synthetic_revision.fully_disposed());
    assert(synthetic_revision.examples.front().disposition
           == grammar_example_disposition_v1::intentional_revision);

    const auto malformed = audit_language_examples_v1("bad.md", "```cpp\ndomain gene;\n");
    assert(!malformed.fully_disposed());
    assert(malformed.unterminated_fence);
}
