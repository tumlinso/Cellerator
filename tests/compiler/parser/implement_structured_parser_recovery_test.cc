#include <Cellerator/compiler/frontend/parser/implement_structured_parser_recovery_v1.hh>

#include <array>
#include <cassert>
#include <string>

using namespace Cellerator::compiler::frontend::parser;

int main() {
    struct malformed_case {
        std::string source;
        parser_recovery_boundary_v1 boundary;
        std::string resume_text;
    };
    const std::array<malformed_case, 5> corpus{{
        {"<[ input -> broken junk ]> domain gene;", parser_recovery_boundary_v1::field,
         " domain gene;"},
        {"domain <missing-name>; axis gene_axis;", parser_recovery_boundary_v1::declaration,
         " axis gene_axis;"},
        {"x -[r]-> ;\ny -[s]-> z;", parser_recovery_boundary_v1::operation,
         "\ny -[s]-> z;"},
        {"state<gene, ordered_by<); relation<gene, cell>;",
         parser_recovery_boundary_v1::qualifier, " ordered_by<); relation<gene, cell>;"},
        {"ceir<semantic> { broken } ceir<planning> {}",
         parser_recovery_boundary_v1::inline_ir, " ceir<planning> {}"}
    }};

    for (const auto &entry : corpus) {
        const auto recovered = recover_parser_v1(entry.source, 0, entry.boundary, 2);
        assert(recovered.diagnostic_count() <= 3);
        assert(!recovered.primary.message.empty());
        assert(recovered.notes.size() <= 2);
        assert(entry.source.substr(recovered.resume_offset) == entry.resume_text);
        assert(recovered.resume_offset > 0);
    }

    const auto bounded = recover_parser_v1("broken without boundary", 0,
                                           parser_recovery_boundary_v1::declaration, 1);
    assert(bounded.diagnostic_count() == 2);
    assert(bounded.resume_offset == 23);

    const auto silent_notes = recover_parser_v1("bad; next;", 0,
                                                parser_recovery_boundary_v1::declaration, 0);
    assert(silent_notes.diagnostic_count() == 1);
    assert(silent_notes.resume_offset == 4);
}
