#include <Cellerator/compiler/ast/create_structured_frontend_diagnostic_records_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ast;
using namespace Cellerator::compiler::frontend::source;

int main() {
    const source_span_v1 primary{{4, 10}, {4, 15}};
    frontend_diagnostic_v1 diagnostic{
        2048, diagnostic_severity_v1::error, diagnostic_category_v1::biological_identity,
        compiler_phase_v1::semantic_analysis, "axis identity does not match relation",
        {primary, {{4, 30}, {4, 35}}},
        {{"relation declared here", source_span_v1{{2, 2}, {2, 8}}}},
        {{primary, "genes"}}, {71, 72}};
    std::string error;
    assert(validate_frontend_diagnostic_v1(diagnostic, &error));
    const auto bytes = serialize_frontend_diagnostic_v1(diagnostic);
    const auto round_trip = deserialize_frontend_diagnostic_v1(bytes, &error);
    assert(round_trip && error.empty());
    assert(round_trip->stable_id == diagnostic.stable_id);
    assert(round_trip->message == diagnostic.message);
    assert(round_trip->source_ranges.size() == 2);
    assert(round_trip->notes[0].source->begin.space == 2);
    assert(round_trip->fix_its[0].replacement == "genes");
    assert(round_trip->related_symbols == diagnostic.related_symbols);

    const auto terminal = render_terminal_diagnostic_v1(*round_trip);
    const auto lsp = render_lsp_diagnostic_v1(*round_trip);
    assert(terminal.find("CE2048") != std::string::npos);
    assert(lsp.find("CE2048") != std::string::npos);
    assert(terminal.find(diagnostic.message) != std::string::npos);
    assert(lsp.find(diagnostic.message) != std::string::npos);
    assert(!deserialize_frontend_diagnostic_v1(bytes.substr(0, bytes.size() - 1), &error));

    std::cout << "serialized_bytes=" << bytes.size()
              << " terminal_bytes=" << terminal.size() << " lsp_bytes=" << lsp.size() << '\n';
}
