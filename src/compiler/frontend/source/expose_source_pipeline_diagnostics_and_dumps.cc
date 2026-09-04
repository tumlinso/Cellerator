#include <Cellerator/compiler/frontend/source/expose_source_pipeline_diagnostics_and_dumps_v1.hh>

#include <sstream>

namespace Cellerator::compiler::frontend::source {

std::string render_source_pipeline_dump_v1(const source_dump_request_v1& request,
                                           const source_dump_inputs_v1& inputs) {
    if (!request.tokens && !request.activation_map && !request.shadow_source && !request.source_map) return {};
    auto path = inputs.path;
    if (!request.path_prefix.empty() && path.rfind(request.path_prefix, 0) == 0)
        path.replace(0, request.path_prefix.size(), request.remapped_prefix);
    std::ostringstream out;
    out << "source " << path << '\n';
    if (request.tokens && inputs.tokens) {
        out << "tokens\n";
        for (const auto& token : inputs.tokens->tokens)
            out << token.span.begin.byte_offset << ':' << token.span.end.byte_offset << ' '
                << (token.dialect_active ? "active " : "ordinary ") << token.spelling << '\n';
    }
    if (request.activation_map && inputs.tokens) {
        out << "activation";
        for (const auto& token : inputs.tokens->tokens) out << ' ' << (token.dialect_active ? '1' : '0');
        out << '\n';
    }
    if (request.shadow_source) out << "shadow\n" << inputs.shadow_source << '\n';
    if (request.source_map) out << "source-map\n" << inputs.source_map << '\n';
    return out.str();
}

} // namespace Cellerator::compiler::frontend::source
