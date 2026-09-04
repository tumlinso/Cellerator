#include <Cellerator/compiler/ast/freeze_the_ast_and_source_diagnostics_interface_v1.hh>

namespace Cellerator::compiler::ast {

std::optional<ast_source_diagnostics_interface_v1>
freeze_ast_source_diagnostics_interface_v1(
    std::uint64_t original_source_hash,
    const ast_snapshot_v1& syntax,
    const ast_semantic_table_v1& semantics,
    const cxx_ast_reference_table_v1& cxx_references,
    const ast_query_index_v1& queries,
    const token_provenance_sidecar_v1& provenance,
    const std::vector<resolved_source_capture_v1>& captures,
    const std::vector<frontend_diagnostic_v1>& diagnostics,
    std::string* error) {
    const auto fail = [&](std::string message)
        -> std::optional<ast_source_diagnostics_interface_v1> {
        if (error) *error = std::move(message);
        return std::nullopt;
    };
    if (original_source_hash == 0 || syntax.arena_id() == 0 ||
        semantics.arena_id() != syntax.arena_id() ||
        semantics.size() != syntax.node_count() || queries.size() != syntax.node_count())
        return fail("AST interface inputs do not describe one complete source snapshot");
    for (const auto& capture : captures) {
        if (!syntax.node(capture.parse_node) || !queries.find(capture.parse_node) ||
            !cxx_references.resolve(capture.resolved_cxx) ||
            !provenance.find(capture.original_source))
            return fail("resolved source capture is not traceable across every interface layer");
    }
    for (const auto& diagnostic : diagnostics)
        if (!validate_frontend_diagnostic_v1(diagnostic, error)) return std::nullopt;
    if (error) error->clear();
    return ast_source_diagnostics_interface_v1{
        ast_source_diagnostics_interface_version_v1, syntax.arena_id(), original_source_hash,
        syntax.node_count(), captures.size(), provenance.size(), diagnostics.size(), true};
}

} // namespace Cellerator::compiler::ast
