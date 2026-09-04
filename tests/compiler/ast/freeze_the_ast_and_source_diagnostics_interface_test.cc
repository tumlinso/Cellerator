#include <Cellerator/compiler/ast/freeze_the_ast_and_source_diagnostics_interface_v1.hh>

#include <cassert>
#include <iostream>
#include <type_traits>

using namespace Cellerator::compiler::ast;
using namespace Cellerator::compiler::frontend::source;

int main() {
    static_assert(std::is_trivially_copyable_v<resolved_source_capture_v1>);
    ast_arena_v1 arena{51};
    const auto region = arena.append_region();
    assert(region);
    const auto node = arena.append_node(ast_node_class_v1::expression, {}, *region, 700);
    assert(node);
    auto syntax = std::move(arena).freeze();

    auto semantics = freeze_semantic_nodes_v1(
        syntax, {{*node, ast_semantic_family_v1::operation, 1, 2, 3}});
    assert(semantics);
    const cxx_ast_reference_key_v1 key{cxx_ast_entity_kind_v1::expression, 3, 700, 8, 9};
    auto cxx = freeze_cxx_ast_references_v1(61, 1, 1, {key});
    assert(cxx);
    const auto reference = cxx->reference(key);
    assert(reference);
    auto queries = freeze_ast_query_index_v1(
        51, {{*node, ast_query_kind_v1::operation, 1, 3, 1, 10, 20}});
    assert(queries);
    const compilation_source_identity_v1 source_identity{70, 71};
    auto provenance = freeze_token_provenance_v1(
        {{source_identity,
          {{provenance_frame_kind_v1::token_spelling, {{1, 10}, {1, 20}}, 0},
           {provenance_frame_kind_v1::physical_file, {{1, 10}, {1, 20}}, 0}}}});
    assert(provenance);
    const frontend_diagnostic_v1 diagnostic{
        1, diagnostic_severity_v1::note, diagnostic_category_v1::syntax,
        compiler_phase_v1::semantic_analysis, "resolved capture", {{{1, 10}, {1, 20}}}};
    std::string error;
    auto interface = freeze_ast_source_diagnostics_interface_v1(
        99, syntax, *semantics, *cxx, *queries, *provenance,
        {{source_identity, *node, *reference}}, {diagnostic}, &error);
    assert(interface && error.empty());
    assert(interface->interface_version == 1 && interface->node_count == 1);
    assert(interface->resolved_capture_count == 1 && interface->provenance_record_count == 1);
    assert(interface->diagnostic_count == 1 && interface->incremental_identity_reuse);

    auto stale_reference = *reference;
    ++stale_reference.generation;
    assert(!freeze_ast_source_diagnostics_interface_v1(
        99, syntax, *semantics, *cxx, *queries, *provenance,
        {{source_identity, *node, stale_reference}}, {diagnostic}, &error));

    std::cout << "interface_version=" << interface->interface_version
              << " mapped_captures=" << interface->resolved_capture_count << '\n';
}
