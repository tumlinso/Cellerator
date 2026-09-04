#include <Cellerator/compiler/ast/create_deterministic_ast_dump_and_snapshot_formats_v1.hh>

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace Cellerator::compiler::ast {
namespace {

std::string escape_json(std::string_view input) {
    std::string result;
    for (const char byte : input) {
        if (byte == '"' || byte == '\\') result.push_back('\\');
        if (byte == '\n') result += "\\n";
        else result.push_back(byte);
    }
    return result;
}

std::string_view family_name(ast_semantic_family_v1 family) {
    switch (family) {
    case ast_semantic_family_v1::declaration: return "declaration";
    case ast_semantic_family_v1::execution_field: return "execution_field";
    case ast_semantic_family_v1::operation: return "operation";
    case ast_semantic_family_v1::policy_directive: return "policy_directive";
    case ast_semantic_family_v1::effect_contract: return "effect_contract";
    case ast_semantic_family_v1::profile_binding: return "profile_binding";
    case ast_semantic_family_v1::inline_ir: return "inline_ir";
    case ast_semantic_family_v1::reflection: return "reflection";
    case ast_semantic_family_v1::compiler_pass: return "compiler_pass";
    case ast_semantic_family_v1::native_fragment: return "native_fragment";
    default: return "invalid";
    }
}

} // namespace

std::optional<ast_dump_document_v1>
canonicalize_ast_dump_v1(ast_dump_document_v1 document, std::string* error) {
    const auto fail = [&](std::string message) -> std::optional<ast_dump_document_v1> {
        if (error) *error = std::move(message);
        return std::nullopt;
    };
    if (document.schema_version != ast_dump_schema_version_v1 || document.language_revision == 0)
        return fail("unsupported AST dump version");
    std::sort(document.nodes.begin(), document.nodes.end(), [](const auto& left, const auto& right) {
        return left.semantic_identity < right.semantic_identity;
    });
    std::unordered_set<std::uint64_t> identities;
    for (const auto& node : document.nodes) {
        if (node.semantic_identity == 0 || family_name(node.family) == "invalid" ||
            node.name.empty() || (node.source_identity.high | node.source_identity.low) == 0 ||
            !identities.insert(node.semantic_identity).second)
            return fail("AST dump contains an invalid or duplicate node");
    }
    for (const auto& node : document.nodes)
        if (node.parent_semantic_identity != 0 &&
            !identities.count(node.parent_semantic_identity))
            return fail("AST dump parent identity is missing");
    if (error) error->clear();
    return document;
}

std::string render_ast_text_v1(const ast_dump_document_v1& document) {
    std::ostringstream out;
    out << "cellerator-ast-dump-v" << document.schema_version
        << " language-revision=" << document.language_revision << '\n';
    for (const auto& node : document.nodes) {
        out << "node " << node.semantic_identity << " parent=" << node.parent_semantic_identity
            << " family=" << family_name(node.family) << " form=" << node.form
            << " source=" << node.source_identity.high << ':' << node.source_identity.low
            << " name=\"" << node.name << "\"\n";
    }
    return out.str();
}

std::string render_ast_json_v1(const ast_dump_document_v1& document) {
    std::ostringstream out;
    out << "{\"schema\":\"cellerator.ast.snapshot\",\"version\":"
        << document.schema_version << ",\"languageRevision\":" << document.language_revision
        << ",\"nodes\":[";
    for (std::size_t index = 0; index < document.nodes.size(); ++index) {
        const auto& node = document.nodes[index];
        if (index) out << ',';
        out << "{\"id\":" << node.semantic_identity << ",\"parent\":"
            << node.parent_semantic_identity << ",\"family\":\"" << family_name(node.family)
            << "\",\"form\":" << node.form << ",\"source\":{\"high\":"
            << node.source_identity.high << ",\"low\":" << node.source_identity.low
            << "},\"name\":\"" << escape_json(node.name) << "\"}";
    }
    out << "]}";
    return out.str();
}

} // namespace Cellerator::compiler::ast
