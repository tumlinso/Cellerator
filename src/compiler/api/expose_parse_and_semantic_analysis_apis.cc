#include <Cellerator/compiler/api/expose_parse_and_semantic_analysis_apis_v1.hh>

#include <sstream>

namespace cellerator::compiler::api::v1 {

parse_snapshot_v1 parse_source_v1(const source_document_v1& source) {
    parse_snapshot_v1 snapshot{source.revision, {}};
    std::istringstream input(source.text);
    for (std::string token; input >> token;) snapshot.tokens.push_back(std::move(token));
    return snapshot;
}

void update_source_v1(source_document_v1& source, std::string text) {
    source.text = std::move(text);
    ++source.revision;
}

bool analyze_semantics_v1(const source_document_v1& source,
    semantic_snapshot_v1& output, analysis_cancelled_v1 cancelled,
    void* user_data) noexcept {
    if (cancelled != nullptr && cancelled(user_data)) return false;
    try {
        const auto parsed = parse_source_v1(source);
        semantic_snapshot_v1 result{source.revision, {}, false, false};
        for (std::size_t index = 0; index + 1 < parsed.tokens.size(); ++index) {
            if (parsed.tokens[index] == "cell" || parsed.tokens[index] == "gene")
                result.declared_symbols.push_back(parsed.tokens[index + 1]);
        }
        output = std::move(result);
        return true;
    } catch (...) { return false; }
}

}  // namespace cellerator::compiler::api::v1
