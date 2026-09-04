#include <Cellerator/compiler/ir/semantic/implement_semantic_ir_inlining_and_composition_v1.hh>

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace Cellerator::compiler::ir::semantic {

semantic_inline_status_v1 inline_semantic_graph_v1(
    const semantic_inline_graph_v1& callee,
    const semantic_inline_request_v1& request,
    semantic_inline_graph_v1* result) noexcept {
    if (result == nullptr || callee.identity == 0 ||
        std::any_of(callee.nodes.begin(), callee.nodes.end(), [](const auto& node) {
            return node.identity == 0 || node.operation.empty() || node.results.empty();
        })) return semantic_inline_status_v1::invalid_graph;
    if (request.caller_identity == 0 || request.minimum_generation == 0 ||
        !request.profile_replacement.valid() ||
        (request.identity_policy == semantic_inline_identity_policy_v1::rebase &&
         request.identity_seed == 0)) return semantic_inline_status_v1::invalid_request;
    std::unordered_map<std::uint64_t, std::uint64_t> substitutions;
    for (const auto& substitution : request.captures) {
        if (substitution.capture == 0 || substitution.replacement == 0 ||
            !substitutions.emplace(substitution.capture, substitution.replacement).second)
            return semantic_inline_status_v1::duplicate_substitution;
    }
    for (const auto capture : callee.captures)
        if (substitutions.count(capture) == 0)
            return semantic_inline_status_v1::missing_capture;

    semantic_inline_graph_v1 composed;
    composed.identity = request.caller_identity;
    std::unordered_set<std::uint64_t> identities;
    for (std::size_t index = 0; index < callee.nodes.size(); ++index) {
        auto node = callee.nodes[index];
        if (request.identity_policy == semantic_inline_identity_policy_v1::rebase)
            node.identity = request.identity_seed + index;
        if (!identities.insert(node.identity).second)
            return semantic_inline_status_v1::identity_collision;
        for (auto& operand : node.operands) {
            const auto replacement = substitutions.find(operand);
            if (replacement != substitutions.end()) operand = replacement->second;
        }
        for (auto& output : node.results) {
            const auto replacement = substitutions.find(output);
            if (replacement != substitutions.end()) output = replacement->second;
        }
        node.profile = request.profile_replacement;
        node.generation = std::max(node.generation, request.minimum_generation);
        node.provenance.insert(node.provenance.begin(), callee.identity);
        composed.nodes.push_back(std::move(node));
    }
    *result = std::move(composed);
    return semantic_inline_status_v1::success;
}

std::optional<std::vector<std::string>> canonicalize_semantic_inline_graph_v1(
    const semantic_inline_graph_v1& graph) {
    if (graph.identity == 0) return std::nullopt;
    std::vector<std::string> records;
    for (const auto& node : graph.nodes) {
        if (node.operation.empty() || node.results.empty() || !node.profile.valid() ||
            node.generation == 0 || node.provenance.empty()) return std::nullopt;
        std::ostringstream record;
        record << node.operation << '|';
        for (const auto operand : node.operands) record << operand << ',';
        record << '|';
        for (const auto output : node.results) record << output << ',';
        record << '|' << node.profile.low << ':' << node.profile.high
               << '|' << node.generation << '|';
        for (const auto origin : node.provenance) record << origin << ',';
        records.push_back(record.str());
    }
    std::sort(records.begin(), records.end());
    return records;
}

}  // namespace Cellerator::compiler::ir::semantic
