#include <Cellerator/compiler/frontend/source/map_shadow_ast_nodes_back_to_cellerator_syntax_v1.hh>

namespace Cellerator::compiler::frontend::source {

bool shadow_ast_map_v1::bind(shadow_ast_binding_v1 binding) {
    if (binding.shadow_ast_node == 0 || binding.placeholder_id == 0 || binding.cellerator_parse_nodes.empty() ||
        (binding.cellerator_parse_nodes.size() > 1 && !binding.explicit_many_to_one)) return false;
    for (const auto& capture : binding.captures) if (capture.name.empty() || !capture.source.valid()) return false;
    return bindings_.emplace(binding.shadow_ast_node, std::move(binding)).second;
}

const shadow_ast_binding_v1* shadow_ast_map_v1::find(std::uint64_t node) const noexcept {
    const auto found = bindings_.find(node); return found == bindings_.end() ? nullptr : &found->second;
}

bool shadow_ast_map_v1::traceable() const noexcept {
    for (const auto& entry : bindings_) {
        const auto& binding = entry.second;
        if (binding.cellerator_parse_nodes.empty()) return false;
        for (const auto& capture : binding.captures) if (!capture.source.valid()) return false;
    }
    return true;
}

std::string generated_shadow_symbol_v1(std::uint64_t id) {
    return "__cellerator_generated_shadow_v1_" + std::to_string(id);
}

bool shadow_symbol_is_reserved_v1(std::string_view symbol) noexcept {
    return symbol.rfind("__cellerator_generated_shadow_v1_", 0) == 0;
}

} // namespace Cellerator::compiler::frontend::source
