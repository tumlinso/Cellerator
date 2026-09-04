#pragma once

#include <Cellerator/compiler/frontend/source/define_the_unified_source_location_model_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace Cellerator::compiler::frontend::source {

struct syntax_capture_v1 { std::string name; source_span_v1 source; };
struct shadow_ast_binding_v1 {
    std::uint64_t shadow_ast_node = 0;
    std::uint64_t placeholder_id = 0;
    std::vector<std::uint64_t> cellerator_parse_nodes;
    std::vector<syntax_capture_v1> captures;
    bool explicit_many_to_one = false;
};

class shadow_ast_map_v1 {
  public:
    [[nodiscard]] bool bind(shadow_ast_binding_v1 binding);
    [[nodiscard]] const shadow_ast_binding_v1* find(std::uint64_t shadow_ast_node) const noexcept;
    [[nodiscard]] bool traceable() const noexcept;
  private:
    std::unordered_map<std::uint64_t, shadow_ast_binding_v1> bindings_;
};

[[nodiscard]] std::string generated_shadow_symbol_v1(std::uint64_t placeholder_id);
[[nodiscard]] bool shadow_symbol_is_reserved_v1(std::string_view symbol) noexcept;

} // namespace Cellerator::compiler::frontend::source
