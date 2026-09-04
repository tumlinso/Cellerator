#include <Cellerator/compiler/frontend/source/map_shadow_ast_nodes_back_to_cellerator_syntax_v1.hh>

#include <iostream>
#include <stdexcept>

using namespace Cellerator::compiler::frontend::source;

int main() {
    try {
        shadow_ast_map_v1 map;
        if (!map.bind({11, 91, {101}, {{"x", {{4, 8}, {4, 9}}}}, false}) ||
            !map.bind({12, 92, {102, 103}, {{"r", {{4, 10}, {4, 11}}}}, true}) || !map.traceable())
            throw std::runtime_error("valid AST bindings rejected");
        if (map.bind({13, 93, {104, 105}, {}, false}) || map.find(11) == nullptr)
            throw std::runtime_error("implicit many-to-one mapping accepted");
        const auto symbol = generated_shadow_symbol_v1(91);
        if (!shadow_symbol_is_reserved_v1(symbol) || shadow_symbol_is_reserved_v1("user_symbol"))
            throw std::runtime_error("generated symbol namespace not isolated");
        std::cout << "validated traceable shadow AST mappings and reserved symbols\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
