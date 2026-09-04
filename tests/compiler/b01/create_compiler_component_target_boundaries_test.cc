#include <Cellerator/compiler/build/create_compiler_component_target_boundaries_v1.hh>

#include <cassert>

int main() {
    using namespace Cellerator::compiler::build;
    static_assert(compiler_component_names_v1.size() == 9);
    static_assert(compiler_component_graph_is_acyclic_v1());
    for (const auto name : compiler_component_names_v1) {
        assert(!name.empty());
    }
}
