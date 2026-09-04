#include <Cellerator/compiler/build/define_build_tree_generated_header_ownership_v1.hh>

int main() {
    using namespace Cellerator::compiler::build;
    static_assert(generated_header_fields_v1.size() == 5);
    static_assert(generated_header_owner_v1 == "compiler build tree");
}
