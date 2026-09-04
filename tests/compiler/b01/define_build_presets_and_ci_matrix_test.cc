#include <Cellerator/compiler/build/define_build_presets_and_ci_matrix_v1.hh>

int main() {
    using namespace Cellerator::compiler::build;
    static_assert(compiler_ci_presets_v1.size() == 7);
    static_assert(non_hardware_presets_v1.size() == 5);
}
