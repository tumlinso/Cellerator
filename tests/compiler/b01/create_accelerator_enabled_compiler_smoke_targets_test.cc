#include <Cellerator/compiler/build/create_accelerator_enabled_compiler_smoke_targets_v1.hh>

int main() {
    using namespace Cellerator::compiler::build;
    static_assert(accelerator_smoke_v1.conditional);
    static_assert(accelerator_smoke_v1.links_realization);
    static_assert(accelerator_smoke_v1.links_runtime_when_available);
    static_assert(accelerator_smoke_v1.baseline_sm == 70);
    static_assert(!accelerator_smoke_v1.changes_host_only_dependencies);
}
