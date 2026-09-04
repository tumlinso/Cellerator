#include <Cellerator/compiler/build/create_host_only_compiler_smoke_targets_v1.hh>

#include <cassert>

#ifndef CELLERATOR_SMOKE_COMPONENT
#define CELLERATOR_SMOKE_COMPONENT "aggregate"
#endif

int main() {
    using namespace Cellerator::compiler::build;
    static_assert(host_smoke_components_v1.size() == 5);
    static_assert(!host_smokes_link_cuda_v1);
    assert(std::string_view{CELLERATOR_SMOKE_COMPONENT}.size() != 0);
}
