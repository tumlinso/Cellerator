#include <Cellerator/compiler/build/define_the_host_only_root_project_contract_v1.hh>

#include <cassert>

int main() {
    using namespace Cellerator::compiler::build;
    static_assert(is_host_only(host_only_root_project_contract_v1));
    static_assert(
        host_only_root_project_contract_v1.default_accelerator_enablement ==
        accelerator_enablement_v1::automatic);
    assert(host_only_root_project_contract_v1.required_languages.front() ==
           "CXX");
}
