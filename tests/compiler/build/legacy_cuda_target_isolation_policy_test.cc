#include <Cellerator/compiler/build/legacy_cuda_target_isolation_policy_v1.hh>

int main() {
    using namespace Cellerator::compiler::build;
    static_assert(legacy_cuda_isolation_contract_v1.discovery_is_target_scoped);
    static_assert(legacy_cuda_isolation_contract_v1.architecture_is_target_scoped);
    static_assert(legacy_cuda_isolation_contract_v1.provider_manifest_is_conditional);
    static_assert(!legacy_cuda_isolation_contract_v1.host_only_performs_cuda_detection);
}
