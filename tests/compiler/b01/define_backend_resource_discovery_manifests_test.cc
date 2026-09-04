#include <Cellerator/compiler/build/define_backend_resource_discovery_manifests_v1.hh>

#include <cassert>

int main() {
    using namespace Cellerator::compiler::build;
    static_assert(backend_resource_keys_v1.size() == 8);
    static_assert(backend_resource_manifest_is_complete_v1());
    assert(backend_resource_keys_v1.front() == "host_cxx");
    assert(backend_resource_keys_v1.back() == "resource_dir");
}
