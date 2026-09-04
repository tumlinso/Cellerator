#include <Cellerator/compiler/backend/implement_backend_registry_and_selection_v1.hh>

#include "mock_backend_provider_v1.hh"

#include <cassert>

namespace cb = cellerator::compiler::backend::v1;

int main() {
    const auto provider = make_mock_backend_provider_v1();
    cb::backend_registry_v1 registry;
    assert(registry.register_backend({{"fallback", 8}, {"builtin", 7}, &provider, 0}) ==
           cb::backend_selection_status_v1::selected);
    assert(registry.register_backend({{"fast", 4}, {"plugin.cc", 9}, &provider, 10}) ==
           cb::backend_selection_status_v1::selected);
    const cb::backend_target_v1 target{{"mock-cpu-unknown", 16}, {}, {}};
    auto result = registry.select({target});
    assert(result.status == cb::backend_selection_status_v1::selected);
    assert(result.entry != nullptr && result.entry->priority == 10);
    assert(!result.used_conventional_fallback);

    result = registry.select({target, cb::backend_capability_ordinary_object_v1,
                              cb::backend_selection_policy_v1::force_named,
                              {"fallback", 8}, true});
    assert(result.status == cb::backend_selection_status_v1::selected);
    assert(result.used_conventional_fallback);

    result = registry.select({target, cb::backend_capability_native_fragment_v1});
    assert(result.status == cb::backend_selection_status_v1::unavailable);
    result = registry.select({target, cb::backend_capability_ordinary_object_v1,
                              cb::backend_selection_policy_v1::force_named,
                              {"missing", 7}, false});
    assert(result.status == cb::backend_selection_status_v1::forced_backend_unavailable);

    cb::backend_registry_v1 ambiguous;
    assert(ambiguous.register_backend({{"one", 3}, {"one.cc", 6}, &provider, 1}) ==
           cb::backend_selection_status_v1::selected);
    assert(ambiguous.register_backend({{"two", 3}, {"two.cc", 6}, &provider, 1}) ==
           cb::backend_selection_status_v1::selected);
    assert(ambiguous.select({target}).status ==
           cb::backend_selection_status_v1::ambiguous);
}
