#include <Cellerator/execution/planning/providers/provider.hh>
#include <Cellerator/execution/planning/candidates/relation_algebra_catalog.hh>

namespace cellerator::execution::planning::architecture::providers::nvidia {
compiled_provider_manifest_v1 sm70_compiled_provider_manifest_v1() noexcept;
}

namespace cellerator::execution::operation {
relation_algebra_catalog_view_v1 assembled_relation_algebra_catalog_v1()
    noexcept;

bool validate_relation_catalog_readiness_v1() noexcept {
    const ::cellerator::execution::planning::architecture::compiled_provider_manifest_v1 providers =
        ::cellerator::execution::planning::architecture::providers::nvidia::sm70_compiled_provider_manifest_v1();
    const relation_algebra_catalog_view_v1 relations =
        assembled_relation_algebra_catalog_v1();
    return providers.registrations != nullptr && providers.count == 1u
        && relations.entries != nullptr
        && relations.entry_count == relation_algebra_catalog_entry_count_v1
        && relations.fragments != nullptr
        && relations.fragment_count
            == relation_algebra_catalog_fragment_count_v1;
}

} // namespace cellerator::execution::operation
