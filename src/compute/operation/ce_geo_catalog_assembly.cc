#include <Cellerator/compute/architecture/provider.hh>
#include <Cellerator/compute/operation/relation_algebra_catalog.hh>

namespace cellerator::compute::architecture::providers::nvidia {
compiled_provider_manifest_v1 sm70_compiled_provider_manifest_v1() noexcept;
}

namespace cellerator::compute::operation {
relation_algebra_catalog_view_v1 assembled_relation_algebra_catalog_v1()
    noexcept;

bool validate_ce_geo_catalog_assembly_v1() noexcept {
    const architecture::compiled_provider_manifest_v1 providers =
        architecture::providers::nvidia::sm70_compiled_provider_manifest_v1();
    const relation_algebra_catalog_view_v1 relations =
        assembled_relation_algebra_catalog_v1();
    return providers.registrations != nullptr && providers.count == 1u
        && relations.entries != nullptr
        && relations.entry_count == relation_algebra_catalog_entry_count_v1
        && relations.fragments != nullptr
        && relations.fragment_count
            == relation_algebra_catalog_fragment_count_v1;
}

} // namespace cellerator::compute::operation
