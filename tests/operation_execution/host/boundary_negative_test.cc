#include <Cellerator/compute/architecture/provider.hh>
#include <Cellerator/compute/operation/candidate_catalog_v2.hh>
#include <Cellerator/execution/geometry_acquisition_v2.hh>
#include <Cellerator/execution/opaque_artifact.hh>

#include <cstdlib>

namespace architecture = cellerator::compute::architecture;
namespace catalog = cellerator::compute::math::core;
namespace acquisition = cellerator::execution::acquisition_v2;
namespace execution = cellerator::execution;

// The production device query lives in a CUDA translation unit.  This host-
// only validator supplies the registry's unused validation seam so provider
// metadata failures can be tested without broadening into CUDA validation.
namespace cellerator::runtime {
bool valid_device_descriptor_v1(const device_descriptor_v1 &) noexcept {
    return false;
}
}  // namespace cellerator::runtime

namespace {

void require(bool value) {
    if (!value) {
        std::abort();
    }
}

void catalog_rejects_malformed_metadata() {
    catalog::candidate_descriptor_v2 descriptor{};
    require(catalog::validate_candidate_descriptor_v2(descriptor)
            == catalog::candidate_catalog_status_v2::invalid_identity);
    descriptor.schema_version = 99u;
    require(catalog::validate_candidate_descriptor_v2(descriptor)
            == catalog::candidate_catalog_status_v2::invalid_header);

    catalog::candidate_catalog_fragment_v2 fragment{};
    fragment.fragment_identity = {1u, 2u};
    fragment.provider_identity = {3u, 4u};
    fragment.name = "negative";
    fragment.entries = &descriptor;
    fragment.entry_count = 1u;
    require(catalog::validate_candidate_catalog_fragment_v2(fragment)
            == catalog::candidate_catalog_status_v2::invalid_header);
    fragment.reserved[2] = 1u;
    require(catalog::validate_candidate_catalog_fragment_v2(fragment)
            == catalog::candidate_catalog_status_v2::nonzero_reserved);
}

void acquisition_rejects_identity_route_and_capacity_failures() {
    acquisition::acquisition_request request{};
    require(acquisition::validate_request(request).code
            == acquisition::status_code::invalid_identity);
    request.structure = {1u, 2u};
    request.epoch.value = 1u;
    request.preferred_route = static_cast<acquisition::route>(99u);
    require(acquisition::validate_request(request).code
            == acquisition::status_code::invalid_route);

    acquisition::acquisition_facade missing{};
    acquisition::acquisition_requirements requirements{};
    const char source = 'x';
    const acquisition::projection_requirement projection{
        {9u, 10u}, 1u, 1u, 4u,
        acquisition::logical_primary_values, {}};
    request.preferred_route = acquisition::route::compile_now;
    request.source = {&source, 1u};
    request.projection_requirements = &projection;
    request.projection_requirement_count = 1u;
    require(acquisition::query_requirements(missing, request, &requirements).code
            == acquisition::status_code::callback_unavailable);

    const acquisition::default_assembly assembly{};
    require(acquisition::validate_default_assembly(assembly).code
            == acquisition::status_code::invalid_argument);
}

void opaque_transport_rejects_unvalidated_residency() {
    execution::validated_opaque_execution_artifact_v2 validated{};
    const execution::opaque_artifact_status host_status =
        execution::validate_opaque_execution_artifact_v2_host(
            {}, {}, &validated);
    require(host_status.code == execution::opaque_artifact_code::invalid_argument);

    execution::bound_opaque_execution_artifact_v2 bound{};
    const execution::opaque_artifact_status bind_status =
        execution::bind_opaque_execution_artifact_v2_device(
            validated, {}, {}, &bound);
    require(bind_status.code == execution::opaque_artifact_code::invalid_argument);
}

void provider_registry_is_fail_closed_and_transactional() {
    architecture::architecture_provider_v1 provider{};
    require(architecture::validate_architecture_provider_v1(provider)
            == architecture::provider_status_v1::invalid_identity);
    architecture::architecture_provider_registry_v1 registry{};
    require(architecture::register_architecture_provider_v1(&registry, provider)
            == architecture::provider_status_v1::invalid_identity);
    require(registry.size == 0u);
    require(architecture::register_compiled_providers_v1(
                &registry, {nullptr, 1u})
            == architecture::provider_status_v1::invalid_argument);
    require(registry.size == 0u);
    require(architecture::seal_architecture_provider_registry_v1(&registry)
            == architecture::provider_status_v1::success);
    require(architecture::register_architecture_provider_v1(&registry, provider)
            == architecture::provider_status_v1::invalid_argument);
}

}  // namespace

int main() {
    catalog_rejects_malformed_metadata();
    acquisition_rejects_identity_route_and_capacity_failures();
    opaque_transport_rejects_unvalidated_residency();
    provider_registry_is_fail_closed_and_transactional();
    return 0;
}
