#include <Cellerator/compute/architecture/provider.hh>
#include <Cellerator/compute/architecture/providers/nvidia/sm70_provider.hh>

namespace cellerator::compute::architecture::providers::nvidia {
namespace {

constexpr provider_registration_function_v1 registrations[] = {
    &register_sm70_provider_v1};

} // namespace

// Cold, allocation-free source manifest for the central catalog assembly.
// Provider selection remains in the sealed architecture registry; this file
// does not perform a device query or make a planner decision.
compiled_provider_manifest_v1 sm70_compiled_provider_manifest_v1() noexcept {
    return {registrations, 1u};
}

} // namespace cellerator::compute::architecture::providers::nvidia
