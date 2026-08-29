#include <Cellerator/compute/architecture/provider.hh>

#include <cstdint>

namespace cellerator::compute::architecture {
namespace {

bool memory_interface_present(
    const architecture_provider_v1 &provider,
    architecture_identity_v1 identity) noexcept {
    for (std::uint32_t index = 0u;
         index < provider.memory_interface_count; ++index) {
        if (same_architecture_identity_v1(
                provider.memory_interfaces[index].identity, identity))
            return true;
    }
    return false;
}

bool capability_matches_device(
    const matrix_engine_capability_v1 &capability,
    const runtime::device_descriptor_v1 &device) noexcept {
    if (capability.vendor == architecture_vendor_v1::nvidia
        && device.vendor != runtime::nvidia_pci_vendor_id)
        return false;
    if (capability.vendor != architecture_vendor_v1::generic
        && capability.vendor != architecture_vendor_v1::nvidia)
        return false;
    if (capability.architecture_class
        != static_cast<std::uint16_t>(device.architecture))
        return false;
    return compute_capability_in_range_v1(
        capability, device.compute_major, device.compute_minor);
}

} // namespace

provider_status_v1 validate_architecture_provider_v1(
    const architecture_provider_v1 &provider) noexcept {
    if (provider.schema_version != architecture_provider_schema_version_v1
        || provider.record_bytes != sizeof(architecture_provider_v1))
        return provider_status_v1::invalid_header;
    if (!valid_architecture_identity_v1(provider.identity))
        return provider_status_v1::invalid_identity;
    if (provider.name == nullptr || provider.capabilities == nullptr
        || provider.capability_count == 0u
        || (provider.memory_interface_count != 0u
            && provider.memory_interfaces == nullptr)
        || provider.flags != 0u)
        return provider_status_v1::invalid_argument;
    for (std::uint32_t value : provider.reserved)
        if (value != 0u) return provider_status_v1::nonzero_reserved;

    for (std::uint32_t index = 0u; index < provider.memory_interface_count;
         ++index) {
        if (validate_matrix_memory_interface_v1(
                provider.memory_interfaces[index])
            != capability_status_v1::success)
            return provider_status_v1::invalid_memory_interface;
        for (std::uint32_t previous = 0u; previous < index; ++previous)
            if (same_architecture_identity_v1(
                    provider.memory_interfaces[index].identity,
                    provider.memory_interfaces[previous].identity))
                return provider_status_v1::invalid_memory_interface;
    }

    for (std::uint32_t index = 0u; index < provider.capability_count; ++index) {
        const matrix_engine_capability_v1 &capability =
            provider.capabilities[index];
        if (validate_matrix_engine_capability_v1(capability)
                != capability_status_v1::success
            || !same_architecture_identity_v1(
                capability.provider_identity, provider.identity))
            return provider_status_v1::invalid_capability;
        if ((capability.flags & capability_memory_interface_present) != 0u
            && !memory_interface_present(
                provider, capability.memory_interface_identity))
            return provider_status_v1::invalid_memory_interface;
        for (std::uint32_t previous = 0u; previous < index; ++previous)
            if (same_architecture_identity_v1(capability.identity,
                    provider.capabilities[previous].identity))
                return provider_status_v1::invalid_capability;
    }
    return provider_status_v1::success;
}

provider_status_v1 register_architecture_provider_v1(
    architecture_provider_registry_v1 *registry,
    const architecture_provider_v1 &provider) noexcept {
    if (registry == nullptr || registry->sealed
        || registry->size > architecture_provider_registry_capacity_v1)
        return provider_status_v1::invalid_argument;
    const provider_status_v1 valid =
        validate_architecture_provider_v1(provider);
    if (valid != provider_status_v1::success) return valid;
    if (find_architecture_provider_v1(*registry, provider.identity) != nullptr)
        return provider_status_v1::duplicate_provider;
    if (registry->size == architecture_provider_registry_capacity_v1)
        return provider_status_v1::registry_full;
    registry->providers[registry->size++] = provider;
    return provider_status_v1::success;
}

provider_status_v1 register_compiled_providers_v1(
    architecture_provider_registry_v1 *registry,
    compiled_provider_manifest_v1 manifest) noexcept {
    if (registry == nullptr || registry->sealed
        || (manifest.count != 0u && manifest.registrations == nullptr))
        return provider_status_v1::invalid_argument;
    architecture_provider_registry_v1 staged = *registry;
    for (std::uint32_t index = 0u; index < manifest.count; ++index) {
        if (manifest.registrations[index] == nullptr)
            return provider_status_v1::registration_failed;
        const provider_status_v1 status =
            manifest.registrations[index](&staged);
        if (status != provider_status_v1::success) return status;
    }
    *registry = staged;
    return provider_status_v1::success;
}

provider_status_v1 seal_architecture_provider_registry_v1(
    architecture_provider_registry_v1 *registry) noexcept {
    if (registry == nullptr || registry->sealed)
        return provider_status_v1::invalid_argument;
    registry->sealed = true;
    return provider_status_v1::success;
}

provider_status_v1 active_architecture_providers_v1(
    const architecture_provider_registry_v1 &registry,
    const runtime::device_descriptor_v1 &device,
    const architecture_provider_v1 **output,
    std::uint32_t output_capacity,
    std::uint32_t *output_count) noexcept {
    if (!registry.sealed || registry.size
            > architecture_provider_registry_capacity_v1
        || !runtime::valid_device_descriptor_v1(device)
        || output_count == nullptr
        || (output_capacity != 0u && output == nullptr))
        return provider_status_v1::invalid_argument;
    std::uint32_t required = 0u;
    for (std::uint32_t provider_index = 0u;
         provider_index < registry.size; ++provider_index) {
        const architecture_provider_v1 &provider =
            registry.providers[provider_index];
        bool active = false;
        for (std::uint32_t capability_index = 0u;
             capability_index < provider.capability_count; ++capability_index) {
            if (capability_matches_device(
                    provider.capabilities[capability_index], device)) {
                active = true;
                break;
            }
        }
        if (active) {
            if (required < output_capacity) output[required] = &provider;
            ++required;
        }
    }
    *output_count = required;
    return required <= output_capacity
        ? provider_status_v1::success
        : provider_status_v1::output_capacity;
}

const architecture_provider_v1 *find_architecture_provider_v1(
    const architecture_provider_registry_v1 &registry,
    architecture_identity_v1 identity) noexcept {
    if (!valid_architecture_identity_v1(identity)
        || registry.size > architecture_provider_registry_capacity_v1)
        return nullptr;
    for (std::uint32_t index = 0u; index < registry.size; ++index)
        if (same_architecture_identity_v1(
                registry.providers[index].identity, identity))
            return &registry.providers[index];
    return nullptr;
}

} // namespace cellerator::compute::architecture
