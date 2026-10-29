#include <Cellerator/execution/identity.hh>
#include <Cellerator/planner/end_to_end_planner.hh>
#include <Cellerator/runtime/device_descriptor.hh>
#include <Cellerator/runtime/session.cuh>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <iostream>

namespace {

namespace runtime = cellerator::runtime;

int require(bool condition, const char *message) {
    if (condition) return 0;
    std::cerr << "device descriptor test failed: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    std::uint64_t query_count = 0u;
    runtime::device_descriptor_v1 descriptor{};
    if (require(runtime::query_device_descriptor_v1(
                    -1, false, &descriptor, &query_count)
                    == runtime::device_descriptor_status_v1::success,
                "cold descriptor query")
        || require(runtime::valid_device_descriptor_v1(descriptor),
                   "descriptor validation")
        || require(query_count >= 2u, "hardware queries were not accounted")
        || require(descriptor.vendor == runtime::nvidia_pci_vendor_id,
                   "NVIDIA vendor identity")
        || require(descriptor.warp_size == 32u, "NVIDIA warp size")
        || require(descriptor.maximum_threads_per_multiprocessor
                       >= descriptor.maximum_threads_per_block,
                   "thread residency limits")
        || require(descriptor.hardware_compatibility_identity !=
                       descriptor.performance_class_identity,
                   "compatibility and performance identities are distinct")) {
        return 1;
    }

    const auto runtime_view =
        runtime::derive_runtime_device_performance_class(descriptor);
    const auto execution_view =
        runtime::derive_execution_device_performance_class(descriptor);
    const auto planner_view =
        runtime::derive_planner_device_performance_key(descriptor);
    if (require(runtime_view.device == descriptor.ordinal
                    && runtime_view.compute_major == descriptor.compute_major
                    && runtime_view.compute_minor == descriptor.compute_minor,
                "runtime compatibility view")
        || require(execution_view.vendor == descriptor.vendor
                       && execution_view.build_identity
                           == descriptor.performance_class_identity,
                   "execution compatibility view")
        || require(planner_view.vendor == descriptor.vendor
                       && planner_view.performance_class
                           == descriptor.performance_class_identity,
                   "planner compatibility view")) {
        return 1;
    }

    const std::uint64_t before_seal_query_count = query_count;
    const runtime::device_descriptor_v1 before_seal_descriptor = descriptor;
    if (require(runtime::query_device_descriptor_v1(
                    descriptor.ordinal, true, &descriptor, &query_count)
                    == runtime::device_descriptor_status_v1::invalid_state,
                "sealed query rejection")
        || require(query_count == before_seal_query_count,
                   "sealed session performed a device query")
        || require(descriptor.hardware_compatibility_identity
                       == before_seal_descriptor.hardware_compatibility_identity
                       && descriptor.performance_class_identity
                           == before_seal_descriptor.performance_class_identity,
                   "sealed rejection modified cold truth")) {
        return 1;
    }

    runtime::device_descriptor_v1 explicit_descriptor{};
    std::uint64_t explicit_query_count = 0u;
    if (require(runtime::query_device_descriptor_v1(
                    before_seal_descriptor.ordinal,
                    false,
                    &explicit_descriptor,
                    &explicit_query_count)
                    == runtime::device_descriptor_status_v1::success,
                "explicit ordinal query")
        || require(explicit_descriptor.hardware_compatibility_identity
                       == before_seal_descriptor.hardware_compatibility_identity
                       && explicit_descriptor.performance_class_identity
                           == before_seal_descriptor.performance_class_identity,
                   "stable cold identities")
        || require(explicit_query_count + 1u == before_seal_query_count,
                   "current-device query accounting")) {
        return 1;
    }

    std::cout << "celleratorCeGeoDeviceDescriptorTest passed"
              << " ordinal=" << descriptor.ordinal
              << " sm=" << descriptor.compute_major << descriptor.compute_minor
              << " queries=" << query_count << '\n';
    return 0;
}
