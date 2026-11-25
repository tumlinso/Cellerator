#pragma once

#include <Cellerator/execution/planning/operation_planning.hh>
#include <Cellerator/runtime/device_descriptor.hh>

namespace cellerator::execution::planning {

inline device_performance_key derive_device_performance_key(
    const runtime::device_descriptor_v1 &descriptor) noexcept {
    device_performance_key result{};
    if (!runtime::valid_device_descriptor_v1(descriptor)) return result;
    result.vendor = descriptor.vendor;
    result.architecture_major = descriptor.compute_major;
    result.architecture_minor = descriptor.compute_minor;
    result.performance_class = descriptor.performance_class_identity;
    return result;
}

} // namespace cellerator::execution::planning
