#include <Cellerator/compiler/ir/realization/implement_target_and_capability_descriptions_v1.hh>

#include <algorithm>
#include <set>

namespace cellerator::compiler::ir::realization::v1 {
namespace {

capability_status_v1 fail(
    capability_status_v1 status,
    std::string* error,
    const char* message) noexcept {
    if (error != nullptr) {
        *error = message;
    }
    return status;
}

bool compute_at_least(compute_capability_v1 value, compute_capability_v1 minimum) noexcept {
    return value.major > minimum.major ||
        (value.major == minimum.major && value.minor >= minimum.minor);
}

bool contains_all(
    const std::vector<std::string>& values,
    const std::vector<std::string>& required) {
    return std::all_of(required.begin(), required.end(), [&](const std::string& value) {
        return std::find(values.begin(), values.end(), value) != values.end();
    });
}

} // namespace

capability_status_v1 validate_target_capability_v1(
    const target_capability_v1& capability,
    std::string* error) noexcept {
    std::set<std::string> instructions;
    for (const auto& instruction : capability.instruction_families) {
        if (instruction.empty() || !instructions.insert(instruction).second) {
            return fail(capability_status_v1::invalid_description, error,
                "instruction families must be nonempty and unique");
        }
    }
    if (!valid(capability.identity) || capability.memory_interfaces == 0u ||
        capability.numeric_support == 0u || capability.toolchain.empty() ||
        capability.runtime.empty() || capability.backend.empty()) {
        return fail(capability_status_v1::invalid_description, error,
            "identity, memory, numeric, toolchain, runtime, and backend are required");
    }
    if (capability.architecture != architecture_class_v1::host &&
        capability.compute.major == 0u) {
        return fail(capability_status_v1::invalid_description, error,
            "accelerator targets require a compute capability");
    }
    if (error != nullptr) {
        error->clear();
    }
    return capability_status_v1::compatible;
}

capability_status_v1 satisfies_target_requirement_v1(
    const target_capability_v1& capability,
    const target_requirement_v1& requirement,
    std::string* error) noexcept {
    const auto valid_status = validate_target_capability_v1(capability, error);
    if (valid_status != capability_status_v1::compatible) {
        return valid_status;
    }
    if (capability.architecture != requirement.architecture) {
        return fail(capability_status_v1::architecture_mismatch, error,
            "architecture class mismatch");
    }
    if (!compute_at_least(capability.compute, requirement.minimum_compute)) {
        return fail(capability_status_v1::compute_capability_insufficient, error,
            "compute capability is insufficient");
    }
    if (!contains_all(capability.instruction_families, requirement.instruction_families)) {
        return fail(capability_status_v1::missing_instruction, error,
            "required instruction family is missing");
    }
    if (capability.maximum_collective_scope < requirement.minimum_collective_scope) {
        return fail(capability_status_v1::collective_scope_insufficient, error,
            "collective scope is insufficient");
    }
    if ((capability.memory_interfaces & requirement.memory_interfaces) !=
        requirement.memory_interfaces) {
        return fail(capability_status_v1::memory_interface_missing, error,
            "required memory interface is missing");
    }
    if ((capability.numeric_support & requirement.numeric_support) !=
        requirement.numeric_support) {
        return fail(capability_status_v1::numeric_support_missing, error,
            "required numeric support is missing");
    }
    if (requirement.graph_capture && !capability.graph_capture) {
        return fail(capability_status_v1::graph_capture_missing, error,
            "graph capture is required");
    }
    if ((!requirement.toolchain.empty() && capability.toolchain != requirement.toolchain) ||
        (!requirement.runtime.empty() && capability.runtime != requirement.runtime) ||
        (!requirement.backend.empty() && capability.backend != requirement.backend)) {
        return fail(capability_status_v1::provider_mismatch, error,
            "toolchain, runtime, or backend provider mismatch");
    }
    if (error != nullptr) {
        error->clear();
    }
    return capability_status_v1::compatible;
}

} // namespace cellerator::compiler::ir::realization::v1
