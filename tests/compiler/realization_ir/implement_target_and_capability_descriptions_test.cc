#include <Cellerator/compiler/ir/realization/implement_target_and_capability_descriptions_v1.hh>

#include <cassert>

using namespace cellerator::compiler::ir::realization::v1;

int main() {
    target_capability_v1 v100;
    v100.identity = {1u, 70u};
    v100.architecture = architecture_class_v1::nvidia_volta;
    v100.compute = {7u, 0u};
    v100.instruction_families = {"cuda.core", "mma.sync.m8n8k4"};
    v100.maximum_collective_scope = collective_scope_v1::multi_device;
    v100.memory_interfaces = memory_device_global_v1 | memory_device_shared_v1 |
        memory_peer_v1;
    v100.numeric_support = numeric_f16_v1 | numeric_f32_v1 | numeric_f64_v1 |
        numeric_i8_v1 | numeric_i32_v1;
    v100.graph_capture = true;
    v100.toolchain = "cuda-12";
    v100.runtime = "cuda";
    v100.backend = "cellerator-native";

    target_requirement_v1 requirement;
    requirement.architecture = architecture_class_v1::nvidia_volta;
    requirement.minimum_compute = {7u, 0u};
    requirement.instruction_families = {"cuda.core"};
    requirement.minimum_collective_scope = collective_scope_v1::block;
    requirement.memory_interfaces = memory_device_global_v1;
    requirement.numeric_support = numeric_f32_v1;
    requirement.graph_capture = true;
    requirement.runtime = "cuda";
    requirement.backend = "cellerator-native";

    assert(validate_target_capability_v1(v100) == capability_status_v1::compatible);
    assert(satisfies_target_requirement_v1(v100, requirement) ==
        capability_status_v1::compatible);

    auto missing_numeric = requirement;
    missing_numeric.numeric_support |= numeric_bf16_v1;
    assert(satisfies_target_requirement_v1(v100, missing_numeric) ==
        capability_status_v1::numeric_support_missing);

    auto wrong_architecture = requirement;
    wrong_architecture.architecture = architecture_class_v1::nvidia_ampere;
    assert(satisfies_target_requirement_v1(v100, wrong_architecture) ==
        capability_status_v1::architecture_mismatch);

    auto invalid = v100;
    invalid.instruction_families.push_back("cuda.core");
    assert(validate_target_capability_v1(invalid) ==
        capability_status_v1::invalid_description);
}
