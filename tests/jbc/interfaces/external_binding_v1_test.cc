#include <Cellerator/execution/joint_compiler/external_binding_v1.hh>

#include <cassert>
#include <cstdint>

namespace joint_compiler = cellerator::execution::joint_compiler;
namespace execution = cellerator::execution;

int main() {
    alignas(16) std::uint8_t first[16]{};
    alignas(16) std::uint8_t second[32]{};
    joint_compiler::external_extent_v1 extents[2]{};
    extents[0].address = first;
    extents[0].location = {execution::residency_kind::host, {}, -1, 1u};
    extents[0].bytes = sizeof(first);
    extents[0].alignment = 16u;
    extents[0].order = {1u, 2u};
    extents[0].generation = {3u};
    extents[0].readiness = {1u, 1u};
    extents[0].lease = {2u, 1u};
    extents[1] = extents[0];
    extents[1].address = second;
    extents[1].plane_byte_offset = sizeof(first);
    extents[1].bytes = sizeof(second);
    extents[1].readiness = {1u, 2u};
    extents[1].lease = {2u, 2u};

    joint_compiler::external_binding_v1 binding{};
    binding.binding_identity = {4u, 1u};
    binding.atom_identity = {4u, 2u};
    binding.plane_identity = {4u, 3u};
    binding.extents = extents;
    binding.extent_count = 2u;
    binding.total_bytes = sizeof(first) + sizeof(second);
    assert(joint_compiler::validate_external_binding_v1(binding));

    extents[1].plane_byte_offset += 1u;
    assert(joint_compiler::validate_external_binding_v1(binding).code
        == joint_compiler::external_binding_validation_code_v1::
            extent_offset_mismatch);
    extents[1].plane_byte_offset = sizeof(first);
    extents[1].generation.value += 1u;
    assert(joint_compiler::validate_external_binding_v1(binding).code
        == joint_compiler::external_binding_validation_code_v1::
            inconsistent_generation);
    extents[1].generation = extents[0].generation;
    extents[1].readiness = {};
    assert(joint_compiler::validate_external_binding_v1(binding).code
        == joint_compiler::external_binding_validation_code_v1::
            invalid_readiness_token);
    extents[1].readiness = {1u, 2u};
    binding.total_bytes -= 1u;
    assert(joint_compiler::validate_external_binding_v1(binding).code
        == joint_compiler::external_binding_validation_code_v1::
            total_bytes_mismatch);
    return 0;
}
