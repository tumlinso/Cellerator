#include <Cellerator/compiler/backend/nvptx/map_target_capabilities_and_instruction_families_v1.hh>

namespace Cellerator::compiler::backend::nvptx {

target_capability_mapping_v1 map_target_capability_v1(
    const target_instruction_requirement_v1& requirement,
    const cellpack::persistence::execution_capability_manifest_v1& manifest,
    const target_capability_validation_mode_v1 mode) {
    using namespace cellpack::persistence;
    target_capability_mapping_v1 result;
    if (!validate_execution_capability_manifest_v1(manifest)) {
        result.status = target_capability_mapping_status_v1::invalid_manifest;
        return result;
    }
    if (requirement.compute_major == 0u || requirement.compute_minor > 99u ||
        requirement.instruction_family == execution_instruction_family_v1::invalid ||
        requirement.collective_scope == execution_collective_scope_v1::invalid ||
        requirement.collective_threads == 0u || requirement.instruction_m == 0u ||
        requirement.instruction_n == 0u || requirement.instruction_k == 0u ||
        requirement.relation_storage_type == execution_capability_numeric_type_v1::invalid ||
        requirement.dense_input_type == execution_capability_numeric_type_v1::invalid ||
        requirement.accumulation_type == execution_capability_numeric_type_v1::invalid ||
        requirement.output_type == execution_capability_numeric_type_v1::invalid ||
        requirement.operand_a_layout == execution_matrix_layout_v1::invalid ||
        requirement.operand_b_layout == execution_matrix_layout_v1::invalid ||
        requirement.accumulation_layout == execution_matrix_layout_v1::invalid ||
        requirement.output_layout == execution_matrix_layout_v1::invalid ||
        requirement.instruction_sparsity == execution_instruction_sparsity_v1::invalid) {
        result.status = target_capability_mapping_status_v1::invalid_requirement;
        return result;
    }
    const auto below_minimum = requirement.compute_major < manifest.minimum_compute_capability_major ||
        (requirement.compute_major == manifest.minimum_compute_capability_major &&
         requirement.compute_minor < manifest.minimum_compute_capability_minor);
    const auto above_maximum = requirement.compute_major > manifest.maximum_compute_capability_major ||
        (requirement.compute_major == manifest.maximum_compute_capability_major &&
         requirement.compute_minor > manifest.maximum_compute_capability_minor);
    if (below_minimum || above_maximum) result.mismatches.emplace_back("compute capability");
    if (requirement.instruction_family != manifest.instruction_family)
        result.mismatches.emplace_back("instruction family");
    if (requirement.collective_scope != manifest.collective_scope ||
        requirement.collective_threads != manifest.collective_threads)
        result.mismatches.emplace_back("collective scope or participants");
    if (requirement.instruction_m != manifest.instruction_m ||
        requirement.instruction_n != manifest.instruction_n ||
        requirement.instruction_k != manifest.instruction_k)
        result.mismatches.emplace_back("instruction shape");
    if (requirement.relation_storage_type != manifest.relation_storage_type ||
        requirement.dense_input_type != manifest.dense_input_type ||
        requirement.accumulation_type != manifest.accumulation_type ||
        requirement.output_type != manifest.output_type)
        result.mismatches.emplace_back("numeric tuple");
    if (requirement.operand_a_layout != manifest.operand_a_layout ||
        requirement.operand_b_layout != manifest.operand_b_layout ||
        requirement.accumulation_layout != manifest.accumulation_layout ||
        requirement.output_layout != manifest.output_layout)
        result.mismatches.emplace_back("matrix layouts");
    if (requirement.instruction_sparsity != manifest.instruction_sparsity ||
        requirement.structured_operand != manifest.structured_operand ||
        requirement.structured_group_semantics != manifest.structured_group_semantics)
        result.mismatches.emplace_back("sparsity contract");
    if ((manifest.flags & capability_memory_interface_present) == 0u ||
        (requirement.required_memory_interface_flags & ~manifest.memory_interface_flags) != 0u)
        result.mismatches.emplace_back("memory interface");

    if (result.mismatches.empty()) {
        result.status = target_capability_mapping_status_v1::supported;
        result.admissible = true;
        return result;
    }
    if (mode == target_capability_validation_mode_v1::checked) {
        result.status = target_capability_mapping_status_v1::rejected;
        return result;
    }
    result.status = target_capability_mapping_status_v1::supported_with_warning;
    result.admissible = true;
    result.unsafe_override = mode == target_capability_validation_mode_v1::unsafe;
    return result;
}

}  // namespace Cellerator::compiler::backend::nvptx
