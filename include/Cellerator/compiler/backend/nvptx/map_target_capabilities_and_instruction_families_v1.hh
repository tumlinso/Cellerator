#pragma once

#include <Cellerator/geometry/persistence/execution_capability_manifest_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::backend::nvptx {

enum class target_capability_validation_mode_v1 : std::uint8_t {
    checked = 1u,
    trusted,
    unsafe,
};

struct target_instruction_requirement_v1 {
    std::uint32_t compute_major = 0u;
    std::uint32_t compute_minor = 0u;
    cellpack::persistence::execution_instruction_family_v1 instruction_family =
        cellpack::persistence::execution_instruction_family_v1::invalid;
    cellpack::persistence::execution_collective_scope_v1 collective_scope =
        cellpack::persistence::execution_collective_scope_v1::invalid;
    std::uint32_t collective_threads = 0u;
    std::uint32_t instruction_m = 0u;
    std::uint32_t instruction_n = 0u;
    std::uint32_t instruction_k = 0u;
    cellpack::persistence::execution_capability_numeric_type_v1 relation_storage_type =
        cellpack::persistence::execution_capability_numeric_type_v1::invalid;
    cellpack::persistence::execution_capability_numeric_type_v1 dense_input_type =
        cellpack::persistence::execution_capability_numeric_type_v1::invalid;
    cellpack::persistence::execution_capability_numeric_type_v1 accumulation_type =
        cellpack::persistence::execution_capability_numeric_type_v1::invalid;
    cellpack::persistence::execution_capability_numeric_type_v1 output_type =
        cellpack::persistence::execution_capability_numeric_type_v1::invalid;
    cellpack::persistence::execution_matrix_layout_v1 operand_a_layout =
        cellpack::persistence::execution_matrix_layout_v1::invalid;
    cellpack::persistence::execution_matrix_layout_v1 operand_b_layout =
        cellpack::persistence::execution_matrix_layout_v1::invalid;
    cellpack::persistence::execution_matrix_layout_v1 accumulation_layout =
        cellpack::persistence::execution_matrix_layout_v1::invalid;
    cellpack::persistence::execution_matrix_layout_v1 output_layout =
        cellpack::persistence::execution_matrix_layout_v1::invalid;
    cellpack::persistence::execution_instruction_sparsity_v1 instruction_sparsity =
        cellpack::persistence::execution_instruction_sparsity_v1::invalid;
    cellpack::persistence::execution_structured_operand_v1 structured_operand =
        cellpack::persistence::execution_structured_operand_v1::none;
    cellpack::persistence::execution_structured_group_semantics_v1 structured_group_semantics =
        cellpack::persistence::execution_structured_group_semantics_v1::none;
    std::uint32_t required_memory_interface_flags = 0u;
};

enum class target_capability_mapping_status_v1 : std::uint8_t {
    supported = 0u,
    supported_with_warning,
    rejected,
    invalid_manifest,
    invalid_requirement,
};

struct target_capability_mapping_v1 {
    target_capability_mapping_status_v1 status =
        target_capability_mapping_status_v1::invalid_requirement;
    bool admissible = false;
    bool unsafe_override = false;
    std::vector<std::string> mismatches;

    explicit operator bool() const noexcept { return admissible; }
};

[[nodiscard]] target_capability_mapping_v1 map_target_capability_v1(
    const target_instruction_requirement_v1& requirement,
    const cellpack::persistence::execution_capability_manifest_v1& manifest,
    target_capability_validation_mode_v1 mode);

}  // namespace Cellerator::compiler::backend::nvptx
