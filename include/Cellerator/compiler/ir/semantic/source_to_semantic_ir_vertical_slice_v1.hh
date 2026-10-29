#pragma once

#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace Cellerator::compiler::ir::semantic {

struct semantic_source_location_v1 {
    std::uint32_t line = 0;
    std::uint32_t column = 0;
};

struct source_linked_semantic_operation_v1 {
    cellerator::compute::operation::v2::stable_id identity{};
    cellerator::compute::operation::v2::operation_kind kind =
        cellerator::compute::operation::v2::operation_kind::relation_apply;
    std::string result;
    std::vector<std::string> operands;
    std::string destination_domain;
    semantic_source_location_v1 source{};
};

struct source_linked_semantic_module_v1 {
    std::string field;
    std::string profile;
    std::vector<source_linked_semantic_operation_v1> operations;
};

struct semantic_source_receipt_v1 {
    std::uint64_t source_hash = 0;
    std::uint64_t semantic_hash = 0;
    std::uint32_t operation_count = 0;
    bool exact_source_mapping = false;
    bool operation_core_compatible = false;
};

enum class semantic_vertical_slice_status_v1 : std::uint8_t {
    success = 0,
    invalid_source,
    missing_field,
    missing_profile,
    unsupported_operation,
    malformed_operation,
    invalid_semantic_ir,
    missing_referee_binding,
    invalid_referee_operation,
};

[[nodiscard]] std::optional<source_linked_semantic_module_v1>
lower_cell_source_to_semantic_ir_v1(
    const std::string& source,
    semantic_vertical_slice_status_v1* status = nullptr) noexcept;

[[nodiscard]] std::optional<std::string> write_semantic_ir_v1(
    const source_linked_semantic_module_v1& module,
    semantic_vertical_slice_status_v1* status = nullptr) noexcept;

[[nodiscard]] std::optional<source_linked_semantic_module_v1> read_semantic_ir_v1(
    const std::string& text,
    semantic_vertical_slice_status_v1* status = nullptr) noexcept;

[[nodiscard]] bool operation_core_compatible_v1(
    const source_linked_semantic_module_v1& module) noexcept;

[[nodiscard]] std::optional<std::unordered_map<std::string, double>>
execute_semantic_referee_v1(
    const source_linked_semantic_module_v1& module,
    std::unordered_map<std::string, double> bindings,
    semantic_vertical_slice_status_v1* status = nullptr) noexcept;

[[nodiscard]] std::optional<semantic_source_receipt_v1> make_source_linked_receipt_v1(
    const std::string& source,
    const source_linked_semantic_module_v1& module,
    semantic_vertical_slice_status_v1* status = nullptr) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
