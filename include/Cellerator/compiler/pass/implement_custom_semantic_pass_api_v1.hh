#pragma once

#include <Cellerator/compiler/ir/common/implement_validation_mode_plumbing_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_relation_apply_and_transpose_operations_v1.hh>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cellerator::compiler::pass::v1 {

struct semantic_source_mapping_v1 {
    std::uint64_t operation_identity = 0;
    std::string source_file;
    std::uint32_t line = 0;
    std::uint32_t column = 0;
};

struct semantic_pass_context_v1 {
    std::vector<Cellerator::compiler::ir::semantic::relation_apply_operation_ir_v1>*
        relation_applies = nullptr;
    const std::vector<std::string>* profiles = nullptr;
    const std::vector<semantic_source_mapping_v1>* source_mappings = nullptr;
    std::vector<std::string>* diagnostics = nullptr;
    std::unordered_map<std::string, std::uint64_t>* analysis_cache = nullptr;
    cellerator::compiler::ir::trust_mode validation_mode =
        cellerator::compiler::ir::trust_mode::checked;
};

using semantic_pass_run_v1 = bool (*)(semantic_pass_context_v1&) noexcept;
using semantic_pass_validate_v1 = bool (*)(
    const semantic_pass_context_v1&) noexcept;

enum class semantic_pass_status_v1 : std::uint8_t {
    success = 0,
    invalid_context,
    pass_failed,
    validation_failed,
};

[[nodiscard]] semantic_pass_status_v1 run_custom_semantic_pass_v1(
    semantic_pass_context_v1& context, semantic_pass_run_v1 run,
    semantic_pass_validate_v1 validate) noexcept;

}  // namespace cellerator::compiler::pass::v1
