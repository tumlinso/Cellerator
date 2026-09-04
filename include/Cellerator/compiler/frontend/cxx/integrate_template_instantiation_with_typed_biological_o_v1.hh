#pragma once

#include <Cellerator/compiler/frontend/cxx/freeze_the_upstream_clang_adapter_boundary_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {

inline constexpr std::uint32_t biological_template_operation_schema_version_v1 = 1;

enum class biological_template_operation_status_v1 : std::uint8_t {
    success = 0,
    schema_mismatch,
    invalid_adapter,
    missing_template,
    ambiguous_template,
    no_instantiations,
    unresolved_substitution,
};

struct biological_operation_candidate_v1 {
    std::string identity;
    std::string mechanism;
};

struct biological_template_operation_v1 {
    const void* function_declaration = nullptr;
    std::string operation_identity;
    std::string template_name;
    std::string dependent_constraint;
    std::string numeric_user_spelling;
    std::string numeric_canonical_spelling;
    std::string domain_user_spelling;
    std::string domain_canonical_spelling;
    std::string result_canonical_spelling;
    std::vector<biological_operation_candidate_v1> candidates;
};

biological_template_operation_status_v1 instantiate_biological_template_operations_v1(
    std::uint32_t schema_version,
    const upstream_clang_adapter_v1& adapter,
    const std::string& qualified_template_name,
    std::vector<biological_template_operation_v1>* operations) noexcept;

}  // namespace Cellerator::compiler::frontend::cxx
