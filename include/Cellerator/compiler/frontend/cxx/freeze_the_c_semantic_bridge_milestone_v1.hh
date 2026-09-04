#pragma once

#include <Cellerator/compiler/frontend/cxx/bind_source_captures_to_c_declarations_and_expressions_v1.hh>
#include <Cellerator/compiler/frontend/cxx/expose_constexpr_and_constant_evaluation_results_v1.hh>
#include <Cellerator/compiler/frontend/cxx/extract_canonical_and_spelled_c_types_v1.hh>
#include <Cellerator/compiler/frontend/cxx/integrate_template_instantiation_with_typed_biological_o_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {

inline constexpr std::uint32_t cxx_semantic_bridge_milestone_schema_version_v1 = 1;

enum class cxx_semantic_bridge_milestone_status_v1 : std::uint8_t {
    success = 0,
    null_output,
    schema_mismatch,
    invalid_adapter,
    placeholder_resolution_failed,
    numeric_type_resolution_failed,
    template_resolution_failed,
    constexpr_resolution_failed,
};

struct cxx_semantic_bridge_milestone_request_v1 {
    std::uint32_t schema_version = cxx_semantic_bridge_milestone_schema_version_v1;
    const upstream_clang_adapter_v1* adapter = nullptr;
    source_capture_request_v1 activated_placeholder;
    std::string biological_template_name;
    std::vector<constexpr_import_request_v1> constants;
};

// Cold, versioned evidence record. AST pointers remain borrowed from the
// immutable frontend snapshot that owns the supplied adapter.
struct cxx_semantic_bridge_milestone_v1 {
    std::uint32_t schema_version = cxx_semantic_bridge_milestone_schema_version_v1;
    std::uint32_t clang_adapter_schema_version = upstream_clang_adapter_schema_version_v1;
    std::uint32_t llvm_major = 0;
    bound_source_capture_v1 placeholder;
    cxx_type_record_v1 numeric_type;
    std::vector<biological_template_operation_v1> operations;
    std::vector<constexpr_value_v1> constants;
};

cxx_semantic_bridge_milestone_status_v1 freeze_cxx_semantic_bridge_milestone_v1(
    const cxx_semantic_bridge_milestone_request_v1& request,
    cxx_semantic_bridge_milestone_v1* milestone) noexcept;

}  // namespace Cellerator::compiler::frontend::cxx
