#pragma once

#include <Cellerator/compiler/frontend/cxx/freeze_the_upstream_clang_adapter_boundary_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {

inline constexpr std::uint32_t overload_semantic_candidate_schema_version_v1 = 1;

enum class overload_semantic_candidate_status_v1 : std::uint8_t {
    success = 0,
    schema_mismatch,
    invalid_adapter,
    missing_resolved_call,
    ambiguous_source_location,
    biological_operation_mismatch,
    biological_domain_mismatch,
};

struct overload_semantic_request_v1 {
    std::uint32_t source_offset = 0;
    std::string required_operation;
    std::string required_domain_type;
};

struct overload_semantic_candidate_v1 {
    const void* selected_declaration = nullptr;
    std::string qualified_name;
    std::string function_type;
    bool cellerator_aware = false;
    bool biologically_compatible = false;
    std::string declared_operation;
    std::string declared_domain_type;
    std::string mechanism;
};

struct overload_semantic_result_v1 {
    std::vector<overload_semantic_candidate_v1> candidates;
    std::vector<overload_semantic_candidate_status_v1> statuses;
};

overload_semantic_candidate_status_v1 resolve_overloads_and_semantic_candidates_v1(
    std::uint32_t schema_version,
    const upstream_clang_adapter_v1& adapter,
    const std::vector<overload_semantic_request_v1>& requests,
    overload_semantic_result_v1* result) noexcept;

}  // namespace Cellerator::compiler::frontend::cxx
