#pragma once

#include <Cellerator/compiler/frontend/cxx/freeze_the_upstream_clang_adapter_boundary_v1.hh>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {

inline constexpr std::uint32_t source_capture_binding_schema_version_v1 = 1;
inline constexpr std::uint32_t unspecified_source_offset_v1 =
    std::numeric_limits<std::uint32_t>::max();

enum class source_capture_kind_v1 : std::uint8_t {
    domain = 1,
    state,
    relation,
    qualifier_expression,
    native_call,
    inline_ir,
};

enum class source_capture_ast_kind_v1 : std::uint8_t {
    declaration = 1,
    expression,
};

enum class source_capture_binding_status_v1 : std::uint8_t {
    success = 0,
    schema_mismatch,
    invalid_adapter,
    invalid_request,
    missing_capture,
    ambiguous_capture,
};

struct source_capture_request_v1 {
    source_capture_kind_v1 kind = source_capture_kind_v1::domain;
    std::string spelling;
    std::uint32_t source_offset = unspecified_source_offset_v1;
};

struct source_provenance_v1 {
    std::string file;
    std::uint32_t begin_offset = 0;
    std::uint32_t end_offset = 0;
    std::uint32_t line = 0;
    std::uint32_t column = 0;
};

struct bound_source_capture_v1 {
    source_capture_kind_v1 kind = source_capture_kind_v1::domain;
    source_capture_ast_kind_v1 ast_kind = source_capture_ast_kind_v1::declaration;
    const void* ast_node = nullptr;
    std::string spelling;
    std::string resolved_type;
    source_provenance_v1 provenance;
};

struct source_capture_diagnostic_v1 {
    std::uint32_t capture_index = 0;
    source_capture_binding_status_v1 code = source_capture_binding_status_v1::success;
    source_provenance_v1 provenance;
    std::string message;
};

struct source_capture_binding_result_v1 {
    std::vector<bound_source_capture_v1> captures;
    std::vector<source_capture_diagnostic_v1> diagnostics;
};

source_capture_binding_status_v1 bind_source_captures_v1(
    std::uint32_t schema_version,
    const upstream_clang_adapter_v1& adapter,
    const std::vector<source_capture_request_v1>& requests,
    source_capture_binding_result_v1* result) noexcept;

}  // namespace Cellerator::compiler::frontend::cxx
