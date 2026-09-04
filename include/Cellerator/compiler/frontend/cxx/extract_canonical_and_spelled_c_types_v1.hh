#pragma once

#include <Cellerator/compiler/frontend/cxx/bind_source_captures_to_c_declarations_and_expressions_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {

inline constexpr std::uint32_t cxx_type_extraction_schema_version_v1 = 1;

enum cxx_type_trait_v1 : std::uint32_t {
    cxx_type_builtin_v1 = 1u << 0,
    cxx_type_half_v1 = 1u << 1,
    cxx_type_bfloat16_v1 = 1u << 2,
    cxx_type_vector_v1 = 1u << 3,
    cxx_type_pointer_v1 = 1u << 4,
    cxx_type_lvalue_reference_v1 = 1u << 5,
    cxx_type_rvalue_reference_v1 = 1u << 6,
    cxx_type_address_space_v1 = 1u << 7,
    cxx_type_user_defined_v1 = 1u << 8,
};

enum class cxx_type_extraction_status_v1 : std::uint8_t {
    success = 0,
    schema_mismatch,
    invalid_adapter,
    invalid_capture,
    incomplete_type,
};

struct cxx_type_record_v1 {
    std::string user_spelling;
    std::string canonical_spelling;
    std::string canonical_identity;
    std::uint64_t size_bytes = 0;
    std::uint64_t alignment_bytes = 0;
    std::uint32_t traits = 0;
    std::uint32_t address_space = 0;
};

cxx_type_extraction_status_v1 extract_cxx_types_v1(
    std::uint32_t schema_version,
    const upstream_clang_adapter_v1& adapter,
    const std::vector<bound_source_capture_v1>& captures,
    std::vector<cxx_type_record_v1>* records) noexcept;

}  // namespace Cellerator::compiler::frontend::cxx
