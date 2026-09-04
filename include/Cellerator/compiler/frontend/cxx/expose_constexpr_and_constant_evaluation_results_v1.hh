#pragma once

#include <Cellerator/compiler/frontend/cxx/freeze_the_upstream_clang_adapter_boundary_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {

inline constexpr std::uint32_t constexpr_import_schema_version_v1 = 1;

enum class constexpr_value_kind_v1 : std::uint8_t {
    signed_integer = 1,
    unsigned_integer,
    floating_point,
    boolean,
    string,
};

enum class constexpr_import_status_v1 : std::uint8_t {
    success = 0,
    schema_mismatch,
    invalid_adapter,
    missing_constant,
    ambiguous_constant,
    not_constant,
    unsupported_value,
};

struct constexpr_import_request_v1 {
    std::string qualified_name;
};

struct constexpr_value_v1 {
    const void* declaration = nullptr;
    std::string qualified_name;
    std::string canonical_type;
    constexpr_value_kind_v1 kind = constexpr_value_kind_v1::signed_integer;
    std::int64_t signed_value = 0;
    std::uint64_t unsigned_value = 0;
    double floating_value = 0.0;
    bool boolean_value = false;
    std::string string_value;
};

constexpr_import_status_v1 import_constexpr_values_v1(
    std::uint32_t schema_version,
    const upstream_clang_adapter_v1& adapter,
    const std::vector<constexpr_import_request_v1>& requests,
    std::vector<constexpr_value_v1>* values) noexcept;

}  // namespace Cellerator::compiler::frontend::cxx
