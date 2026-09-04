#pragma once

#include <Cellerator/compiler/frontend/source/define_the_unified_source_location_model_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::sema::field {

inline constexpr std::uint32_t execution_field_semantics_schema_version_v1 = 1;

struct execution_field_identity_v1 {
    std::uint64_t low = 0;
    std::uint64_t high = 0;

    friend constexpr bool operator==(execution_field_identity_v1 lhs,
                                     execution_field_identity_v1 rhs) noexcept {
        return lhs.low == rhs.low && lhs.high == rhs.high;
    }
};

enum class captured_value_access_v1 : std::uint8_t {
    read = 1,
    write,
    read_write,
};

struct captured_value_v1 {
    std::string canonical_name;
    std::uint64_t declaration_identity = 0;
    captured_value_access_v1 access = captured_value_access_v1::read;
};

enum observable_effect_v1 : std::uint32_t {
    field_effect_none_v1 = 0,
    field_effect_reads_memory_v1 = 1u << 0,
    field_effect_writes_memory_v1 = 1u << 1,
    field_effect_volatile_v1 = 1u << 2,
    field_effect_atomic_v1 = 1u << 3,
    field_effect_may_throw_v1 = 1u << 4,
    field_effect_io_v1 = 1u << 5,
    field_effect_synchronizes_v1 = 1u << 6,
    field_effect_opaque_v1 = 1u << 7,
};

struct observable_boundary_v1 {
    frontend::source::source_span_v1 source{};
    std::uint32_t effects = field_effect_none_v1;
};

struct profile_environment_v1 {
    std::string stable_name;
    std::uint64_t content_digest_low = 0;
    std::uint64_t content_digest_high = 0;

    [[nodiscard]] bool bound() const noexcept {
        return !stable_name.empty() &&
            (content_digest_low != 0 || content_digest_high != 0);
    }
};

struct execution_field_definition_v1 {
    std::uint32_t schema_version = execution_field_semantics_schema_version_v1;
    std::string stable_source_name;
    std::string explicit_field_name;
    frontend::source::source_span_v1 source{};
    std::vector<captured_value_v1> captured_values;
    std::vector<observable_boundary_v1> observable_boundaries;
    profile_environment_v1 profile_environment;
    std::uint32_t semantic_effects = field_effect_none_v1;
};

struct execution_field_semantics_v1 {
    execution_field_identity_v1 identity{};
    std::string stable_source_name;
    std::string explicit_field_name;
    frontend::source::source_span_v1 source{};
    std::vector<captured_value_v1> captured_values;
    std::vector<observable_boundary_v1> observable_boundaries;
    profile_environment_v1 profile_environment;
    std::uint32_t semantic_effects = field_effect_none_v1;
};

enum class execution_field_definition_status_v1 : std::uint8_t {
    success = 0,
    schema_mismatch,
    invalid_source,
    missing_source_identity,
    invalid_capture,
    duplicate_capture,
    boundary_outside_field,
};

[[nodiscard]] execution_field_definition_status_v1
define_execution_field_semantic_ownership_v1(
    const execution_field_definition_v1& definition,
    execution_field_semantics_v1* semantics) noexcept;

// Field ownership is lexical and closed: an operation is jointly planned only
// when its complete source span is inside this exact field's source space.
[[nodiscard]] bool execution_field_owns_operation_v1(
    const execution_field_semantics_v1& field,
    frontend::source::source_span_v1 operation) noexcept;

}  // namespace Cellerator::compiler::sema::field
