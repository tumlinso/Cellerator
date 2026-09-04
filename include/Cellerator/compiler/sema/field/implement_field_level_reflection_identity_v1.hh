#pragma once

#include <Cellerator/compiler/sema/field/define_execution_field_semantic_ownership_v1.hh>

#include <cstdint>
#include <string>

namespace Cellerator::compiler::sema::field {

inline constexpr std::uint32_t field_reflection_identity_schema_version_v1 = 1;

struct field_reflection_identity_v1 {
    std::uint32_t schema_version = field_reflection_identity_schema_version_v1;
    execution_field_identity_v1 field_identity{};
    std::string stable_export_name;

    friend bool operator==(const field_reflection_identity_v1& lhs,
                           const field_reflection_identity_v1& rhs) noexcept {
        return lhs.schema_version == rhs.schema_version &&
            lhs.field_identity == rhs.field_identity &&
            lhs.stable_export_name == rhs.stable_export_name;
    }
};

enum class field_reflection_identity_status_v1 : std::uint8_t {
    success = 0,
    invalid_field,
    invalid_output,
};

[[nodiscard]] field_reflection_identity_status_v1
implement_field_level_reflection_identity_v1(
    const execution_field_semantics_v1& field,
    field_reflection_identity_v1* identity) noexcept;

}  // namespace Cellerator::compiler::sema::field
