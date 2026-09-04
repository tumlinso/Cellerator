#pragma once

#include <Cellerator/compiler/ir/semantic/freeze_semantic_ir_module_and_symbol_scopes_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_domain_and_axis_ir_types_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Cellerator::compiler::ir::semantic {

enum class execution_field_kind_ir_v1 : std::uint8_t { named = 1, anonymous_field };
enum class execution_field_boundary_ir_v1 : std::uint8_t { transparent = 1, explicit_boundary };
enum class execution_field_visibility_ir_v1 : std::uint8_t {
    unavailable = 1,
    call_boundary,
    inline_semantics,
};
enum execution_field_effect_ir_v1 : std::uint32_t {
    field_effect_none_ir_v1 = 0,
    field_effect_reads_ir_v1 = 1u << 0,
    field_effect_writes_ir_v1 = 1u << 1,
    field_effect_synchronizes_ir_v1 = 1u << 2,
    field_effect_opaque_ir_v1 = 1u << 3,
};

struct execution_field_value_ir_v1 {
    std::uint64_t symbol_identity = 0;
    bool mutable_access = false;
};

struct execution_field_fact_ir_v1 {
    std::string name;
    std::string value;
};

struct execution_field_constraint_ir_v1 {
    std::string name;
    std::string value;
    bool hard = true;
};

struct execution_field_region_ir_v1 {
    std::uint64_t identity = 0;
    semantic_scope_id_v1 scope = invalid_semantic_scope_id_v1;
    std::uint64_t parent_field_identity = 0;
    execution_field_kind_ir_v1 kind = execution_field_kind_ir_v1::anonymous_field;
    execution_field_boundary_ir_v1 boundary = execution_field_boundary_ir_v1::transparent;
    std::vector<execution_field_value_ir_v1> captures;
    std::vector<execution_field_value_ir_v1> results;
    semantic_identity_v1 profile_environment{};
    std::vector<execution_field_fact_ir_v1> facts;
    std::vector<execution_field_constraint_ir_v1> constraints;
    std::vector<std::uint64_t> operations;
    std::uint32_t observable_effects = field_effect_none_ir_v1;
    bool semantic_body_available = true;
};

struct effective_field_environment_ir_v1 {
    semantic_identity_v1 profile_environment{};
    std::vector<execution_field_fact_ir_v1> facts;
    std::vector<execution_field_constraint_ir_v1> constraints;
    std::uint32_t observable_effects = field_effect_none_ir_v1;
};

enum class execution_field_ir_validation_code_v1 : std::uint8_t {
    success = 0,
    invalid_identity,
    invalid_scope,
    invalid_parent,
    invalid_kind,
    invalid_boundary,
    invalid_profile,
    invalid_binding,
    duplicate_binding,
    invalid_directive,
    duplicate_operation,
};

class frozen_execution_field_regions_v1 {
public:
    [[nodiscard]] const execution_field_region_ir_v1* field(
        std::uint64_t identity) const noexcept;
    [[nodiscard]] std::optional<effective_field_environment_ir_v1>
    effective_environment(std::uint64_t identity) const;
    [[nodiscard]] execution_field_visibility_ir_v1 visibility(
        std::uint64_t caller_identity,
        std::uint64_t callee_identity,
        bool request_inline_semantics) const noexcept;

private:
    std::vector<execution_field_region_ir_v1> fields_;
    friend std::optional<frozen_execution_field_regions_v1>
    freeze_execution_field_operations_and_regions_v1(
        std::vector<execution_field_region_ir_v1>,
        const frozen_semantic_scope_module_v1&,
        execution_field_ir_validation_code_v1*) noexcept;
};

[[nodiscard]] std::optional<frozen_execution_field_regions_v1>
freeze_execution_field_operations_and_regions_v1(
    std::vector<execution_field_region_ir_v1> fields,
    const frozen_semantic_scope_module_v1& scopes,
    execution_field_ir_validation_code_v1* status = nullptr) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
