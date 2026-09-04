#pragma once

#include <Cellerator/compiler/sema/field/define_execution_field_semantic_ownership_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::sema::field {

struct planning_fact_v1 {
    std::string key;
    std::string value;
};

// Strength is domain-specific but monotonic: a child may retain or strengthen
// an inherited hard constraint, never weaken it.
struct hard_constraint_v1 {
    std::string key;
    std::string value;
    std::uint32_t strength = 0;
};

struct nested_field_request_v1 {
    const execution_field_semantics_v1* parent = nullptr;
    execution_field_semantics_v1 child;
    std::vector<planning_fact_v1> inherited_facts;
    std::vector<hard_constraint_v1> inherited_constraints;
    std::vector<planning_fact_v1> local_fact_overlays;
    std::vector<hard_constraint_v1> local_constraint_overlays;
    bool explicitly_inline = false;
};

struct resolved_nested_field_v1 {
    execution_field_identity_v1 parent_identity{};
    execution_field_identity_v1 child_identity{};
    std::vector<planning_fact_v1> effective_facts;
    std::vector<hard_constraint_v1> effective_constraints;
    bool separately_nameable = false;
    bool planning_subproblem = true;
    bool optimization_boundary = true;
    bool movement_barrier = true;
};

enum class nested_field_status_v1 : std::uint8_t {
    success = 0,
    missing_parent,
    invalid_parent,
    invalid_child,
    child_outside_parent,
    child_not_separately_named,
    duplicate_overlay,
    weakened_inherited_constraint,
};

[[nodiscard]] nested_field_status_v1 resolve_and_implement_nested_field_semantics_v1(
    const nested_field_request_v1& request,
    resolved_nested_field_v1* resolved) noexcept;

}  // namespace Cellerator::compiler::sema::field
