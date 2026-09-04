#pragma once

#include <Cellerator/compiler/sema/implement_numerical_tuple_semantics_v1.hh>
#include <Cellerator/compiler/sema/implement_operation_kind_resolution_v1.hh>
#include <Cellerator/compiler/sema/implement_output_update_effect_semantics_v1.hh>
#include <Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh>

#include <cstdint>

namespace cellerator::compiler::sema::v1 {

inline constexpr std::uint32_t biological_sema_conformance_version = 1u;

struct biological_sema_problem {
    compute::operation::v2::relation_algebra_problem preserved{};
    source_operation_kind operation = source_operation_kind::relation_apply;
    numerical_tuple numeric{};
    output_effect_semantics output{};
    bool relation_algebra_present = false;
};

biological_sema_problem lower_through_biological_sema(
    const compute::operation::v2::operation_problem &problem) noexcept;
biological_sema_problem lower_through_biological_sema(
    const compute::operation::v2::relation_algebra_problem &problem) noexcept;
compute::operation::v2::operation_problem recover_operation_problem(
    const biological_sema_problem &problem) noexcept;
compute::operation::v2::relation_algebra_problem recover_relation_algebra_problem(
    const biological_sema_problem &problem) noexcept;
bool planning_information_preserved(
    const compute::operation::v2::relation_algebra_problem &source,
    const biological_sema_problem &lowered) noexcept;

}  // namespace cellerator::compiler::sema::v1
