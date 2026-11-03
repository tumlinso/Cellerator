#pragma once

#include <Cellerator/geometry/dependency/reads.hh>
#include <Cellerator/execution/launch_bindings.hh>

#include <span>

namespace cellerator::ops::mechanism {

namespace dependency = geometry::dependency;

struct operation_tag;
struct access_contract_tag;
struct omission_rule_tag;
using operation_id = structure::persistent_identity<operation_tag>;
using access_contract_id = structure::persistent_identity<access_contract_tag>;
using omission_rule_id = structure::persistent_identity<omission_rule_tag>;

enum class Phase : std::uint8_t {
    condition, primal, input_vjp, condition_vjp, parameter_vjp, jvp
};

struct OutputAccess {
    dependency::SourceRef destination{};
    dependency::SelectionRef selection{};
    execution::output_effect_contract effect{
        execution::output_update_kind::overwrite, false, false, 0,
        execution::invalid_scalar_binding_id,
        execution::invalid_scalar_binding_id};
    // Zero denotes the complete source derivative, rather than one argument
    // role. A nonzero role names an explicitly declared role-specific write.
    dependency::role_id role{};
};

// Each phase declares its actual reads and writes. Derivative demands need not
// equal primal demands; in particular a realized forward zero is not enough to
// remove condition or parameter derivative access.
struct PhaseDemand {
    Phase phase = Phase::primal;
    std::span<const dependency::ReadDependency> reads{};
    std::span<const OutputAccess> writes{};
};

struct AccessContract {
    access_contract_id identity{};
    std::uint64_t revision = 0;
    operation_id operation{};
    std::span<const PhaseDemand> phases{};
    std::span<const dependency::ConditionReads> conditions{};
};

struct DerivativeRequest {
    bool inputs = false;
    bool conditions = false;
    bool parameters = false;
    bool jvp = false;
    std::uint32_t order = 0;
};

dependency::Status validate_access(
    const AccessContract& access,
    std::span<const dependency::SourceDescriptor> sources,
    std::span<const dependency::SelectionDescriptor> selections) noexcept;

// Condition validation currently requires one matching condition-phase read
// to cover each declared condition input. A union of smaller phase selections
// is conservatively rejected until a union-capable preparation is supplied.

} // namespace cellerator::ops::mechanism
