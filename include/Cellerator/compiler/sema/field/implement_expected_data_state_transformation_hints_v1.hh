#pragma once

#include <Cellerator/compiler/sema/field/implement_automatic_lifetime_and_generation_transfer_v1.hh>
#include <Cellerator/compiler/sema/field/implement_named_representative_profile_binding_v1.hh>

#include <cstdint>
#include <string>

namespace Cellerator::compiler::sema::field {

enum class data_state_transformation_kind_v1 : std::uint8_t {
    value_only = 1,
    support_changing,
    topology_changing,
    unknown,
};

struct expected_data_state_transformation_hint_v1 {
    std::uint64_t operation_identity = 0;
    std::uint64_t input_profile_state_identity = 0;
    std::string expected_post_state;
    data_state_transformation_kind_v1 transformation =
        data_state_transformation_kind_v1::unknown;
};

enum class profile_state_precision_v1 : std::uint8_t {
    exact = 1,
    inferred,
    widened,
    unknown,
};

struct expected_data_state_transformation_v1 {
    std::uint64_t operation_identity = 0;
    std::uint64_t input_profile_state_identity = 0;
    std::uint64_t output_profile_state_identity = 0;
    std::string output_profile_state_name;
    std::uint32_t advance_components = state_component_none_v1;
    profile_state_precision_v1 precision = profile_state_precision_v1::unknown;
    std::uint32_t widening_cost = 0;
    bool explicitly_selected = false;
    bool costly_widening = false;
    std::string warning;
};

enum class expected_data_state_transformation_status_v1 : std::uint8_t {
    success = 0,
    invalid_binding,
    invalid_operation,
    unknown_input_state,
    unavailable_expected_state,
};

[[nodiscard]] expected_data_state_transformation_status_v1
implement_expected_data_state_transformation_hints_v1(
    const representative_profile_binding_v1& binding,
    const expected_data_state_transformation_hint_v1& hint,
    expected_data_state_transformation_v1* transformation) noexcept;

}  // namespace Cellerator::compiler::sema::field
