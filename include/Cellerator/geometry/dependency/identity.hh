#pragma once

#include <Cellerator/structure/identity.hh>

namespace cellerator::geometry::dependency {

struct foundation_tag;
struct regime_tag;
struct coordinate_system_tag;
struct assumption_set_tag;
struct parameter_family_tag;
struct selection_tag;
struct source_tag;
struct representation_tag;
struct condition_tag;
struct mechanism_tag;
struct role_tag;

using foundation_id = structure::persistent_identity<foundation_tag>;
using regime_id = structure::persistent_identity<regime_tag>;
using coordinate_system_id = structure::persistent_identity<coordinate_system_tag>;
using assumption_set_id = structure::persistent_identity<assumption_set_tag>;
using parameter_family_id = structure::persistent_identity<parameter_family_tag>;
using selection_id = structure::persistent_identity<selection_tag>;
using source_id = structure::persistent_identity<source_tag>;
using representation_id = structure::persistent_identity<representation_tag>;
using condition_id = structure::persistent_identity<condition_tag>;
using mechanism_id = structure::persistent_identity<mechanism_tag>;
using role_id = structure::persistent_identity<role_tag>;

enum class StatusCode : std::uint8_t {
    success,
    invalid_identity,
    invalid_epoch,
    invalid_applicability,
    invalid_selection,
    duplicate_identity,
    domain_mismatch,
    order_mismatch,
    extent_mismatch,
    unknown_source,
    unknown_selection,
    invalid_read,
    invalid_pattern,
    invalid_participant,
    invalid_condition,
    invalid_output_effect,
    allocation_failure,
    insufficient_capacity,
    unsupported_scope
};

struct Status {
    StatusCode code = StatusCode::success;
    std::uint64_t index = 0;
    constexpr explicit operator bool() const noexcept {
        return code == StatusCode::success;
    }
};

} // namespace cellerator::geometry::dependency
