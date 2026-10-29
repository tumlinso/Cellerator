#pragma once
#include <cstdint>
#include <type_traits>
namespace Cellerator::compiler::migration {
struct portable_schedule_identity_v1{std::uint64_t ruleset_high=0,ruleset_low=0,profile_high=0,profile_low=0,exact_coverage_high=0,exact_coverage_low=0,realization_family=0,target_capability_class=0;};
[[nodiscard]] constexpr bool valid(portable_schedule_identity_v1 i)noexcept{return i.ruleset_high&&i.ruleset_low&&i.profile_high&&i.profile_low&&i.exact_coverage_high&&i.exact_coverage_low&&i.realization_family&&i.target_capability_class;}
static_assert(std::is_trivially_copyable_v<portable_schedule_identity_v1>);
} // namespace Cellerator::compiler::migration
