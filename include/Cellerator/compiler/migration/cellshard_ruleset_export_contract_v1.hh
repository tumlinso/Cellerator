#pragma once
#include <cstdint>
#include <type_traits>
namespace Cellerator::compiler::migration {
inline constexpr std::uint32_t ruleset_export_schema_v1=1;
struct ruleset_export_v1{std::uint32_t schema=ruleset_export_schema_v1,record_bytes=sizeof(ruleset_export_v1);std::uint64_t ruleset_high=0,ruleset_low=0,profile_high=0,profile_low=0,exact_coverage_high=0,exact_coverage_low=0,realization_requirements_high=0,realization_requirements_low=0,structure_generation=0;};
[[nodiscard]] constexpr bool valid(ruleset_export_v1 r)noexcept{return r.schema==ruleset_export_schema_v1&&r.record_bytes==sizeof(r)&&r.ruleset_high&&r.ruleset_low&&r.profile_high&&r.profile_low&&r.exact_coverage_high&&r.exact_coverage_low&&r.realization_requirements_high&&r.realization_requirements_low&&r.structure_generation;}
static_assert(std::is_trivially_copyable_v<ruleset_export_v1>);
} // namespace Cellerator::compiler::migration
