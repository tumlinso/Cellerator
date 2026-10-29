#pragma once
#include <cstdint>
namespace Cellerator::compiler::migration {
enum class basis_outcome_v1 : std::uint8_t { selected=1, redundant_bases, no_basis };
struct representative_profile_basis_input_v1 {std::uint64_t profile_identity=0,profile_generation=0,workload_family_identity=0,structure_epoch=0;};
struct portable_ruleset_basis_output_v1 {std::uint64_t ruleset_identity=0,basis_identity=0,input_profile_identity=0,input_profile_generation=0;basis_outcome_v1 outcome=basis_outcome_v1::no_basis;};
struct complete_basis_cost_v1 {std::uint64_t build=0,storage=0,materialization=0,execution=0,canonicalization=0,invalidation=0;};
[[nodiscard]] constexpr std::uint64_t total(complete_basis_cost_v1 c) noexcept{return c.build+c.storage+c.materialization+c.execution+c.canonicalization+c.invalidation;}
[[nodiscard]] constexpr bool traceable(representative_profile_basis_input_v1 i,portable_ruleset_basis_output_v1 o) noexcept{return i.profile_identity&&i.profile_generation&&i.workload_family_identity&&i.structure_epoch&&o.ruleset_identity&&o.input_profile_identity==i.profile_identity&&o.input_profile_generation==i.profile_generation&&(o.outcome==basis_outcome_v1::no_basis||o.basis_identity);}
} // namespace Cellerator::compiler::migration
