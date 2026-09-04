#pragma once
#include <Cellerator/compiler/lto/freeze_the_ceir_companion_object_artifact_contract_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::lto::v1 {
struct merge_profile_state_v1{artifact_identity_v1 identity{},evidence{};std::string name,biological_semantics;std::uint64_t revision=0;};
struct merged_profile_environment_v1{std::vector<merge_profile_state_v1>states;std::vector<std::string>diagnostics;};
enum class profile_merge_status_v1:std::uint8_t{valid=0,invalid_state,semantic_conflict,too_many_alternatives};
[[nodiscard]] profile_merge_status_v1 merge_profile_environments_v1(const std::vector<std::vector<merge_profile_state_v1>>&,std::size_t alternative_limit,merged_profile_environment_v1*)noexcept;
}
