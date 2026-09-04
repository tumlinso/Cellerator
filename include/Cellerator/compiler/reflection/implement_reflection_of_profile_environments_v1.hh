#pragma once
#include <Cellerator/compiler/reflection/freeze_the_compile_time_ir_handle_model_v1.hh>
#include <optional>
#include <string>
#include <vector>
namespace cellerator::compiler::reflection::v1 {
struct profile_evidence_summary_v1{std::uint64_t samples=0;double confidence=0;};
struct reflected_profile_state_v1{std::string name;ir_handle_v1 handle{};profile_evidence_summary_v1 evidence{};std::vector<std::string>mutation_expectations,joins,unknown_dimensions;std::string branch_condition;};
struct reflected_profile_environment_v1{std::vector<reflected_profile_state_v1>states;std::string selected;};
[[nodiscard]] std::optional<reflected_profile_state_v1> query_profile_state_v1(const reflected_profile_environment_v1&,const std::string&,const std::string&branch={});
[[nodiscard]] bool validate_profile_environment_v1(const reflected_profile_environment_v1&)noexcept;
}
