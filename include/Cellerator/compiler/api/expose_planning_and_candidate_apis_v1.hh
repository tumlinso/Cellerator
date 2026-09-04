#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::api::v1 {
struct candidate_v1{std::uint64_t id=0;std::string provider;std::string decomposition;double cost=0;std::string evidence;std::string ruleset;};
using candidate_provider_v1=bool(*)(std::vector<candidate_v1>&,void*) noexcept;
using planner_v1=std::uint64_t(*)(const std::vector<candidate_v1>&,void*) noexcept;
struct planning_request_v1{std::vector<candidate_provider_v1> providers;planner_v1 planner=nullptr;void* user_data=nullptr;std::uint64_t forced_candidate=0;};
struct planning_report_v1{std::vector<candidate_v1> discovered;std::uint64_t selected=0;std::string selected_ruleset;bool forced=false;};
[[nodiscard]] bool plan_candidates_v1(const planning_request_v1&,planning_report_v1&) noexcept;
}
