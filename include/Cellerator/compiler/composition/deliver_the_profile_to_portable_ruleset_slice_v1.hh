#pragma once
#include <Cellerator/compiler/composition/import_portable_schedule_ruleset_representation_v1.hh>
#include <Cellerator/compiler/composition/import_workload_family_representation_v1.hh>
#include <cstddef>
#include <string>
#include <vector>
namespace Cellerator::compiler::composition {
struct profile_ruleset_request_v1{workload_family_v1 family;std::vector<std::string> semantic_operations,discovered_atoms;std::size_t candidate_bound=0;double basis_cost=0,no_basis_cost=0;};
struct profile_ruleset_metrics_v1{std::size_t candidates_considered=0,exactly_certified=0,peak_records=0;double compiler_milliseconds=0;};
struct profile_ruleset_result_v1{bool valid=false,no_basis=false;portable_schedule_v1 schedule;profile_ruleset_metrics_v1 metrics;std::vector<std::string> diagnostics;};
[[nodiscard]] profile_ruleset_result_v1 compile_profile_to_ruleset_v1(const profile_ruleset_request_v1&);
} // namespace Cellerator::compiler::composition
