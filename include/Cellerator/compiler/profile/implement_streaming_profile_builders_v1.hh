#pragma once
#include <cstdint>
namespace cellerator::compiler::profile::v1 {
struct streaming_profile_policy_v1 { std::uint64_t memory_budget_bytes=0; std::uint32_t histogram_bins=0,top_l=0,sketch_slots=0; bool exact_small_mode=false; };
struct streaming_profile_workspace_v1 { std::uint64_t* histogram=nullptr; double* top_values=nullptr; std::uint64_t* sketch=nullptr; double* exact_values=nullptr; std::uint64_t capacity_bytes=0; };
struct streaming_profile_builder_v1 { streaming_profile_policy_v1 policy{}; streaming_profile_workspace_v1 workspace{}; std::uint64_t count=0,exact_count=0; double minimum=0,maximum=0,mean=0,m2=0; };
struct streaming_profile_result_v1 { std::uint64_t count=0,workspace_bytes=0; double mean=0,variance=0,estimated_distinct=0,estimator_error_bound=0; };
enum class streaming_profile_status_v1 : std::uint8_t { ok=0,invalid_argument,insufficient_budget,capacity_exceeded };
std::uint64_t streaming_profile_workspace_bytes_v1(const streaming_profile_policy_v1&) noexcept;
streaming_profile_status_v1 initialize_streaming_profile_builder_v1(const streaming_profile_policy_v1&,streaming_profile_workspace_v1,streaming_profile_builder_v1*) noexcept;
streaming_profile_status_v1 update_streaming_profile_builder_v1(streaming_profile_builder_v1*,const double*,std::uint64_t) noexcept;
streaming_profile_status_v1 finalize_streaming_profile_builder_v1(const streaming_profile_builder_v1&,streaming_profile_result_v1*) noexcept;
streaming_profile_status_v1 count_scan_fill_offsets_v1(const std::uint32_t*,std::uint32_t,std::uint64_t*,std::uint32_t,std::uint64_t*) noexcept;
}  // namespace cellerator::compiler::profile::v1
