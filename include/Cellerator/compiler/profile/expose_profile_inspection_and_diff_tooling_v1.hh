#pragma once
#include <cstddef>
#include <cstdint>
namespace cellerator::compiler::profile::v1 {
struct profile_inspection_state_v1{std::uint64_t identity_low=0,identity_high=0,evidence_revision=0;double confidence=0,minimum=0,maximum=0;std::uint32_t expected_mutations=0,missing_evidence=0;};
enum class profile_inspection_format_v1:std::uint8_t{human=0,machine_json};
enum class profile_inspection_status_v1:std::uint8_t{ok=0,invalid_argument,insufficient_capacity};
profile_inspection_status_v1 dump_profile_summary_v1(const profile_inspection_state_v1&,profile_inspection_format_v1,char*,std::size_t,std::size_t*) noexcept;
profile_inspection_status_v1 diff_profile_states_v1(const profile_inspection_state_v1&,const profile_inspection_state_v1&,profile_inspection_format_v1,char*,std::size_t,std::size_t*) noexcept;
}  // namespace cellerator::compiler::profile::v1
