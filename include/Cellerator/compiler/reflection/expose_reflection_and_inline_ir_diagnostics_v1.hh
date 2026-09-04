#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::reflection::v1 {
enum class reflection_diagnostic_code_v1:std::uint8_t{phase_unavailable=1,type_mismatch,stale_handle,capture_effect,unknown_extension,validation_mode,compiler_invalidation};
enum invalidation_flag_v1:std::uint32_t{invalidate_none_v1=0,invalidate_semantic_v1=1,invalidate_profile_v1=2,invalidate_planning_v1=4,invalidate_realization_v1=8};
struct reflection_diagnostic_v1{reflection_diagnostic_code_v1 code=reflection_diagnostic_code_v1::phase_unavailable;std::string source,message,expected,observed;std::uint32_t invalidations=0;bool warning=false;};
[[nodiscard]] std::string format_reflection_diagnostic_v1(const reflection_diagnostic_v1&);
[[nodiscard]] std::string serialize_reflection_diagnostics_v1(const std::vector<reflection_diagnostic_v1>&);
}
