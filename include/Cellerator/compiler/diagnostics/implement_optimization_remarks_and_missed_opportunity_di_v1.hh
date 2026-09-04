#pragma once
#include <cstdint>
#include <string_view>
#include <vector>
namespace cellerator::compiler::diagnostics::v1 {enum class optimization_remark:std::uint8_t{persistence_assumption=0,missing_profile_hint,expensive_canonicalization,avoidable_packing,unshared_order,uncertain_branch};struct remark_record{optimization_remark kind=optimization_remark::persistence_assumption;std::string_view stable_code;};[[nodiscard]] std::vector<remark_record> emit_optimization_remarks(std::uint32_t observed_mask,bool enabled,std::uint32_t suppressed_mask);}
