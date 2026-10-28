#pragma once
#include <cstdint>
#include <vector>
namespace cellerator::compiler::diagnostics::v1 {enum class candidate_source:std::uint8_t{compiler=0,user_edit,forced,fallback};struct candidate_record{std::uint64_t id=0,complete_cost_ns=0,evidence_age=0,uncertainty_ns=0;candidate_source source=candidate_source::compiler;bool legal=true,selected=false;std::uint32_t rejection_reason=0;};struct decision_report{std::vector<candidate_record> candidates;std::uint64_t legal_total_ns=0,selected_cost_ns=0,selected_id=0;bool valid=false;};[[nodiscard]] decision_report build_candidate_report(std::vector<candidate_record>);}
