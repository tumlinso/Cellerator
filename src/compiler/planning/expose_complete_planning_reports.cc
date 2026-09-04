#include <Cellerator/compiler/planning/expose_complete_planning_reports_v1.hh>

#include <sstream>
#include <unordered_set>

namespace Cellerator::compiler::planning {
namespace {

const char* rejection_name(planning_report_rejection_v1 value) noexcept {
    switch (value) {
    case planning_report_rejection_v1::none: return "none";
    case planning_report_rejection_v1::invalid: return "invalid";
    case planning_report_rejection_v1::incorrect: return "incorrect";
    case planning_report_rejection_v1::resource_limit: return "resource-limit";
    case planning_report_rejection_v1::unavailable: return "unavailable";
    }
    return "unknown";
}

const char* source_name(planning_report_selection_source_v1 value) noexcept {
    switch (value) {
    case planning_report_selection_source_v1::none: return "none";
    case planning_report_selection_source_v1::automatic: return "automatic";
    case planning_report_selection_source_v1::source_edit: return "source-edit";
    case planning_report_selection_source_v1::pipeline_edit: return "pipeline-edit";
    case planning_report_selection_source_v1::user_edit: return "user-edit";
    case planning_report_selection_source_v1::external: return "external";
    case planning_report_selection_source_v1::cache: return "cache";
    case planning_report_selection_source_v1::fallback: return "fallback";
    }
    return "unknown";
}

}  // namespace

complete_planning_report_v1 expose_complete_planning_report_v1(
    const std::vector<planning_candidate_report_v1>& candidates) {
    complete_planning_report_v1 result{};
    std::unordered_set<std::uint64_t> identities;
    std::uint64_t selected_count = 0u;
    std::ostringstream out;
    out << "planning-report-v1\n";
    for (const auto& candidate : candidates) {
        if (candidate.candidate_identity == 0u || candidate.current_evidence_revision == 0u) {
            return result;
        }
        if (!identities.insert(candidate.candidate_identity).second) {
            result.code = complete_planning_report_code_v1::duplicate_candidate;
            return result;
        }
        if (candidate.selected) {
            ++selected_count;
            if (candidate.selected_source == planning_report_selection_source_v1::none ||
                candidate.rejection != planning_report_rejection_v1::none) {
                result.code = complete_planning_report_code_v1::invalid_selection;
                return result;
            }
        }
        const bool fresh = candidate.evidence_revision == candidate.current_evidence_revision;
        out << "candidate=" << candidate.candidate_identity
            << " coverage=" << (candidate.exact_coverage ? "exact" : "incomplete")
            << " cost_ns=" << candidate.complete_cost_nanoseconds
            << " evidence=" << candidate.evidence_revision << "/"
            << candidate.current_evidence_revision
            << " fresh=" << (fresh ? "yes" : "no")
            << " rejection=" << rejection_name(candidate.rejection)
            << " dominated=" << (candidate.dominated ? "yes" : "no")
            << " selected=" << (candidate.selected ? "yes" : "no")
            << " source=" << source_name(candidate.selected_source)
            << " forced=" << (candidate.forced_edit ? "yes" : "no")
            << " fallback=" << (candidate.fallback ? "yes" : "no") << '\n';
    }
    if (selected_count != 1u) {
        result.code = complete_planning_report_code_v1::invalid_selection;
        return result;
    }
    result.code = complete_planning_report_code_v1::ok;
    result.snapshot = out.str();
    return result;
}

}  // namespace Cellerator::compiler::planning
