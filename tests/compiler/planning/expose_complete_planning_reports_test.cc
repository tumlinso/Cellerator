#include <Cellerator/compiler/planning/expose_complete_planning_reports_v1.hh>

#include <cassert>
#include <string>
#include <vector>

namespace planning = Cellerator::compiler::planning;

int main() {
    const std::vector<planning::planning_candidate_report_v1> candidates{
        {10u, true, 1200u, 7u, 7u, planning::planning_report_rejection_v1::none,
         false, true, planning::planning_report_selection_source_v1::user_edit,
         true, false},
        {20u, true, 1400u, 6u, 7u, planning::planning_report_rejection_v1::none,
         true, false, planning::planning_report_selection_source_v1::none,
         false, false},
        {30u, false, 900u, 7u, 7u,
         planning::planning_report_rejection_v1::incorrect,
         false, false, planning::planning_report_selection_source_v1::none,
         false, true},
    };
    const auto report = planning::expose_complete_planning_report_v1(candidates);
    assert(report);
    const std::string expected =
        "planning-report-v1\n"
        "candidate=10 coverage=exact cost_ns=1200 evidence=7/7 fresh=yes rejection=none dominated=no selected=yes source=user-edit forced=yes fallback=no\n"
        "candidate=20 coverage=exact cost_ns=1400 evidence=6/7 fresh=no rejection=none dominated=yes selected=no source=none forced=no fallback=no\n"
        "candidate=30 coverage=incomplete cost_ns=900 evidence=7/7 fresh=yes rejection=incorrect dominated=no selected=no source=none forced=no fallback=yes\n";
    assert(report.snapshot == expected);

    auto invalid = candidates;
    invalid[1].selected = true;
    invalid[1].selected_source = planning::planning_report_selection_source_v1::automatic;
    assert(planning::expose_complete_planning_report_v1(invalid).code ==
        planning::complete_planning_report_code_v1::invalid_selection);
}
