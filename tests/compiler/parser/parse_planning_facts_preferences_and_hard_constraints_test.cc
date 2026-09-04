#include <Cellerator/compiler/frontend/parser/parse_planning_facts_preferences_and_hard_constraints_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::frontend::parser;

int main() {
    const auto parsed = parse_planning_directives_v1(R"(
given profile pbmc;
given persists(structure, trajectory);
prefer minimum_latency;
offer candidate tiled;
force operation(transfer): candidate exact_sm70;
require operation(transfer): budget(transient, 4096);
require target_class(volta);
force realization frozen_cover;
)");
    assert(parsed.accepted());
    assert(parsed.directives.size() == 8);
    assert(parsed.directives[0].subject == planning_subject_v1::profile);
    assert(parsed.directives[1].subject == planning_subject_v1::persistence);
    assert(parsed.directives[4].operation_scope == "transfer");
    assert(parsed.directives[4].authority == planning_authority_v1::forced_selection);
    assert(parsed.directives[5].authority == planning_authority_v1::hard_requirement);
    assert(static_cast<int>(parsed.directives[5].authority)
           > static_cast<int>(parsed.directives[4].authority));

    const auto forced_conflict = parse_planning_directives_v1(
        "force candidate a; force candidate b;");
    assert(!forced_conflict.accepted());
    assert(forced_conflict.diagnostics.back().message == "conflicting forced selections");

    const auto hard_conflict = parse_planning_directives_v1(
        "force candidate a; require exclude candidate a;");
    assert(!hard_conflict.accepted());
    assert(hard_conflict.diagnostics.back().message
           == "hard requirement excludes forced candidate");
}
