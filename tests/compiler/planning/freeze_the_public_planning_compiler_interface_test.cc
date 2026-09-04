#include <Cellerator/compiler/planning/freeze_the_public_planning_compiler_interface_v1.hh>

#include <cassert>
#include <type_traits>

using namespace Cellerator::compiler::planning;

int main() {
    static_assert(std::is_trivially_copyable_v<public_planning_compiler_interface_v1>);
    static_assert(std::is_same_v<decltype(planning_provider_v1::provider_identity),
                                 std::uint64_t>);
    static_assert(std::is_pointer_v<planner_function_v1>);

    public_planning_compiler_interface_v1 interface;
    assert(freeze_public_planning_compiler_interface_v1(interface) ==
           public_planning_interface_status_v1::ready);

    planning_provider_v1 provider;
    provider.provider_identity = 41u;
    planner_portfolio_v1 planner;
    planning_cache_key_v1 cache;
    planning_candidate_report_v1 report;
    custom_candidate_registration_v1 custom;
    external_global_cost_query_v1 external_cost;
    candidate_edit_v1 force;
    assert(provider.provider_identity == 41u);
    assert(planner.exact.plan == nullptr);
    assert(cache.structure_epoch == 0u);
    assert(!report.selected);
    assert(custom.provided_protocols == 0u);
    assert(external_cost.deadline_nanoseconds == 0u);
    assert(force.mode == candidate_edit_mode_v1::offer);

    auto stale = interface;
    ++stale.dependencies.planning_ir;
    assert(freeze_public_planning_compiler_interface_v1(stale) ==
           public_planning_interface_status_v1::unsupported_planning_ir);

    auto incomplete = interface;
    incomplete.capabilities &= ~public_planning_external_cost_v1;
    assert(freeze_public_planning_compiler_interface_v1(incomplete) ==
           public_planning_interface_status_v1::incomplete_capabilities);
}
