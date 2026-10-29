#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/execution/conditional/scheduler.hh>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace cellerator::execution::conditional;

template<class Exception, class F> void rejects(F&& callback) {
    bool rejected = false;
    try { callback(); } catch (const Exception&) { rejected = true; }
    assert(rejected);
}
ParameterBuffers parameters(float value) {
    auto storage = std::make_shared<std::vector<float>>(250, value);
    ParameterBuffers result;
    result.parameters = {storage, storage->data(), storage->size() * sizeof(float), "float32", "host"};
    return result;
}

int main() {
    // Independent arithmetic oracle: direct route wins small messages; two-hop
    // high-bandwidth route wins large messages. Reverse links are not invented.
    HardwareTopology routes({{0, 5000}, {1, 5000}, {2, 5000}},
        {{0, 2, 10, 0, false}, {0, 1, 1000, 1, false}, {1, 2, 1000, 1, false}});
    assert(routes.route(0, 2, 1).path == (std::vector<DeviceId>{0, 2}));
    const auto large = routes.route(0, 2, 1000);
    assert(large.path == (std::vector<DeviceId>{0, 1, 2}));
    assert(std::abs(large.seconds - 4.0) < 1e-12);
    rejects<UnreachableDevice>([&] { routes.route(2, 0, 1); });
    assert(routes.route(2, 0, 0).seconds == 0);
    rejects<std::invalid_argument>([] { HardwareTopology({{0, 10}}, {{0, 0, 0}}); });

    const ExpertKey exact{7, 3};
    HardwareTopology topology({{0, 10000, 1, false}, {1, 4096}, {2, 4096}},
        {{0, 1, 1000, 0, false}, {0, 2, 1, 0, false}, {1, 2, 1000, 0, true}});
    ResidencySnapshot snapshot{19, {{exact, 0, 1000, ResidencyState::ready, false}}, {}};
    PlacementPlanner planner(topology, {{exact, 1000, .01}}, snapshot);
    // Repeated, overlapping and signed routes retain original occurrence IDs.
    DispatchPlan work({{exact, {1, 50, 8}, .5}, {exact, {2, 60, 2}, -2},
                       {exact, {1, 50, 8}, 0}});
    auto mapping = [](const Origin& origin) { return origin.rank; };
    const auto plan = planner.plan(work, mapping, 4, 4, 22, {{{exact, 1}, 5}});
    assert(plan.decisions.size() == 2);
    assert(plan.decisions[0].demand.selection_indices == (std::vector<std::size_t>{0, 2}));
    assert(plan.decisions[0].chosen.strategy == ExecutionStrategy::replicate);
    assert(plan.decisions[1].chosen.strategy == ExecutionStrategy::remote_activations);
    assert(plan.decisions[1].chosen.parameter_source == 1);
    assert(plan.decisions[1].chosen.dependencies == (std::vector<std::size_t>{0}));
    assert(plan.proposed_replicas.size() == 1 && plan.proposed_replicas[0].state == ResidencyState::pending);
    assert(plan.projected_memory.at(1).bytes[0] == 1000);
    assert(plan.peak_total_bytes.at(1) == 1016);
    assert(plan.work->selections()[1].weight == -2 && plan.work->selections()[2].weight == 0);

    // Caller costs include setup/allocation/synchronization, changing strategy.
    ResidencySnapshot resident{0, {{exact, 1, 1000}, {exact, 2, 1000}}, {}};
    PlacementPlanner complete(topology, {{exact, 1000, 0}}, resident,
        [](const auto&, DeviceId device, auto) { return ExecutionCosts{0, 0, device == 1 ? 10.0 : 0, .1}; });
    DispatchPlan one({{exact, {1, 4, 5}, 1}});
    assert(complete.plan(one, mapping, 4, 4).decisions[0].chosen.strategy == ExecutionStrategy::remote_activations);

    // An old pinned revision and an in-flight unrelated copy both consume space.
    ResidencySnapshot full{0, {{exact, 0, 1000, ResidencyState::ready, false},
        {{7, 2}, 1, 2000, ResidencyState::ready, true, 3},
        {{9, 1}, 1, 2000, ResidencyState::pending, true, 1, 90, .2}}, {}};
    PlacementPlanner blocked(topology, {{exact, 1000, .01, {1}}}, full);
    rejects<PlanningError>([&] { blocked.plan(one, mapping, 4, 4); });
    // There is no substitution of ready revision 2 for selected revision 3.
    full.copies.erase(full.copies.begin());
    PlacementPlanner missing(topology, {{exact, 1000, .01}}, full);
    rejects<PlanningError>([&] { missing.plan(one, mapping, 4, 4); });
    full.copies.push_back({{15, 0}, 1, 200});
    rejects<std::invalid_argument>([&] { PlacementPlanner invalid(topology, {{exact, 1000}}, full); });

    // Temporary move is the stable tie winner without a reuse hint; no retained
    // replica is incorrectly added to the projected residency.
    auto temporary = planner.plan(one, mapping, 4, 4);
    assert(temporary.decisions[0].chosen.strategy == ExecutionStrategy::move_weights);
    assert(temporary.projected_memory.at(1).total() == 0);
    assert(temporary.peak_total_bytes.at(1) == 1008);

    ParameterStore store;
    store.register_initial(exact, parameters(3));
    rejects<std::invalid_argument>([&] { PlannedExecution stale(plan, store, 23, 19); });
    rejects<std::invalid_argument>([&] { PlannedExecution stale(plan, store, 22, 20); });
    rejects<std::length_error>([&] { PlannedExecution limited(plan, store, 22, 19, 1); });
    auto corrupt = plan;
    corrupt.decisions[1].demand.selection_indices = {0};
    rejects<std::invalid_argument>([&] { PlannedExecution invalid(corrupt, store, 22, 19); });

    PlannedExecution queue(plan, store, 22, 19);
    rejects<std::invalid_argument>([&] { queue.signal(999); });
    assert(store.leases(exact) == 1);
    store.publish(exact, {7, 4}, parameters(4));
    assert(store.retire(exact)); // Existing execution's exact lease survives.
    std::size_t calls = 0;
    auto callback = [&](std::size_t index, const DispatchDecision& decision,
                        const DispatchPlan& original, const ParameterLease& lease) {
        assert(lease.key() == exact && decision.demand.key == exact);
        assert(*static_cast<const float*>(lease.buffers().parameters.data) == 3);
        assert(original.size() == 3);
        ++calls;
        if (calls == 1) return TaskPoll::yield(123);
        assert(index == (calls == 2 ? 0u : 1u));
        return TaskPoll::done();
    };
    assert(queue.advance(callback, 10) == 1);
    assert(queue.states()[0] == TaskState::yielded && queue.completed_decisions() == 0);
    assert(queue.advance(callback, 10) == 0 && calls == 1);
    rejects<std::logic_error>([&] { queue.close(); });
    queue.signal(123);
    assert(queue.advance(callback, 1) == 1 && queue.completed_decisions() == 1);
    assert(queue.advance(callback, 1) == 1 && queue.done());
    queue.close();

    // Pending source readiness blocks even the first callback. Failure stops
    // subsequent submissions and retains leases until explicit backend drain.
    const ExpertKey next{7, 4};
    ResidencySnapshot pending{21, {{next, 1, 1000, ResidencyState::pending, true, 1, 456, .25}}, {}};
    PlacementPlanner pending_planner(topology, {{next, 1000}}, pending);
    DispatchPlan next_work({{next, {1, 0, 0}, 1}});
    auto pending_plan = pending_planner.plan(next_work, mapping, 4, 4);
    PlannedExecution waiting(pending_plan, store, 0, 21);
    assert(waiting.advance(callback) == 0);
    waiting.signal(456);
    rejects<std::runtime_error>([&] { waiting.advance([](auto, const auto&, const auto&, const auto&) -> TaskPoll {
        throw std::runtime_error("backend failed");
    }); });
    assert(waiting.states()[0] == TaskState::failed && store.leases(next) == 1);
    rejects<std::logic_error>([&] { waiting.close(); });
    waiting.close(true);
    assert(store.leases(next) == 0);

    DispatchPlan empty({});
    auto empty_plan = planner.plan(empty, mapping, 4, 4);
    PlannedExecution empty_queue(empty_plan, store, 0, 19, 0);
    assert(empty_queue.done());
    empty_queue.close();
    std::cout << "native topology, exact revision placement, capacity and event lifecycle passed\n";
}
