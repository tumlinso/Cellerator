#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/incremental.hh>

#include <cassert>
#include <iostream>
#include <map>

namespace m = cellerator::modeling;
template<class E, class F> void rejects(F f) {
    bool caught = false;
    try { f(); } catch (const E&) { caught = true; }
    assert(caught);
}

int main() {
    auto relation = std::make_shared<m::SparseRelation>(std::vector<m::RelationEdge>{
        {40, {{0,0,400}}}, {20, {{0,0,200}, {0,0,300}}},
        {10, {{0,0,100}, {0,0,200}, {0,0,200}}}}, 11, 12, 5);
    std::size_t calls = 0;
    bool fail = false;
    auto evaluator = [&](std::uint64_t id, const m::SourceReadSnapshot<double>& reads,
                          m::EvaluationEpochs epochs) {
        ++calls;
        if (fail && id == 20) throw std::runtime_error("callback failure");
        double sum = 0;
        for (std::size_t i = 0; i < reads.size(); ++i) sum += reads.at(i).value;
        return sum * (1 + epochs.parameters) + epochs.context;
    };
    m::IncrementalInteractions<double, double> cache(relation,
        {{{0,0,100}, 0, 1}, {{0,0,200}, 0, 2}, {{0,0,300}, 0, 3}, {{0,0,400}, 0, 4}}, evaluator);
    assert(cache.pending_invalid() == 3);
    rejects<std::logic_error>([&] { cache.result(10); });
    auto first = cache.step({}, {});
    assert(first.evaluated_ids == std::vector<std::uint64_t>({10,20,40}));
    assert(calls == 3 && cache.result(10) == 5 && cache.result(20) == 5);
    const auto old = cache.cached(10).record;
    assert(old->reads.size() == 3 && old->reads.at(1).source.entity_id == 200);

    calls = 0;
    auto second = cache.step({{{0,0,100}, 1, 8}}, {});
    assert(second.evaluated_ids == std::vector<std::uint64_t>({10}) && calls == 1);
    // Independent complete evaluation in original incidence order.
    std::map<std::uint64_t,double> complete_state{{100,8}, {200,2}, {300,3}, {400,4}};
    for (const auto& edge : relation->edges()) {
        double expected = 0;
        for (std::size_t i = 0; i < edge.participant_count; ++i)
            expected += complete_state.at(relation->participants()[edge.participant_begin+i].entity_id);
        assert(cache.result(edge.logical_edge_id) == expected);
    }
    assert(old->output == 5 && old->reads.at(0).value == 1);
    assert(cache.cached(10).record->reads.at(0).generation == 1);
    assert(cache.step({}, {}).evaluated_ids.empty());

    // An event is eligibility, not a claim that all source values changed.
    auto no_event = cache.event_eligibility({});
    cache.step({{{0,0,200}, 1, 7}}, {}, &no_event);
    assert(cache.pending_invalid() == 2 && !cache.cached(10).valid && cache.cached(40).valid);
    assert(cache.cached(10).record->output == 12); // explicit stale view
    rejects<std::logic_error>([&] { cache.result(10); });
    auto event = cache.event_eligibility({{0,0,100}});
    assert(cache.step({}, {}, &event).evaluated_ids == std::vector<std::uint64_t>({10}));
    assert(cache.result(10) == 22 && cache.pending_invalid() == 1);
    cache.step({}, {});
    assert(cache.result(20) == 10);

    // Parameter/context changes invalidate even an otherwise unchanged source.
    calls = 0;
    cache.step({}, {2,3}, &event);
    assert(calls == 1 && cache.result(10) == 69 && cache.pending_invalid() == 2);
    cache.step({}, {2,3});
    assert(cache.result(20) == 33 && cache.result(40) == 15);
    const auto generation = cache.step_generation();
    rejects<std::invalid_argument>([&] { cache.step({{{0,0,200},1,8}}, {2,3}); });
    rejects<std::invalid_argument>([&] { cache.step({{{0,0,200},2,8},{{0,0,200},3,9}}, {2,3}); });
    rejects<std::out_of_range>([&] { cache.step({{{0,0,999},1,8}}, {2,3}); });
    assert(cache.step_generation() == generation && cache.result(20) == 33);

    fail = true;
    rejects<std::runtime_error>([&] { cache.step({{{0,0,200},2,100}}, {3,4}); });
    assert(cache.step_generation() == generation && cache.source({0,0,200})->value == 7);
    assert(cache.epochs() == m::EvaluationEpochs({2,3}));
    assert(cache.result(10) == 69 && cache.pending_invalid() == 0);
    fail = false;
    cache.step({{{0,0,200},2,100}}, {3,4});
    assert(cache.result(10) == 836);

    // CSR reuses native posting deduplication despite duplicate incidences.
    const auto& incidence = cache.incidence();
    assert(incidence.offsets() == std::vector<std::uint64_t>({0,1,3,4,5}));
    assert(incidence.ordinals() == std::vector<std::uint64_t>({0,0,1,1,2}));
    assert(incidence.source_ordinal({0,0,300}) == 2);
    auto other = std::make_shared<m::SparseRelation>(*relation);
    m::ActiveWorklist wrong(other, {0});
    rejects<std::invalid_argument>([&] { cache.step({}, {3,4}, &wrong); });

    // Role is part of source identity. Numeric type is caller-selected.
    auto roles = std::make_shared<m::SparseRelation>(std::vector<m::RelationEdge>{
        {1, {{0,0,5}, {1,0,5}}}});
    m::IncrementalInteractions<int,int> typed(roles,
        {{{0,0,5},0,2},{{1,0,5},0,8}},
        [](auto, const auto& reads, auto) { return reads.at(0).value - reads.at(1).value; });
    typed.step({}, {});
    assert(typed.result(1) == -6);
    auto empty = std::make_shared<m::SparseRelation>(std::vector<m::RelationEdge>{});
    m::IncrementalInteractions<int,int> zero(empty, {}, [](auto, const auto&, auto) { return 0; });
    assert(zero.step({}, {}).evaluated_ids.empty());
    assert(zero.incidence().offsets() == std::vector<std::uint64_t>{0});
    std::cout << "incremental interactions: full recompute agreement; 1 vs 3 localized callbacks; rollback and eligibility passed\n";
}
