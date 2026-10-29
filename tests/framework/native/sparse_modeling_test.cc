#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/value_field.hh>

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace m = cellerator::modeling;

template<class Exception, class Callable>
void rejects(Callable&& callable) {
    bool caught = false;
    try { callable(); } catch (const Exception&) { caught = true; }
    assert(caught);
}

int main() {
    // Deliberately unsorted edge IDs; repeated source incidence is meaningful.
    auto relation = std::make_shared<m::SparseRelation>(std::vector<m::RelationEdge>{
        {20, {{0, 0, 100}, {0, 0, 200}, {1, 0, 300}}},
        {10, {{0, 0, 200}, {0, 0, 200}, {1, 0, 100}}}}, 7, 9, 4);
    assert(relation->edges()[0].logical_edge_id == 10);
    assert((relation->query({{0, 0, 200}}) == std::vector<std::size_t>{0, 1}));
    assert((relation->query({{0, 0, 100}, {1, 0, 300}}, true) == std::vector<std::size_t>{1}));
    assert(relation->query({{0, 0, 999}}).empty());
    assert(relation->query({}).empty());

    m::ActivityState active(relation);
    auto old_worklist = active.worklist();
    active.set_active({10}, false);
    assert(active.generation() == 1);
    active.set_active({10}, false);
    assert(active.generation() == 1);
    assert((active.query({{0, 0, 200}}).edge_ids() == std::vector<std::uint64_t>{20}));
    rejects<std::out_of_range>([&] { active.set_active({20, 999}, false); });
    assert(active.worklist().size() == 1); // Failed update did not disable edge 20.
    assert(old_worklist.size() == 2 && old_worklist.activity_generation() == 0);

    // Physical storage order is unrelated to entity or edge ID order.
    m::EntityLayout layout({300, 100, 200});
    m::IncidencePlan plan(old_worklist, layout);
    std::vector<float> values{3, 30, 1, 10, 2, 20};
    assert((m::gather(plan, values, 2) == std::vector<float>{2,20, 2,20, 1,10, 1,10, 2,20, 3,30}));
    assert((m::joint_reduce<float,double,double>(plan, values, 2, m::JointReduction::sum, 0)
            == std::vector<double>{4,40, 3,30}));
    assert((m::joint_reduce(plan, values, 2, m::JointReduction::mean, 0)
            == std::vector<float>{2,20, 1.5f,15}));
    assert((m::joint_reduce(plan, values, 2, m::JointReduction::product, 0)
            == std::vector<float>{4,400, 2,200}));
    assert((m::scatter_add(plan, std::vector<float>{4,40, 3,30}, 2, 1)
            == std::vector<float>{3,30, 4,40, 0,0}));
    assert((m::joint_reduce_scatter(plan, values, 2, m::JointReduction::sum, 0, 1)
            == std::vector<float>{3,30, 4,40, 0,0}));
    // Every repeated output incidence contributes to scatter.
    assert((m::scatter_add(plan, std::vector<float>{1, 2}, 1, 0)
            == std::vector<float>{0, 2, 4}));
    rejects<std::invalid_argument>([&] { m::joint_reduce(plan, values, 2, m::JointReduction::mean, 99); });
    rejects<std::invalid_argument>([&] { m::gather(plan, values, 1); });
    rejects<std::out_of_range>([&] { m::IncidencePlan invalid(old_worklist, m::EntityLayout({100})); });

    m::ValueField<float> field(layout, 2, values);
    field.update({200}, {8, 80});
    assert(field.generation().value == 1 && field.values()[4] == 8);
    rejects<std::out_of_range>([&] { field.update({100, 999}, {7,70, 9,90}); });
    assert(field.values()[2] == 1 && field.generation().value == 1);
    rejects<std::invalid_argument>([&] { field.update({100, 100}, {7,70, 9,90}); });
    assert(field.generation().value == 1);

    m::DynamicRelation dynamic(relation);
    dynamic.stage_add(30, {{0, 0, 300}, {1, 0, 100}});
    dynamic.stage_remove(10);
    assert(dynamic.snapshot()->size() == 2 && dynamic.snapshot()->structure_epoch() == 4);
    rejects<std::invalid_argument>([&] { dynamic.publish(4); });
    auto replacement = dynamic.publish();
    assert(replacement->structure_epoch() == 5 && replacement->size() == 2);
    active.set_active({20}, false);
    const auto rebased = active.rebase(replacement);
    assert((rebased.worklist().edge_ids() == std::vector<std::uint64_t>{30}));
    assert(rebased.generation() == active.generation() + 1);
    assert(active.rebase(replacement, false).worklist().size() == 0);
    rejects<std::invalid_argument>([&] { active.rebase(relation); });
    assert((replacement->query({{0, 0, 300}}) == std::vector<std::size_t>{1}));
    assert(relation->structure_epoch() == 4 && plan.worklist().edge_ids()[0] == 10);
    assert(m::gather(plan, values, 2).size() == 12); // Old plan keeps its snapshot alive.
    rejects<std::invalid_argument>([&] { dynamic.stage_add(10, {{0, 0, 300}}); });
    assert(dynamic.publish() == replacement); // No pending edit is a no-op.

    auto empty = std::make_shared<m::SparseRelation>(std::vector<m::RelationEdge>{});
    m::IncidencePlan empty_plan(m::ActivityState(empty).worklist(), m::EntityLayout({}));
    assert(m::gather(empty_plan, std::vector<double>{}).empty());
    assert(m::joint_reduce(empty_plan, std::vector<double>{}).empty());
    assert(m::scatter_add(empty_plan, std::vector<double>{}).empty());
    rejects<std::invalid_argument>([] {
        m::SparseRelation duplicate({{1, {{0, 0, 1}}}, {1, {{0, 0, 2}}}});
    });
    rejects<std::invalid_argument>([&] { m::ActiveWorklist invalid(relation, {0, 0}); });

    // Reduction nonfinite semantics: a NaN cannot vanish through min/max.
    auto nan_values = values;
    nan_values[4] = std::numeric_limits<float>::quiet_NaN();
    assert(std::isnan(m::joint_reduce(plan, nan_values, 2, m::JointReduction::minimum, 0)[0]));
    return 0;
}
