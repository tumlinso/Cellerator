// Runnable C++ model/cell example and focused integration check. No Python,
// tensor framework, biological ontology, scheduler or device is involved.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/dynamical.hh>
#include <Cellerator/modeling/value_field.hh>

#include <cassert>
#include <iostream>
#include <string>

namespace m = cellerator::modeling;

struct State {
    m::ValueField<float> field;
    std::string phase;
};
struct Parameters {
    std::shared_ptr<const m::IncidencePlan> plan;
    float coupling;
    float threshold;
};
struct Inputs { float drive = 0; };
struct Event { std::uint64_t edge_id; double time; float signal; };
using Context = m::StepContext<Parameters, Inputs>;
using Result = m::StepResult<State, std::vector<Event>>;

int main() {
    // Overlapping directed hyperedges; edge 10 reads entity 200 twice. Physical
    // state order [300,100,200] differs from both identity and relation order.
    auto relation = std::make_shared<m::SparseRelation>(std::vector<m::RelationEdge>{
        {20, {{0,0,100}, {0,0,200}, {1,0,300}}},
        {10, {{0,0,200}, {0,0,200}, {1,0,100}}}});
    m::EntityLayout layout({300,100,200});
    auto plan = std::make_shared<m::IncidencePlan>(m::ActivityState(relation).worklist(), layout);
    auto parameters = std::make_shared<Parameters>(Parameters{plan, 1.0f, 3.5f});

    auto transition = [](const State& state, const Context& context) -> Result {
        if (!context.parameters) throw std::invalid_argument("this transition requires parameters");
        const auto& parameters = *context.parameters;
        const auto& field = state.field;
        const auto signals = m::joint_reduce<float,float,float>(
            *parameters.plan, field.values(), field.features(), m::JointReduction::sum, 0);
        auto increments = m::scatter_add(*parameters.plan, signals, field.features(), 1);
        auto values = field.values(); // State chooses its own ownership/copy policy.
        const float drive = context.inputs ? context.inputs->drive : 0.0f;
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] += static_cast<float>(context.dt) * (parameters.coupling * increments[i] + drive);
        State next = state;
        next.field.replace(std::move(values));
        next.phase = "advanced";
        std::vector<Event> events;
        const auto ids = parameters.plan->worklist().edge_ids();
        for (std::size_t i = 0; i < signals.size(); ++i)
            if (signals[i] > parameters.threshold)
                events.push_back({ids[i], context.time + context.dt, signals[i]});
        return {std::move(next), std::move(events)};
    };
    auto observation = [](const State& state, const Context&) {
        float total = 0;
        for (auto value : state.field.values()) total += value;
        return total;
    };
    auto model = m::make_dynamical_model<State, Context>(transition, parameters, observation);
    auto first = m::make_dynamical_cell(model, State{m::ValueField<float>(layout, 1, {3,1,2}), "initial"},
                              std::string("cell-a"));
    auto second = m::make_dynamical_cell(model, State{m::ValueField<float>(layout, 1, {0,0,1}), "initial"},
                               std::string("cell-b"));
    assert(first.model() == second.model()); // Behavior/parameters are shared.

    auto initial_handle = first.state_handle();
    auto result = first.advance(0.5);
    assert(first.id() == "cell-a" && first.time() == 0.5 && first.value_generation() == 1);
    assert((first.state().field.values() == std::vector<float>{4.5f, 3.0f, 2.0f}));
    assert(result->events.size() == 1 && result->events[0].edge_id == 10);
    assert(result->events[0].time == 0.5 && result->events[0].signal == 4.0f);
    assert(first.observe() == 9.5f);
    assert(initial_handle->field.values()[0] == 3 && initial_handle->phase == "initial");
    assert(second.state().field.values()[0] == 0 && second.time() == 0);

    // Publish a real topology edit while retaining the same cell/model. Disabled
    // activity and old results/plans survive by stable identity and ownership.
    m::DynamicRelation dynamic(relation);
    dynamic.stage_remove(10);
    dynamic.stage_add(30, {{0,0,300}, {1,0,200}});
    const auto changed = dynamic.publish();
    const auto rebased = m::ActivityState(relation).rebase(changed);
    auto new_plan = std::make_shared<m::IncidencePlan>(rebased.worklist(), layout);
    model->set_parameters(std::make_shared<Parameters>(Parameters{new_plan, 1.0f, 3.5f}));
    auto after_edit = first.advance(0.5);
    assert(first.time() == 1.0 && first.state().field.generation().value == 2);
    assert((first.state().field.values() == std::vector<float>{7.0f, 3.0f, 4.25f}));
    assert(result->state.field.values()[0] == 4.5f); // Pinned previous result.
    assert(parameters->plan->worklist().relation()->structure_epoch() == 1);
    assert(new_plan->worklist().relation()->structure_epoch() == 2);
    assert(after_edit->events.size() == 2);

    // Shared model changes do not share cell state. Explicit inputs belong only
    // to this invocation; the same updated topology also serves the other cell.
    Inputs input{2};
    Context context;
    context.dt = 0.25;
    context.inputs = &input;
    second.advance(context);
    assert(second.time() == 0.25 && first.time() == 1.0);
    assert((second.state().field.values() == std::vector<float>{0.75f, 0.5f, 1.5f}));
    std::cout << first.id() << " time=" << first.time() << " observed=" << first.observe()
              << " structure_epoch=" << new_plan->worklist().relation()->structure_epoch() << '\n';
}
