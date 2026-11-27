// Two independent native states share one prepared relation.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/execution/preparation/dependency/incidence.hh>
#include <Cellerator/geometry/dependency/pattern_owner.hh>
#include <Cellerator/layout/entity_layout.hh>

#include <assert.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <memory>
#include <vector>

namespace g = cellerator::geometry::dependency;
namespace e = cellerator::execution;

namespace {

struct State {
    const float* values;
    size_t size;
    double time = 0;
    uint64_t generation = 0;
};

struct Parameters {
    const e::PreparedIncidence& plan;
    float coupling;
    float threshold;
};

struct Event { uint64_t edge_id; double time; float signal; };
struct Scratch { float* signals; size_t capacity; };
struct Output {
    float* values;
    size_t value_capacity;
    Event* events;
    size_t event_capacity;
};

enum class Status {
    success, invalid_clock, generation_overflow, invalid_state,
    value_capacity, signal_capacity, event_capacity
};

struct Result {
    Status status;
    State state{};
    const Event* events = nullptr;
    size_t event_count = 0;
    size_t required = 0;
};

// All buffers are disjoint. Only scratch may be overwritten on failure.
// A successful result borrows the output until the caller reuses its storage.
Result step(const Parameters& params, State current, double dt, float drive,
            Scratch scratch, Output out, const double* endpoint = nullptr) noexcept {
    if (!__builtin_isfinite(current.time) || !__builtin_isfinite(dt))
        return {Status::invalid_clock};
    const double time = endpoint ? *endpoint : current.time + dt;
    if (!__builtin_isfinite(time)) return {Status::invalid_clock};
    if (current.generation == UINT64_MAX) return {Status::generation_overflow};

    const auto& plan = params.plan;
    const size_t n = plan.layout().size(), ne = plan.edge_count();
    if (current.size != n || (n && !current.values)) return {Status::invalid_state};
    if (n && (!out.values || out.value_capacity < n))
        return {.status = Status::value_capacity, .required = n};
    if (ne && (!scratch.signals || scratch.capacity < ne))
        return {.status = Status::signal_capacity, .required = ne};

    const size_t* offsets = plan.offsets().data();
    const size_t* slots = plan.slots().data();
    const uint32_t* roles = plan.roles().data();
    size_t event_count = 0;
    for (size_t edge = 0; edge < ne; ++edge) {
        float signal = 0;
        for (size_t i = offsets[edge]; i < offsets[edge + 1]; ++i)
            if (roles[i] == 0) signal += current.values[slots[i]];
        scratch.signals[edge] = signal;
        event_count += signal > params.threshold;
    }
    if (event_count && (!out.events || out.event_capacity < event_count))
        return {.status = Status::event_capacity, .required = event_count};

    const auto& work = plan.worklist();
    const auto* edges = work.pattern()->view().edges;
    const size_t* ordinals = work.ordinals().data();
    for (size_t i = 0; i < n; ++i) out.values[i] = 0;
    size_t emitted = 0;
    for (size_t edge = 0; edge < ne; ++edge) {
        const float signal = scratch.signals[edge];
        for (size_t i = offsets[edge]; i < offsets[edge + 1]; ++i)
            if (roles[i] == 1) out.values[slots[i]] += signal;
        if (signal > params.threshold)
            out.events[emitted++] = {edges[ordinals[edge]].logical_edge_id, time, signal};
    }
    for (size_t i = 0; i < n; ++i)
        out.values[i] = current.values[i] + static_cast<float>(dt)*(params.coupling*out.values[i] + drive);
    return {Status::success, {out.values, n, time, current.generation + 1}, out.events, event_count};
}

float observe(State state) noexcept {
    float total = 0;
    for (size_t i = 0; i < state.size; ++i) total += state.values[i];
    return total;
}

void expect_values(State state, float a, float b, float c) {
    assert(state.size == 3 && state.values[0] == a && state.values[1] == b && state.values[2] == c);
}

} // namespace

int main() {
    // Edge 10 reads entity 200 twice. Physical order [300,100,200] differs
    // from both identity and relation order.
    const auto relation = std::make_shared<g::ExplicitPattern>(std::vector<g::PatternEdge>{
        {20, {{0,0,100}, {0,0,200}, {1,0,300}}},
        {10, {{0,0,200}, {0,0,200}, {1,0,100}}}});
    const cellerator::layout::EntityLayout layout({300,100,200});
    const e::ActivityState activity(relation);
    const e::PreparedIncidence plan(activity.worklist(), layout);
    const Parameters params{plan, 1.0f, 3.5f};

    const float initial_a[]{3, 1, 2}, initial_b[]{0, 0, 1};
    State first{initial_a, 3}, second{initial_b, 3};
    const State initial = first;
    float values_a[3], values_edit[3], values_b[3], signals[2];
    Event events_a[2], events_edit[2], events_b[2];
    const Scratch scratch{signals, 2};

    const auto result = step(params, first, 0.5, 0, scratch, {values_a, 3, events_a, 2});
    assert(result.status == Status::success);
    first = result.state;
    assert(first.time == 0.5 && first.generation == 1);
    expect_values(first, 4.5f, 3.0f, 2.0f);
    assert(result.event_count == 1 && result.events[0].edge_id == 10);
    assert(result.events[0].time == 0.5 && result.events[0].signal == 4.0f);
    assert(observe(first) == 9.5f);
    expect_values(initial, 3, 1, 2);
    assert(initial.time == 0 && initial.generation == 0);
    expect_values(second, 0, 0, 1);
    assert(second.time == 0 && second.generation == 0);

    // Publish new topology and prepare its physical slots once. The old
    // state, result events, and plan remain in their caller-owned buffers.
    g::PatternOwner owner(relation);
    owner.stage_remove(10);
    owner.stage_add(30, {{0,0,300}, {1,0,200}});
    const auto changed = owner.publish();
    const e::PreparedIncidence new_plan(activity.rebase(changed).worklist(), layout);
    const Parameters updated{new_plan, params.coupling, params.threshold};

    const auto edited = step(updated, first, 0.5, 0, scratch, {values_edit, 3, events_edit, 2});
    assert(edited.status == Status::success);
    first = edited.state;
    assert(first.time == 1.0 && first.generation == 2);
    expect_values(first, 7.0f, 3.0f, 4.25f);
    expect_values(result.state, 4.5f, 3.0f, 2.0f);
    assert(observe(first) == 14.25f && observe(result.state) == 9.5f);
    assert(result.event_count == 1 && result.events[0].edge_id == 10);
    assert(plan.worklist().pattern()->structure_epoch() == 1 && changed->structure_epoch() == 2);
    assert(edited.event_count == 2);
    assert(edited.events[0].edge_id == 20 && edited.events[0].time == 1.0 && edited.events[0].signal == 5.0f);
    assert(edited.events[1].edge_id == 30 && edited.events[1].time == 1.0 && edited.events[1].signal == 4.5f);

    const auto driven = step(updated, second, 0.25, 2, scratch, {values_b, 3, events_b, 2});
    assert(driven.status == Status::success && driven.event_count == 0);
    second = driven.state;
    assert(second.time == 0.25 && second.generation == 1 && first.time == 1.0);
    expect_values(second, 0.75f, 0.5f, 1.5f);

    float trial_values[3]{-1, -2, -3};
    Event trial_events[2]{{99, -1, -2}, {100, -2, -3}};
    const Output trial{trial_values, 3, trial_events, 2};
    const double infinity = __builtin_inf(), nan = __builtin_nan("");
    assert(step(updated, first, nan, 0, scratch, trial).status == Status::invalid_clock);
    assert(step(updated, first, 0.5, 0, scratch, trial, &infinity).status == Status::invalid_clock);
    assert(step(updated, {first.values, 3, infinity}, 0.5, 0, scratch, trial).status == Status::invalid_clock);
    assert(step(updated, {first.values, 3, DBL_MAX}, DBL_MAX, 0, scratch, trial).status == Status::invalid_clock);
    assert(step(updated, {first.values, 3, first.time, UINT64_MAX}, 0.5, 0, scratch, trial).status ==
           Status::generation_overflow);
    const auto no_values = step(updated, first, 0.5, 0, scratch, {trial_values, 2, trial_events, 2});
    assert(no_values.status == Status::value_capacity && no_values.required == 3);
    const auto no_signals = step(updated, first, 0.5, 0, {signals, 1}, trial);
    assert(no_signals.status == Status::signal_capacity && no_signals.required == 2);
    const auto no_events = step(updated, first, 0.5, 0, scratch, {trial_values, 3, trial_events, 1});
    assert(no_events.status == Status::event_capacity && no_events.required == 2);
    expect_values({trial_values, 3}, -1, -2, -3);
    assert(trial_events[0].edge_id == 99 && trial_events[0].time == -1 && trial_events[0].signal == -2);
    assert(trial_events[1].edge_id == 100 && trial_events[1].time == -2 && trial_events[1].signal == -3);
    assert(first.time == 1.0 && first.generation == 2);
    expect_values(first, 7.0f, 3.0f, 4.25f);

    const double endpoint = 2.0;
    const auto explicit_time = step(updated, first, 0.5, 0, scratch, trial, &endpoint);
    assert(explicit_time.status == Status::success && explicit_time.state.time == endpoint);
    assert(explicit_time.state.generation == 3 && explicit_time.event_count == 2);
    assert(explicit_time.events[0].time == endpoint && explicit_time.events[1].time == endpoint);
    expect_values(explicit_time.state, 10.625f, 3.0f, 7.75f);

    printf("cell-a time=%g observed=%g structure_epoch=%llu\n", first.time, observe(first),
           static_cast<unsigned long long>(changed->structure_epoch()));
    return 0;
}
