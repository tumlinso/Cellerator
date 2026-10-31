#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/model.hh>

#include <cassert>
#include <limits>
#include <string>
#include <type_traits>

namespace m = cellerator::modeling;

template<class Exception, class Callable>
void rejects(Callable&& callable) {
    bool caught = false;
    try { callable(); } catch (const Exception&) { caught = true; }
    assert(caught);
}

struct Parameters { int scale; };
struct Inputs { int offset; };
using Context = m::StepContext<Parameters, Inputs>;

int main() {
    // Plain state-returning callbacks normalize to StepResult automatically.
    auto model = m::make_dynamical_model<int, Context>([](const int& state, const Context& context) {
        return state + (context.parameters ? context.parameters->scale : 1)
                     + (context.inputs ? context.inputs->offset : 0);
    }, std::make_shared<Parameters>(Parameters{2}));
    auto cell = m::make_dynamical_cell(model, 5, std::string("arbitrary-id"), 3.0);
    const auto previous = cell.state_handle();
    auto result = cell.advance(0.5);
    assert(result->state == 7 && *result->time == 3.5 && cell.time() == 3.5);
    assert(cell.information() == cell.state() && cell.data() == cell.state());
    assert(*previous == 5 && cell.id() == "arbitrary-id");
    assert(&cell.observe() == &cell.state()); // Default observer returns a view.

    Parameters override{10};
    Inputs inputs{4};
    Context context;
    context.time = -99; // Cell owns its clock.
    context.dt = 0.25;
    context.parameters = &override;
    context.inputs = &inputs;
    cell.advance(context);
    assert(cell.state() == 21 && cell.time() == 3.75);
    model->set_parameters(std::make_shared<Parameters>(Parameters{3}));
    cell.advance();
    assert(cell.state() == 24 && cell.time() == 4.75);
    assert(result->state == 7); // Results remain valid after later transitions.

    // Move-only state is never copied by Model or Cell. Opaque caller events,
    // effects and metadata containers travel unchanged through the result.
    using State = std::unique_ptr<int>;
    using CustomResult = m::StepResult<State, std::vector<std::string>, std::vector<int>, std::string>;
    auto move_model = m::make_dynamical_model<State>([](const State& state, const m::StepContext<>& context) -> CustomResult {
        return {std::make_unique<int>(*state + 1), {"updated"}, {7}, context.time + 0.125, "caller metadata"};
    });
    auto movable = m::make_dynamical_cell(move_model, std::make_unique<int>(8), 42);
    auto move_result = movable.advance(5.0);
    assert(**movable.state_handle() == 9 && movable.time() == 0.125);
    assert(move_result->events[0] == "updated" && move_result->effects[0] == 7);
    assert(move_result->metadata == "caller metadata");
    movable.advance();
    assert(*move_result->state == 9 && *movable.state() == 10);
    assert(&movable.information() == &movable.state());

    // An exception or invalid result clock leaves the adopted state/time alone.
    auto bad_model = m::make_dynamical_model<int>([](const int& state, const m::StepContext<>& context) -> m::StepResult<int> {
        if (context.dt == 2) throw std::runtime_error("caller transition failed");
        return {state + 1, {}, {}, std::numeric_limits<double>::infinity()};
    });
    auto unchanged = m::make_dynamical_cell(bad_model, 10, 99);
    rejects<std::runtime_error>([&] { unchanged.advance(2.0); });
    rejects<std::invalid_argument>([&] { unchanged.advance(); });
    rejects<std::invalid_argument>([&] { unchanged.advance(std::numeric_limits<double>::quiet_NaN()); });
    assert(unchanged.state() == 10 && unchanged.time() == 0 && unchanged.value_generation() == 0);

    // Overflow in an implicit next clock also precedes state adoption.
    auto overflow = m::make_dynamical_cell(model, 1, 100, std::numeric_limits<double>::max());
    rejects<std::invalid_argument>([&] { overflow.advance(std::numeric_limits<double>::max()); });
    assert(overflow.state() == 1 && overflow.value_generation() == 0);

    // Old transition factory names remain a narrow compatibility route.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    auto legacy_transition = [](const int& state, const m::StepContext<>&) {
        return state + 1;
    };
    auto legacy_model = m::make_model<int>(legacy_transition);
    using LegacyModel = m::Model<int, m::StepContext<>, decltype(legacy_transition)>;
    static_assert(std::is_same_v<LegacyModel, typename decltype(legacy_model)::element_type>);
    auto legacy_cell = m::make_cell(legacy_model, 1, 101);
    legacy_cell.advance();
    assert(legacy_cell.state() == 2);
    static_assert(std::is_base_of_v<typename decltype(legacy_cell)::base_type,
                                    decltype(legacy_cell)>,
                  "compatibility factory still returns a DynamicalCell");
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
}
