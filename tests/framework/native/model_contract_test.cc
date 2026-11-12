#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/dynamical.hh>
#include <Cellerator/cell.hh>

#include <cassert>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

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

template<class T, class = void>
struct HasRvalueState : std::false_type {};
template<class T>
struct HasRvalueState<T, std::void_t<decltype(std::declval<T&&>().state())>>
    : std::true_type {};

template<class T, class = void>
struct HasRvalueModel : std::false_type {};
template<class T>
struct HasRvalueModel<T, std::void_t<decltype(std::declval<T&&>().model())>>
    : std::true_type {};

template<class T, class = void>
struct HasRvalueIdentity : std::false_type {};
template<class T>
struct HasRvalueIdentity<T, std::void_t<decltype(std::declval<T&&>().identity())>>
    : std::true_type {};

template<class T, class = void>
struct HasRvalueId : std::false_type {};
template<class T>
struct HasRvalueId<T, std::void_t<decltype(std::declval<T&&>().id())>>
    : std::true_type {};

template<class T, class = void>
struct HasRvalueObservation : std::false_type {};
template<class T>
struct HasRvalueObservation<T, std::void_t<decltype(std::declval<T&&>().observe())>>
    : std::true_type {};

template<class T, class = void>
struct HasRvalueStateHandle : std::false_type {};
template<class T>
struct HasRvalueStateHandle<T, std::void_t<decltype(std::declval<T&&>().state_handle())>>
    : std::true_type {};

template<class T, class = void>
struct HasRvalueSnapshotView : std::false_type {};
template<class T>
struct HasRvalueSnapshotView<T, std::void_t<decltype(
    std::declval<T&&>().template view<m::DynamicalStateSnapshotView<int>>())>>
    : std::true_type {};

template<class T, class = void>
struct HasForward : std::false_type {};
template<class T>
struct HasForward<T, std::void_t<decltype(std::declval<T&>().forward())>>
    : std::true_type {};

int main() {
    // Plain state-returning callbacks normalize to StepResult automatically.
    auto model = m::make_dynamical_model<int, Context>([](const int& state, const Context& context) {
        return state + (context.parameters ? context.parameters->scale : 1)
                     + (context.inputs ? context.inputs->offset : 0);
    }, std::make_shared<Parameters>(Parameters{2}));
    using ModelType = typename decltype(model)::element_type;
    static_assert(!HasForward<ModelType>::value,
                  "DynamicalModel remains an explicit transition model");
    static_assert(!std::is_invocable_v<ModelType&>);
    auto cell = m::make_dynamical_cell(model, 5, std::string("arbitrary-id"), 3.0);
    using CellType = decltype(cell);
    static_assert(!HasRvalueState<CellType>::value);
    static_assert(!HasRvalueModel<CellType>::value);
    static_assert(!HasRvalueIdentity<CellType>::value);
    static_assert(!HasRvalueId<CellType>::value);
    static_assert(!HasRvalueObservation<CellType>::value);
    static_assert(HasRvalueStateHandle<CellType>::value);
    static_assert(!HasRvalueSnapshotView<CellType>::value);
    static_assert(std::is_same_v<decltype(std::declval<CellType&&>().state_handle()),
                                 std::shared_ptr<const int>>);
    static_assert(!HasForward<CellType>::value,
                  "DynamicalCell does not turn forward into step or observe");
    static_assert(!std::is_polymorphic_v<CellType>,
                  "typed snapshots do not require runtime CellViews inheritance");
    static_assert(CellType::template supports<m::DynamicalStateSnapshotView<int>>());
    static_assert(!CellType::template supports_edit<m::DynamicalStateSnapshotView<int>>());
    const auto previous = cell.view<m::DynamicalStateSnapshotView<int>>();
    assert(cell.supports<m::DynamicalStateSnapshotView<int>>());
    assert(cell.supports<cellerator::CellIdentityView<std::string>>());
    assert(cell.model().get() == model.get());
    auto result = cell.advance(0.5);
    assert(result->state == 7 && *result->time == 3.5 && cell.time() == 3.5);
    assert(cell.view<m::DynamicalStateSnapshotView<int>>().state == cell.state_handle());
    assert(*previous.state == 5);
    assert(cell.view<cellerator::CellIdentityView<std::string>>().value == "arbitrary-id");
    assert(cell.id() == "arbitrary-id");
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
    assert(movable.view<cellerator::CellIdentityView<int>>().value == 42);
    auto move_result = movable.advance(5.0);
    assert(**movable.state_handle() == 9 && movable.time() == 0.125);
    assert(move_result->events[0] == "updated" && move_result->effects[0] == 7);
    assert(move_result->metadata == "caller metadata");
    auto move_snapshot = movable.view<m::DynamicalStateSnapshotView<State>>();
    movable.advance();
    assert(*move_result->state == 9 && *movable.state() == 10);
    assert(**move_snapshot.state == 9);

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
}
