#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/reactions.hh>
#include <cassert>
#include <limits>
#include <random>
#include <string>
using namespace cellerator::modeling;
int main() {
    ReactionNetwork<> network({"B", "A", "unrelated"}, {
        mass_action_reaction<>("forward", {{"A", 1}}, {{"B", 1}}, 2.0),
        mass_action_reaction<>("reverse", {{"B", 1}}, {{"A", 1}}, 1.0),
        mass_action_reaction<>("other", {}, {{"unrelated", 1}}, 0.25)
    });
    const auto initial = network.state({{"A", 5}});
    auto work = network.workspace(initial);
    assert(work.refresh().size() == 3);
    assert(work.rates()[0] == 10 && work.rates()[1] == 0);
    work.fire(0);
    assert((work.pending() == std::vector<std::size_t>{0, 1}));
    work.refresh();
    assert(work.rates()[0] == 8 && work.rates()[1] == 1);
    assert(work.snapshot().at("A") == 4 && initial.at("A") == 5);
    const auto conservation = network.conservation_residuals({{"A", 1}, {"B", 1}});
    for (auto residual : conservation) assert(residual == 0);
    auto empty = network.workspace(network.state({}));
    bool failed = false;
    try { empty.fire(0); } catch (const std::domain_error&) { failed = true; }
    assert(failed && empty.snapshot().at("B") == 0);
    std::mt19937_64 engine(42);
    std::uniform_real_distribution<double> uniform(0, 1);
    auto draw = [&] { return uniform(engine); };
    auto path = network.simulate(initial, 2.0, 3.0, draw);
    assert(path.time == 5.0 && path.state.at("A") + path.state.at("B") == 5);
    for (const auto& event : path.events) assert(event.time > 2 && event.time < 5);
    // Large waiting time must not fire past the horizon.
    auto late = network.simulate(initial, 0, 0.001, [] { return 0.99; });
    assert(late.events.empty() && late.state.at("A") == 5);
    ReactionNetwork<> overflow({"A", "B"}, {
        {"overflow", {{"A", -1}, {"B", 1}}, [](const auto&) { return 1.; }, std::vector<std::string>{}}
    });
    auto full = overflow.workspace(overflow.state({{"A", 1}, {"B", std::numeric_limits<std::int64_t>::max()}}));
    failed = false;
    try { full.fire(0); } catch (const std::overflow_error&) { failed = true; }
    assert(failed && full.snapshot().at("A") == 1);
}
