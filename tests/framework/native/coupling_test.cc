#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/coupling.hh>
#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace cellerator::modeling;

int main() {
    ExchangePlan plan(EntityLayout({42, 7}), EntityLayout({90}), EntityLayout({5, 12}),
                      {{100, 42, 90, 1.0}, {200, 7, 90, 1.0}});
    const ExchangeState<double> state{{4., 0., 0., 2.}, {0., 4.}};
    const auto result = plan.step(state, {2., 1.}, {4.}, {1., 0.5}, 0.25L);
    assert(std::abs(result.passive_dt_limit - 1.L) < 1e-15);
    assert(std::abs(result.state.cells[0] - 3.5) < 1e-15);
    assert(result.state.cells[2] == 0.); // incoming to environment is not visible to cell 7
    for (std::size_t s = 0; s < 2; ++s) {
        const auto before = state.cells[s] + state.cells[2+s] + state.compartments[s];
        const auto after = result.state.cells[s] + result.state.cells[2+s] + result.state.compartments[s];
        assert(std::abs(before - after) < 1e-14);
    }
    const ExchangeState<double> reservoir{{0., 0., 0., 0.}, {3., 0.}};
    const std::vector<double> rates{-4., 0., -8., 0.};
    const auto limited = plan.step(reservoir, {1., 1.}, {1.}, {0., 0.}, 1.L,
                                   rates, DonorPolicy::limit);
    assert(limited.limited);
    assert(std::abs(limited.state.cells[0] - 1.) < 1e-14);
    assert(std::abs(limited.state.cells[2] - 2.) < 1e-14);
    assert(limited.state.compartments[0] >= 0);
    assert(reservoir.compartments[0] == 3.);
    bool rejected = false;
    try { (void)plan.step(reservoir, {1., 1.}, {1.}, {0., 0.}, 1.L, rates); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { (void)plan.step(state, {2., 1.}, {4.}, {1., 0.5}, 2.L); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    ExchangePlan reversed(EntityLayout({7, 42}), EntityLayout({90}), EntityLayout({12, 5}),
                          {{200, 7, 90, 1.0}, {100, 42, 90, 1.0}});
    const ExchangeState<float> reordered{{2.f, 0.f, 0.f, 4.f}, {4.f, 0.f}};
    const auto float_result = reversed.step(reordered, {1.f, 2.f}, {4.f}, {0.5f, 1.f}, 0.25L);
    assert(std::abs(float_result.state.cells[3] - result.state.cells[0]) < 1e-6);
    assert(std::abs(float_result.state.compartments[0] - result.state.compartments[1]) < 1e-6);

    // One donor must share its budget across a large sparse fanout.
    std::vector<std::uint64_t> environment_ids;
    std::vector<ExchangeLink> fanout_links;
    for (std::uint64_t i = 0; i < 1000; ++i) {
        environment_ids.push_back(i);
        fanout_links.push_back({i, 1, i, 1.});
    }
    ExchangePlan fanout(EntityLayout({1}), EntityLayout(environment_ids), EntityLayout({1}), fanout_links);
    const auto spread = fanout.step(ExchangeState<double>{{1.}, std::vector<double>(1000)},
                                    std::vector<double>{1.}, std::vector<double>(1000, 1.),
                                    std::vector<double>{1.}, 10.L, {}, DonorPolicy::limit, false);
    assert(spread.state.cells[0] >= 0);
    long double total = spread.state.cells[0];
    for (const auto amount : spread.state.compartments) total += amount;
    assert(std::abs(total - 1.L) < 1e-13);
}
