#include "state_view.hh"
#include <cassert>
#include <iostream>
#include <type_traits>
using namespace cellerator::experimental::moonshot::state;
template<class F> void rejected(F f) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument&) { caught = true; }
    assert(caught);
}
int main() {
    static_assert(std::is_trivially_copyable_v<binding>);
    const coordinate ids[] = {{11, 0, 7}, {12, 0, 3}, {11, 1, 1}};
    double values[] = {0, 2, 0};
    epoch generations{2, 3, 4, 5};
    owner_view<double> owner{values, ids, 3, &generations};
    binding slots[] = {{ids[1], 1, true, true, activity::inactive, false},
                       {ids[0], 0, true, true, activity::inactive, false},
                       {ids[0], 0, true, true, activity::inactive, true},
                       {ids[2], 2, false, true, activity::inactive, true}};
    state_view<double> view(owner, slots, 4);
    assert((view.gather() == std::vector<double>{2, 0, 0, 0}));
    const double g[] = {5, 2, 3, 0};
    assert((view.pullback(g, 4) == std::vector<double>{5, 5, 0}));
    rejected([&] { view.pullback(g, 3); });
    // Borrowed arrays: an owner value update is visible in a freshly bound view.
    values[0] = 9;
    ++generations.values;
    rejected([&] { view.gather(); });
    state_view<double> refreshed(owner, slots, 4);
    assert(refreshed.gather()[1] == 9);
    ++slots[1].logical.incarnation;
    rejected([&] { state_view<double> bad(owner, slots, 4); });
    slots[1].logical = ids[1]; // distinct actor cannot masquerade at offset 0
    rejected([&] { state_view<double> bad(owner, slots, 4); });
    slots[1].logical = ids[0];
    slots[1].capacity = false;
    rejected([&] { state_view<double> bad(owner, slots, 4); });
    slots[1].capacity = true;
    slots[2].runtime_activity = activity::active;
    rejected([&] { state_view<double> bad(owner, slots, 4); });
    rejected([&] { state_view<double> bad({nullptr, ids, 3, &generations}, slots, 4); });
    rejected([&] { state_view<double> bad({values, ids, 3, nullptr}, slots, 4); });
    std::cout << "PASS host state: identity/incarnation, borrowed storage, generations, aliases, capacity\n";
}
