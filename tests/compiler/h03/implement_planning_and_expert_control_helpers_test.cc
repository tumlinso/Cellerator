#include <cassert>
struct cost_record { double preparation, execution, conversion, synchronization; };
constexpr double total(cost_record c) { return c.preparation+c.execution+c.conversion+c.synchronization; }
struct forced_plan { const char* candidate; bool explicit_unsafe; };
constexpr forced_plan force_unsafe(const char* name) { return {name,true}; }
int main() {
    static_assert(total({1,2,3,4}) == 10);
    constexpr auto plan = force_unsafe("direct-ceir");
    static_assert(plan.explicit_unsafe);
    assert(plan.candidate != nullptr);
}
