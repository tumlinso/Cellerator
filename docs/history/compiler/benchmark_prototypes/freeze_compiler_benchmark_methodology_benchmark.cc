#include <array>
#include <cstddef>
#include <string_view>

namespace cellerator::compiler::benchmark {
struct methodology_v1 {
    unsigned warmups;
    unsigned repeats;
    bool benchmark_mutex;
    bool preserve_raw_samples;
    std::string_view cold_state;
    std::string_view warm_state;
};
constexpr methodology_v1 methodology{2, 11, true, true,
    "no process, parsed source, profile, cache, or artifact",
    "explicitly named retained state"};
static_assert(methodology.repeats >= 11 && methodology.repeats % 2 == 1);
}

int main() { return cellerator::compiler::benchmark::methodology.benchmark_mutex ? 0 : 1; }
