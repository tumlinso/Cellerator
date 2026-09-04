#include <Cellerator/compiler/backend/implement_cpu_projection_packing_and_order_transforms_v1.hh>

#include <cassert>
#include <chrono>
#include <vector>

namespace cb = cellerator::compiler::backend::v1;

int main() {
    constexpr std::uint64_t count = 1024;
    constexpr std::uint32_t width = 4;
    std::vector<float> canonical(count * width);
    std::vector<float> packed(count * width);
    std::vector<float> recovered(count * width);
    std::vector<std::uint64_t> map(count);
    std::vector<std::uint8_t> marks(count);
    for (std::uint64_t i = 0; i < count; ++i) {
        map[i] = count - i - 1;
        for (std::uint32_t column = 0; column < width; ++column)
            canonical[i * width + column] = static_cast<float>(i * width + column);
    }
    const auto begin = std::chrono::steady_clock::now();
    assert(cb::run_cpu_order_transform_v1({cb::cpu_order_transform_v1::pack,
               canonical.data(), packed.data(), count, width, map.data(), marks.data()})
        == cb::cpu_order_transform_status_v1::success);
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin).count();
    assert(cb::run_cpu_order_transform_v1({cb::cpu_order_transform_v1::canonicalize,
               packed.data(), recovered.data(), count, width, map.data(), marks.data()})
        == cb::cpu_order_transform_status_v1::success);
    assert(recovered == canonical);

    const auto break_even = cb::evaluate_cpu_pack_break_even_v1(
        static_cast<std::uint64_t>(elapsed), 1000, 750);
    assert(break_even.packing_profitable && break_even.minimum_reuse >= 1);
    const auto fallback = cb::evaluate_cpu_pack_break_even_v1(100, 1000, 1000);
    assert(!fallback.packing_profitable && fallback.minimum_reuse == 0);

    map[1] = map[0];
    assert(cb::run_cpu_order_transform_v1({cb::cpu_order_transform_v1::pack,
               canonical.data(), packed.data(), count, width, map.data(), marks.data()})
        == cb::cpu_order_transform_status_v1::invalid_permutation);
}
