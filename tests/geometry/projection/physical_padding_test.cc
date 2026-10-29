#include <Cellerator/compute/projection/physical_mma_hybrid.hh>
#include <Cellerator/geometry/work_layout.hh>

#include <cassert>
#include <cstdint>

namespace projection = cellerator::compute::projection;
namespace geometry = cellerator::geometry;

namespace cellerator::compute::projection {
bool build_mma_physical_work_layout_v1(
    const geometry::work_layout_view_v1 &, std::uint32_t, std::uint32_t *,
    std::uint32_t, std::uint32_t *, std::uint32_t, std::uint32_t *) noexcept;
}

int main() {
    const std::uint32_t execution_to_window[] = {2u, 0u, 1u};
    const std::uint32_t window_to_execution[] = {1u, 2u, 0u};
    geometry::work_layout_view_v1 semantic{};
    semantic.work_count = 3u;
    semantic.execution_to_window = execution_to_window;
    semantic.window_to_execution = window_to_execution;

    std::uint32_t physical_to_execution[16]{};
    std::uint32_t execution_to_physical[3]{};
    std::uint32_t physical_count = 0u;
    assert(projection::build_mma_physical_work_layout_v1(semantic, 16u,
        physical_to_execution, 16u, execution_to_physical, 3u,
        &physical_count));
    assert(physical_count == 16u);
    for (std::uint32_t i = 0u; i < semantic.work_count; ++i) {
        assert(physical_to_execution[i] == i);
        assert(execution_to_physical[i] == i);
        assert(semantic.execution_to_window[physical_to_execution[i]]
            == execution_to_window[i]);
    }
    for (std::uint32_t i = semantic.work_count; i < physical_count; ++i)
        assert(physical_to_execution[i] == geometry::invalid_work_item);

    // Provider padding cannot be represented as portable work identity.
    assert(semantic.work_count == 3u);
    assert(semantic.execution_to_window[0] == 2u);
    assert(semantic.window_to_execution[2] == 0u);

    std::uint32_t too_small[15]{};
    assert(!projection::build_mma_physical_work_layout_v1(semantic, 16u,
        too_small, 15u, execution_to_physical, 3u, &physical_count));
    assert(!projection::build_mma_physical_work_layout_v1(semantic, 17u,
        physical_to_execution, 16u, execution_to_physical, 3u,
        &physical_count));

    const std::uint32_t invalid_inverse[] = {1u, 0u, 2u};
    geometry::work_layout_view_v1 invalid = semantic;
    invalid.window_to_execution = invalid_inverse;
    assert(!projection::build_mma_physical_work_layout_v1(invalid, 16u,
        physical_to_execution, 16u, execution_to_physical, 3u,
        &physical_count));
    return 0;
}
