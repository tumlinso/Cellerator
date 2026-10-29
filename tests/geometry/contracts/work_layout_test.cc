#include <Cellerator/geometry/work_layout.hh>

#include <cassert>

namespace {

namespace geo = cellerator::geometry;
using cellerator::execution::axis_identity;

constexpr axis_identity make_axis(std::uint32_t seed) noexcept {
    return {
        {seed + 1u, 1u},
        {seed + 2u, 1u},
        {seed + 3u, 1u},
        {seed + 4u, 1u}
    };
}

geo::work_window_view_v1 make_window(const std::uint32_t *members) noexcept {
    geo::work_window_view_v1 window{};
    window.identity = {0x100u, 0x200u};
    window.axis = make_axis(20u);
    window.axis_extent = 16u;
    window.member_count = 4u;
    window.members = members;
    return window;
}

void test_builds_exact_inverse_over_real_items() {
    const std::uint32_t members[] = {11u, 3u, 8u, 1u};
    const geo::work_window_view_v1 window = make_window(members);
    const std::uint32_t execution_to_window[] = {2u, 0u, 3u, 1u};
    std::uint32_t window_to_execution[4]{};
    geo::work_layout_view_v1 layout{};

    assert(geo::build_work_layout(window, execution_to_window, 4u,
        window_to_execution, 4u, &layout));
    assert(geo::validate_work_layout(window, layout));
    assert(window_to_execution[0] == 1u);
    assert(window_to_execution[1] == 3u);
    assert(window_to_execution[2] == 0u);
    assert(window_to_execution[3] == 2u);
    assert(geo::work_layout_axis_position(window, layout, 0u) == 8u);
    assert(geo::work_layout_axis_position(window, layout, 3u) == 3u);
}

void test_builder_rejects_padding_duplicates_and_short_storage() {
    const std::uint32_t members[] = {11u, 3u, 8u, 1u};
    const geo::work_window_view_v1 window = make_window(members);
    std::uint32_t inverse[4]{};
    geo::work_layout_view_v1 layout{};

    const std::uint32_t padding[] = {0u, 1u, geo::invalid_work_item, 3u};
    auto result = geo::build_work_layout(
        window, padding, 4u, inverse, 4u, &layout);
    assert(result.code == geo::work_layout_build_code::work_item_out_of_bounds);
    assert(result.index == 2u);

    const std::uint32_t duplicate[] = {0u, 2u, 2u, 3u};
    result = geo::build_work_layout(
        window, duplicate, 4u, inverse, 4u, &layout);
    assert(result.code == geo::work_layout_build_code::duplicate_work_item);
    assert(result.index == 2u);

    const std::uint32_t valid[] = {3u, 2u, 1u, 0u};
    result = geo::build_work_layout(
        window, valid, 4u, inverse, 3u, &layout);
    assert(result.code
        == geo::work_layout_build_code::insufficient_inverse_capacity);

    std::uint32_t aliased[] = {3u, 2u, 1u, 0u};
    result = geo::build_work_layout(
        window, aliased, 4u, aliased, 4u, &layout);
    assert(result.code == geo::work_layout_build_code::invalid_argument);
}

void test_independent_validator_rejects_broken_inverse_and_identity() {
    const std::uint32_t members[] = {11u, 3u, 8u, 1u};
    const geo::work_window_view_v1 window = make_window(members);
    const std::uint32_t permutation[] = {1u, 3u, 0u, 2u};
    std::uint32_t inverse[4]{};
    geo::work_layout_view_v1 layout{};
    assert(geo::build_work_layout(
        window, permutation, 4u, inverse, 4u, &layout));

    inverse[0] = 0u;
    assert(geo::validate_work_layout(window, layout).code
        == geo::work_layout_validation_code::inverse_mismatch);

    assert(geo::build_work_layout(
        window, permutation, 4u, inverse, 4u, &layout));
    layout.work_window.high += 1u;
    assert(geo::validate_work_layout(window, layout).code
        == geo::work_layout_validation_code::invalid_work_window_identity);

    assert(geo::build_work_layout(
        window, permutation, 4u, inverse, 4u, &layout));
    layout.axis.partition.generation += 1u;
    assert(geo::validate_work_layout(window, layout).code
        == geo::work_layout_validation_code::axis_mismatch);
}

void test_validator_does_not_trust_the_builder() {
    const std::uint32_t members[] = {11u, 3u, 8u, 1u};
    const geo::work_window_view_v1 window = make_window(members);
    const std::uint32_t duplicate[] = {0u, 1u, 1u, 3u};
    const std::uint32_t inverse[] = {0u, 1u, 2u, 3u};
    geo::work_layout_view_v1 layout{};
    layout.work_window = window.identity;
    layout.axis = window.axis;
    layout.work_count = 4u;
    layout.execution_to_window = duplicate;
    layout.window_to_execution = inverse;

    const auto result = geo::validate_work_layout(window, layout);
    assert(result.code == geo::work_layout_validation_code::duplicate_work_item);
    assert(result.index == 2u);

    layout.execution_to_window = inverse;
    layout.window_to_execution = inverse;
    layout.schema_version += 1u;
    assert(geo::validate_work_layout(window, layout).code
        == geo::work_layout_validation_code::unsupported_version);
}

} // namespace

int main() {
    test_builds_exact_inverse_over_real_items();
    test_builder_rejects_padding_duplicates_and_short_storage();
    test_independent_validator_rejects_broken_inverse_and_identity();
    test_validator_does_not_trust_the_builder();
    return 0;
}
