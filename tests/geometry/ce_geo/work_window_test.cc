#include <Cellerator/geometry/work_window.hh>

#include <cassert>

namespace {

using cellerator::execution::axis_identity;
using cellerator::geometry::work_window_kind;
using cellerator::geometry::work_window_validation_code;
using cellerator::geometry::work_window_view_v1;

constexpr axis_identity make_axis(std::uint32_t seed) noexcept {
    return {
        {seed + 1u, 1u},
        {seed + 2u, 1u},
        {seed + 3u, 1u},
        {seed + 4u, 1u}
    };
}

work_window_view_v1 make_window(
    work_window_kind kind,
    const std::uint32_t *members,
    std::uint32_t member_count) noexcept {
    work_window_view_v1 window{};
    window.kind = kind;
    window.identity = {0x1020304050607080ull, 0x90a0b0c0d0e0f001ull};
    window.axis = make_axis(10u);
    window.axis_extent = 8u;
    window.member_count = member_count;
    window.members = members;
    return window;
}

void test_each_work_kind_binds_explicit_membership() {
    const std::uint32_t members[] = {6u, 1u, 4u};
    const work_window_kind kinds[] = {
        work_window_kind::relation_rows,
        work_window_kind::dense_columns,
        work_window_kind::grouped_operation_instances
    };

    for (work_window_kind kind : kinds) {
        const work_window_view_v1 window = make_window(kind, members, 3u);
        assert(cellerator::geometry::validate_work_window(window));
        assert(cellerator::geometry::work_window_contains(window, 1u));
        assert(!cellerator::geometry::work_window_contains(window, 2u));
    }
}

void test_axis_identity_is_mandatory() {
    const std::uint32_t members[] = {0u};
    work_window_view_v1 window =
        make_window(work_window_kind::relation_rows, members, 1u);
    window.axis.order = {};

    const auto result = cellerator::geometry::validate_work_window(window);
    assert(result.code == work_window_validation_code::invalid_axis);
}

void test_identity_and_membership_are_mandatory() {
    const std::uint32_t members[] = {0u};
    work_window_view_v1 window =
        make_window(work_window_kind::dense_columns, members, 1u);
    window.identity = {};
    assert(cellerator::geometry::validate_work_window(window).code
        == work_window_validation_code::invalid_identity);

    window = make_window(work_window_kind::dense_columns, nullptr, 1u);
    assert(cellerator::geometry::validate_work_window(window).code
        == work_window_validation_code::missing_members);

    window = make_window(work_window_kind::dense_columns, members, 0u);
    assert(cellerator::geometry::validate_work_window(window).code
        == work_window_validation_code::invalid_member_count);
}

void test_members_must_be_unique_and_bounded() {
    const std::uint32_t duplicate_members[] = {2u, 5u, 2u};
    work_window_view_v1 window = make_window(
        work_window_kind::grouped_operation_instances,
        duplicate_members,
        3u);
    auto result = cellerator::geometry::validate_work_window(window);
    assert(result.code == work_window_validation_code::duplicate_member);
    assert(result.member_index == 2u);

    const std::uint32_t out_of_bounds_members[] = {2u, 8u};
    window = make_window(
        work_window_kind::grouped_operation_instances,
        out_of_bounds_members,
        2u);
    result = cellerator::geometry::validate_work_window(window);
    assert(result.code == work_window_validation_code::member_out_of_bounds);
    assert(result.member_index == 1u);
}

void test_kind_version_and_reserved_bytes_are_closed() {
    const std::uint32_t members[] = {3u};
    work_window_view_v1 window =
        make_window(work_window_kind::relation_rows, members, 1u);

    window.kind = static_cast<work_window_kind>(0u);
    assert(cellerator::geometry::validate_work_window(window).code
        == work_window_validation_code::invalid_kind);

    window = make_window(work_window_kind::relation_rows, members, 1u);
    window.schema_version += 1u;
    assert(cellerator::geometry::validate_work_window(window).code
        == work_window_validation_code::unsupported_version);

    window = make_window(work_window_kind::relation_rows, members, 1u);
    window.reserved[1] = 1u;
    assert(cellerator::geometry::validate_work_window(window).code
        == work_window_validation_code::nonzero_reserved);
}

} // namespace

int main() {
    test_each_work_kind_binds_explicit_membership();
    test_axis_identity_is_mandatory();
    test_identity_and_membership_are_mandatory();
    test_members_must_be_unique_and_bounded();
    test_kind_version_and_reserved_bytes_are_closed();
    return 0;
}
