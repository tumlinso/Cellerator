#include <Cellerator/geometry/admissibility.hh>

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
    window.identity = {0x11u, 0x22u};
    window.axis = make_axis(10u);
    window.axis_extent = 12u;
    window.member_count = 4u;
    window.members = members;
    return window;
}

geo::admissibility_record_v1 make_record(
    geo::admissibility_constraint_kind kind,
    const axis_identity &axis) noexcept {
    geo::admissibility_record_v1 record{};
    record.kind = kind;
    record.axis = axis;
    return record;
}

void test_default_is_cheap_and_permissive() {
    const geo::work_window_view_v1 unvalidated_window{};
    const geo::admissibility_view_v1 admissibility{};
    assert(geo::admissibility_is_permissive(admissibility));
    assert(geo::validate_admissibility(unvalidated_window, admissibility));
}

void test_each_constraint_kind_is_axis_qualified() {
    const std::uint32_t members[] = {8u, 2u, 6u, 4u};
    const geo::work_window_view_v1 window = make_window(members);
    geo::admissibility_record_v1 records[7]{};

    records[0] = make_record(
        geo::admissibility_constraint_kind::fixed_position, window.axis);
    records[0].subject = 8u;
    records[0].related = 1u;

    records[1] = make_record(
        geo::admissibility_constraint_kind::fixed_original_group_membership,
        window.axis);
    records[1].subject = 2u;
    records[1].related = 2u;

    records[2] = make_record(
        geo::admissibility_constraint_kind::must_link, window.axis);
    records[2].subject = 8u;
    records[2].related = 6u;

    records[3] = make_record(
        geo::admissibility_constraint_kind::cannot_share_group, window.axis);
    records[3].subject = 8u;
    records[3].related = 4u;

    records[4] = make_record(
        geo::admissibility_constraint_kind::precedence, window.axis);
    records[4].subject = 2u;
    records[4].related = 4u;

    records[5] = make_record(
        geo::admissibility_constraint_kind::partition_barrier, window.axis);
    records[5].lower_bound = 1u;
    records[5].upper_bound = 2u;

    records[6] = make_record(
        geo::admissibility_constraint_kind::bounded_exchange_window,
        window.axis);
    records[6].subject = 6u;
    records[6].lower_bound = 1u;
    records[6].upper_bound = 3u;

    geo::admissibility_view_v1 admissibility{};
    admissibility.original_group_count = 4u;
    admissibility.record_count = 7u;
    admissibility.records = records;
    assert(geo::validate_admissibility(window, admissibility));
}

void test_axis_membership_and_bounds_are_rejected() {
    const std::uint32_t members[] = {8u, 2u, 6u, 4u};
    const geo::work_window_view_v1 window = make_window(members);
    geo::admissibility_record_v1 record = make_record(
        geo::admissibility_constraint_kind::must_link, window.axis);
    record.subject = 8u;
    record.related = 5u;
    geo::admissibility_view_v1 admissibility{};
    admissibility.record_count = 1u;
    admissibility.records = &record;
    assert(geo::validate_admissibility(window, admissibility).code
        == geo::admissibility_validation_code::related_not_in_window);

    record.related = 6u;
    record.axis.order.generation += 1u;
    assert(geo::validate_admissibility(window, admissibility).code
        == geo::admissibility_validation_code::axis_mismatch);

    record = make_record(
        geo::admissibility_constraint_kind::fixed_position, window.axis);
    record.subject = 8u;
    record.related = window.member_count;
    assert(geo::validate_admissibility(window, admissibility).code
        == geo::admissibility_validation_code::invalid_position);

    record = make_record(
        geo::admissibility_constraint_kind::bounded_exchange_window,
        window.axis);
    record.subject = 8u;
    record.lower_bound = 2u;
    record.upper_bound = 4u;
    admissibility.original_group_count = 4u;
    assert(geo::validate_admissibility(window, admissibility).code
        == geo::admissibility_validation_code::invalid_exchange_window);
}

void test_direct_conflicts_are_rejected() {
    const std::uint32_t members[] = {8u, 2u, 6u, 4u};
    const geo::work_window_view_v1 window = make_window(members);
    geo::admissibility_record_v1 records[2]{};
    records[0] = make_record(
        geo::admissibility_constraint_kind::must_link, window.axis);
    records[0].subject = 8u;
    records[0].related = 2u;
    records[1] = make_record(
        geo::admissibility_constraint_kind::cannot_share_group, window.axis);
    records[1].subject = 2u;
    records[1].related = 8u;

    geo::admissibility_view_v1 admissibility{};
    admissibility.record_count = 2u;
    admissibility.records = records;
    auto result = geo::validate_admissibility(window, admissibility);
    assert(result.code
        == geo::admissibility_validation_code::conflicting_constraint);
    assert(result.record_index == 1u);
    assert(result.conflicting_record_index == 0u);

    records[0] = make_record(
        geo::admissibility_constraint_kind::fixed_position, window.axis);
    records[0].subject = 8u;
    records[0].related = 0u;
    records[1] = make_record(
        geo::admissibility_constraint_kind::fixed_position, window.axis);
    records[1].subject = 2u;
    records[1].related = 0u;
    result = geo::validate_admissibility(window, admissibility);
    assert(result.code
        == geo::admissibility_validation_code::conflicting_constraint);
}

void test_view_and_record_headers_are_closed() {
    const std::uint32_t members[] = {8u, 2u, 6u, 4u};
    const geo::work_window_view_v1 window = make_window(members);
    geo::admissibility_view_v1 admissibility{};
    admissibility.schema_version += 1u;
    assert(geo::validate_admissibility(window, admissibility).code
        == geo::admissibility_validation_code::unsupported_version);

    geo::admissibility_record_v1 record = make_record(
        geo::admissibility_constraint_kind::fixed_position, window.axis);
    record.subject = 8u;
    record.reserved[1] = 1u;
    admissibility = {};
    admissibility.record_count = 1u;
    admissibility.records = &record;
    assert(geo::validate_admissibility(window, admissibility).code
        == geo::admissibility_validation_code::nonzero_reserved);
}

} // namespace

int main() {
    test_default_is_cheap_and_permissive();
    test_each_constraint_kind_is_axis_qualified();
    test_axis_membership_and_bounds_are_rejected();
    test_direct_conflicts_are_rejected();
    test_view_and_record_headers_are_closed();
    return 0;
}
