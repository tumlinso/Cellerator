#include <Cellerator/geometry/admissibility.hh>

namespace cellerator::geometry {
namespace {

constexpr bool is_pair_constraint(admissibility_constraint_kind kind) noexcept {
    return kind == admissibility_constraint_kind::must_link
        || kind == admissibility_constraint_kind::cannot_share_group
        || kind == admissibility_constraint_kind::precedence;
}

constexpr bool is_group_constraint(admissibility_constraint_kind kind) noexcept {
    return kind
            == admissibility_constraint_kind::fixed_original_group_membership
        || kind == admissibility_constraint_kind::partition_barrier
        || kind == admissibility_constraint_kind::bounded_exchange_window;
}

constexpr bool same_unordered_pair(
    const admissibility_record_v1 &lhs,
    const admissibility_record_v1 &rhs) noexcept {
    return (lhs.subject == rhs.subject && lhs.related == rhs.related)
        || (lhs.subject == rhs.related && lhs.related == rhs.subject);
}

constexpr admissibility_validation_result failure(
    admissibility_validation_code code,
    u32 record_index,
    u32 conflicting_record_index = invalid_admissibility_record_index) noexcept {
    return {code, record_index, conflicting_record_index};
}

admissibility_validation_result validate_record(
    const work_window_view_v1 &window,
    const admissibility_view_v1 &admissibility,
    u32 record_index) noexcept {
    const admissibility_record_v1 &record = admissibility.records[record_index];
    if (!valid_admissibility_constraint_kind(record.kind))
        return failure(admissibility_validation_code::invalid_kind, record_index);
    if (record.reserved[0] != 0u || record.reserved[1] != 0u
        || record.reserved[2] != 0u)
        return failure(
            admissibility_validation_code::nonzero_reserved, record_index);
    if (!execution::valid_axis_identity(record.axis))
        return failure(admissibility_validation_code::invalid_axis, record_index);
    if (!execution::same_axis_identity(record.axis, window.axis))
        return failure(admissibility_validation_code::axis_mismatch, record_index);
    if (is_group_constraint(record.kind)
        && admissibility.original_group_count == 0u)
        return failure(
            admissibility_validation_code::missing_original_groups,
            record_index);

    switch (record.kind) {
    case admissibility_constraint_kind::fixed_position:
        if (!work_window_contains(window, record.subject))
            return failure(
                admissibility_validation_code::subject_not_in_window,
                record_index);
        if (record.related >= window.member_count)
            return failure(
                admissibility_validation_code::invalid_position, record_index);
        if (record.lower_bound != 0u || record.upper_bound != 0u)
            return failure(
                admissibility_validation_code::nonzero_unused_field,
                record_index);
        break;
    case admissibility_constraint_kind::fixed_original_group_membership:
        if (!work_window_contains(window, record.subject))
            return failure(
                admissibility_validation_code::subject_not_in_window,
                record_index);
        if (record.related >= admissibility.original_group_count)
            return failure(
                admissibility_validation_code::group_out_of_bounds,
                record_index);
        if (record.lower_bound != 0u || record.upper_bound != 0u)
            return failure(
                admissibility_validation_code::nonzero_unused_field,
                record_index);
        break;
    case admissibility_constraint_kind::must_link:
    case admissibility_constraint_kind::cannot_share_group:
    case admissibility_constraint_kind::precedence:
        if (!work_window_contains(window, record.subject))
            return failure(
                admissibility_validation_code::subject_not_in_window,
                record_index);
        if (!work_window_contains(window, record.related))
            return failure(
                admissibility_validation_code::related_not_in_window,
                record_index);
        if (record.subject == record.related)
            return failure(
                admissibility_validation_code::self_relation, record_index);
        if (record.lower_bound != 0u || record.upper_bound != 0u)
            return failure(
                admissibility_validation_code::nonzero_unused_field,
                record_index);
        break;
    case admissibility_constraint_kind::partition_barrier:
        if (record.subject != 0u || record.related != 0u)
            return failure(
                admissibility_validation_code::nonzero_unused_field,
                record_index);
        if (record.lower_bound >= admissibility.original_group_count
            || record.upper_bound >= admissibility.original_group_count
            || record.lower_bound + 1u != record.upper_bound)
            return failure(
                admissibility_validation_code::invalid_partition_barrier,
                record_index);
        break;
    case admissibility_constraint_kind::bounded_exchange_window:
        if (!work_window_contains(window, record.subject))
            return failure(
                admissibility_validation_code::subject_not_in_window,
                record_index);
        if (record.related != 0u)
            return failure(
                admissibility_validation_code::nonzero_unused_field,
                record_index);
        if (record.lower_bound > record.upper_bound
            || record.upper_bound >= admissibility.original_group_count)
            return failure(
                admissibility_validation_code::invalid_exchange_window,
                record_index);
        break;
    }
    return {};
}

admissibility_validation_result validate_pair(
    const admissibility_record_v1 &lhs,
    u32 lhs_index,
    const admissibility_record_v1 &rhs,
    u32 rhs_index) noexcept {
    if (lhs.kind == admissibility_constraint_kind::fixed_position
        && rhs.kind == admissibility_constraint_kind::fixed_position) {
        if ((lhs.subject == rhs.subject && lhs.related != rhs.related)
            || (lhs.subject != rhs.subject && lhs.related == rhs.related))
            return failure(admissibility_validation_code::conflicting_constraint,
                rhs_index, lhs_index);
    }

    if (lhs.kind
            == admissibility_constraint_kind::fixed_original_group_membership
        && rhs.kind
            == admissibility_constraint_kind::fixed_original_group_membership
        && lhs.subject == rhs.subject && lhs.related != rhs.related)
        return failure(admissibility_validation_code::conflicting_constraint,
            rhs_index, lhs_index);

    if (is_pair_constraint(lhs.kind) && is_pair_constraint(rhs.kind)
        && same_unordered_pair(lhs, rhs)) {
        const bool link_exclusion =
            (lhs.kind == admissibility_constraint_kind::must_link
                && rhs.kind
                    == admissibility_constraint_kind::cannot_share_group)
            || (rhs.kind == admissibility_constraint_kind::must_link
                && lhs.kind
                    == admissibility_constraint_kind::cannot_share_group);
        const bool reverse_precedence =
            lhs.kind == admissibility_constraint_kind::precedence
            && rhs.kind == admissibility_constraint_kind::precedence
            && lhs.subject == rhs.related && lhs.related == rhs.subject;
        if (link_exclusion || reverse_precedence)
            return failure(admissibility_validation_code::conflicting_constraint,
                rhs_index, lhs_index);
    }

    if (lhs.kind == admissibility_constraint_kind::bounded_exchange_window
        && rhs.kind == admissibility_constraint_kind::bounded_exchange_window
        && lhs.subject == rhs.subject
        && (lhs.upper_bound < rhs.lower_bound
            || rhs.upper_bound < lhs.lower_bound))
        return failure(admissibility_validation_code::conflicting_constraint,
            rhs_index, lhs_index);

    const admissibility_record_v1 *membership = nullptr;
    const admissibility_record_v1 *exchange = nullptr;
    if (lhs.kind
            == admissibility_constraint_kind::fixed_original_group_membership
        && rhs.kind == admissibility_constraint_kind::bounded_exchange_window) {
        membership = &lhs;
        exchange = &rhs;
    } else if (
        rhs.kind
            == admissibility_constraint_kind::fixed_original_group_membership
        && lhs.kind == admissibility_constraint_kind::bounded_exchange_window) {
        membership = &rhs;
        exchange = &lhs;
    }
    if (membership != nullptr && membership->subject == exchange->subject
        && (membership->related < exchange->lower_bound
            || membership->related > exchange->upper_bound))
        return failure(admissibility_validation_code::conflicting_constraint,
            rhs_index, lhs_index);

    return {};
}

} // namespace

admissibility_validation_result validate_admissibility(
    const work_window_view_v1 &window,
    const admissibility_view_v1 &admissibility) noexcept {
    if (admissibility.schema_version != admissibility_schema_version)
        return failure(
            admissibility_validation_code::unsupported_version,
            invalid_admissibility_record_index);
    if (admissibility.reserved != 0u)
        return failure(admissibility_validation_code::nonzero_reserved,
            invalid_admissibility_record_index);
    if (admissibility.record_count == 0u)
        return {};
    if (admissibility.records == nullptr)
        return failure(admissibility_validation_code::missing_records,
            invalid_admissibility_record_index);

    for (u32 index = 0u; index < admissibility.record_count; ++index) {
        const admissibility_validation_result result =
            validate_record(window, admissibility, index);
        if (!result)
            return result;
        for (u32 previous = 0u; previous < index; ++previous) {
            const admissibility_validation_result pair_result = validate_pair(
                admissibility.records[previous], previous,
                admissibility.records[index], index);
            if (!pair_result)
                return pair_result;
        }
    }
    return {};
}

} // namespace cellerator::geometry
