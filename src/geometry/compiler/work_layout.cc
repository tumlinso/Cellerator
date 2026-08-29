#include <Cellerator/geometry/work_layout.hh>

namespace cellerator::geometry {
namespace {

constexpr work_layout_build_result build_failure(
    work_layout_build_code code,
    u32 index = invalid_work_item) noexcept {
    return {code, index};
}

constexpr work_layout_validation_result validation_failure(
    work_layout_validation_code code,
    u32 index = invalid_work_item) noexcept {
    return {code, index};
}

} // namespace

work_layout_build_result build_work_layout(
    const work_window_view_v1 &window,
    const u32 *execution_to_window,
    u32 work_count,
    u32 *window_to_execution,
    u32 inverse_capacity,
    work_layout_view_v1 *output) noexcept {
    if (execution_to_window == nullptr || window_to_execution == nullptr
        || execution_to_window == window_to_execution || output == nullptr
        || work_count == 0u)
        return build_failure(work_layout_build_code::invalid_argument);
    if (!validate_work_window(window))
        return build_failure(work_layout_build_code::invalid_work_window);
    if (work_count != window.member_count)
        return build_failure(work_layout_build_code::invalid_argument);
    if (inverse_capacity < work_count)
        return build_failure(
            work_layout_build_code::insufficient_inverse_capacity);

    for (u32 index = 0u; index < work_count; ++index)
        window_to_execution[index] = invalid_work_item;

    for (u32 execution_position = 0u; execution_position < work_count;
         ++execution_position) {
        const u32 window_index = execution_to_window[execution_position];
        if (window_index >= work_count)
            return build_failure(
                work_layout_build_code::work_item_out_of_bounds,
                execution_position);
        if (window_to_execution[window_index] != invalid_work_item)
            return build_failure(work_layout_build_code::duplicate_work_item,
                execution_position);
        window_to_execution[window_index] = execution_position;
    }

    work_layout_view_v1 result{};
    result.work_window = window.identity;
    result.axis = window.axis;
    result.work_count = work_count;
    result.execution_to_window = execution_to_window;
    result.window_to_execution = window_to_execution;
    *output = result;
    return {};
}

work_layout_validation_result validate_work_layout(
    const work_window_view_v1 &window,
    const work_layout_view_v1 &layout) noexcept {
    if (layout.schema_version != work_layout_schema_version)
        return validation_failure(
            work_layout_validation_code::unsupported_version);
    if (layout.reserved != 0u)
        return validation_failure(work_layout_validation_code::nonzero_reserved);
    if (!validate_work_window(window))
        return validation_failure(
            work_layout_validation_code::invalid_work_window);
    if (!execution::valid_identity(layout.work_window)
        || !execution::same_identity(layout.work_window, window.identity))
        return validation_failure(
            work_layout_validation_code::invalid_work_window_identity);
    if (!execution::same_axis_identity(layout.axis, window.axis))
        return validation_failure(work_layout_validation_code::axis_mismatch);
    if (layout.work_count != window.member_count)
        return validation_failure(
            work_layout_validation_code::work_count_mismatch);
    if (layout.execution_to_window == nullptr)
        return validation_failure(
            work_layout_validation_code::missing_permutation);
    if (layout.window_to_execution == nullptr)
        return validation_failure(work_layout_validation_code::missing_inverse);

    for (u32 execution_position = 0u;
         execution_position < layout.work_count; ++execution_position) {
        const u32 window_index =
            layout.execution_to_window[execution_position];
        if (window_index >= layout.work_count)
            return validation_failure(
                work_layout_validation_code::work_item_out_of_bounds,
                execution_position);
        for (u32 previous = 0u; previous < execution_position; ++previous)
            if (layout.execution_to_window[previous] == window_index)
                return validation_failure(
                    work_layout_validation_code::duplicate_work_item,
                    execution_position);
        if (layout.window_to_execution[window_index] != execution_position)
            return validation_failure(
                work_layout_validation_code::inverse_mismatch,
                execution_position);
    }

    for (u32 window_index = 0u; window_index < layout.work_count;
         ++window_index) {
        const u32 execution_position =
            layout.window_to_execution[window_index];
        if (execution_position >= layout.work_count)
            return validation_failure(
                work_layout_validation_code::inverse_out_of_bounds,
                window_index);
        if (layout.execution_to_window[execution_position] != window_index)
            return validation_failure(
                work_layout_validation_code::inverse_mismatch, window_index);
    }
    return {};
}

} // namespace cellerator::geometry
