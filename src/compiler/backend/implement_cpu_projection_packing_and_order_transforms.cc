#include <Cellerator/compiler/backend/implement_cpu_projection_packing_and_order_transforms_v1.hh>

#include <algorithm>
#include <limits>

namespace cellerator::compiler::backend::v1 {

cpu_order_transform_status_v1 run_cpu_order_transform_v1(
    const cpu_order_transform_request_v1& request) noexcept {
    if (request.input == nullptr || request.output == nullptr
        || request.item_count == 0 || request.width == 0)
        return cpu_order_transform_status_v1::invalid_argument;
    if (request.transform == cpu_order_transform_v1::preserve) {
        if (request.input != request.output)
            std::copy_n(request.input, request.item_count * request.width,
                request.output);
        return cpu_order_transform_status_v1::success;
    }
    if (request.input == request.output)
        return cpu_order_transform_status_v1::illegal_alias;
    if (request.physical_to_canonical == nullptr
        || request.permutation_marks == nullptr)
        return cpu_order_transform_status_v1::invalid_argument;
    std::fill_n(request.permutation_marks, request.item_count, std::uint8_t{0});
    for (std::uint64_t physical = 0; physical < request.item_count; ++physical) {
        const auto canonical = request.physical_to_canonical[physical];
        if (canonical >= request.item_count
            || request.permutation_marks[canonical] != 0)
            return cpu_order_transform_status_v1::invalid_permutation;
        request.permutation_marks[canonical] = 1;
        const auto source = request.transform == cpu_order_transform_v1::pack
            ? canonical : physical;
        const auto destination = request.transform == cpu_order_transform_v1::pack
            ? physical : canonical;
        std::copy_n(request.input + source * request.width, request.width,
            request.output + destination * request.width);
    }
    return cpu_order_transform_status_v1::success;
}

cpu_pack_break_even_v1 evaluate_cpu_pack_break_even_v1(
    std::uint64_t pack_nanoseconds,
    std::uint64_t unpacked_execution_nanoseconds,
    std::uint64_t packed_execution_nanoseconds) noexcept {
    cpu_pack_break_even_v1 result{pack_nanoseconds,
        unpacked_execution_nanoseconds, packed_execution_nanoseconds, 0, false};
    if (packed_execution_nanoseconds >= unpacked_execution_nanoseconds)
        return result;
    const auto saved = unpacked_execution_nanoseconds
        - packed_execution_nanoseconds;
    result.minimum_reuse = pack_nanoseconds / saved
        + (pack_nanoseconds % saved == 0 ? 0 : 1);
    result.minimum_reuse = std::max<std::uint64_t>(1, result.minimum_reuse);
    result.packing_profitable = true;
    return result;
}

}  // namespace cellerator::compiler::backend::v1
