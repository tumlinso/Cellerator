#include <Cellerator/compute/decomposition/segment_disjoint_v1.hh>

#include <limits>

namespace cellerator::compute::decomposition {
namespace {

segment_disjoint_validation_result_v1 failure(
    segment_disjoint_validation_code_v1 code,
    std::uint64_t interval_index = 0u) noexcept {
    return {code, interval_index};
}

bool all_zero(const std::uint8_t *values, std::uint64_t count) noexcept {
    for (std::uint64_t index = 0u; index < count; ++index) {
        if (values[index] != 0u)
            return false;
    }
    return true;
}

}  // namespace

segment_disjoint_validation_result_v1 validate_segment_disjoint_v1(
    const segment_disjoint_decomposition_v1 &decomposition) noexcept {
    using code = segment_disjoint_validation_code_v1;

    if (decomposition.schema_version != segment_disjoint_schema_version_v1)
        return failure(code::unsupported_schema);
    if (decomposition.reserved != 0u
        || !all_zero(decomposition.reserved2, sizeof(decomposition.reserved2)))
        return failure(code::nonzero_reserved);
    if (!operation::v2::valid_stable_id(decomposition.decomposition_identity))
        return failure(code::invalid_identity);
    if (decomposition.problem == nullptr)
        return failure(code::missing_problem);
    if (!operation::v2::validate_relation_algebra_problem(
            *decomposition.problem))
        return failure(code::invalid_problem);
    if (decomposition.problem->core.kind
            != operation::v2::operation_kind::segment_reduce
        && decomposition.problem->core.kind
            != operation::v2::operation_kind::segment_normalize)
        return failure(code::unsupported_operation);
    if (decomposition.segment_count == 0u)
        return failure(code::invalid_segment_count);
    if (decomposition.segment_interval_count == 0u)
        return failure(code::invalid_interval_count);
    if (decomposition.segment_intervals == nullptr)
        return failure(code::missing_intervals);
    if (decomposition.kind != decomposition_kind_v1::disjoint
        || decomposition.fragment_role != fragment_role_v1::owned)
        return failure(code::invalid_vocabulary);
    if (decomposition.produces_partial_results
        || decomposition.requires_partial_algebra)
        return failure(code::invalid_partial_result_contract);

    std::uint64_t expected_begin = 0u;
    for (std::uint64_t index = 0u;
         index < decomposition.segment_interval_count; ++index) {
        const auto &interval = decomposition.segment_intervals[index];
        if (interval.count == 0u)
            return failure(code::empty_interval, index);
        if (interval.begin != expected_begin)
            return failure(code::interval_offset_mismatch, index);
        if (interval.count
            > std::numeric_limits<std::uint64_t>::max() - interval.begin)
            return failure(code::interval_range_overflow, index);
        expected_begin = interval.begin + interval.count;
        if (expected_begin > decomposition.segment_count)
            return failure(code::interval_range_overflow, index);
    }
    if (expected_begin != decomposition.segment_count)
        return failure(code::incomplete_partition,
            decomposition.segment_interval_count - 1u);
    return {};
}

}  // namespace cellerator::compute::decomposition
