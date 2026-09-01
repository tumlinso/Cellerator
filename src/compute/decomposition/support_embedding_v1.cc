#include <Cellerator/compute/decomposition/support_embedding_v1.hh>

#include <limits>

namespace cellerator::compute::decomposition {
namespace {

support_embedding_validation_result_v1 failure(
    support_embedding_validation_code_v1 code,
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

support_embedding_validation_result_v1
validate_support_embedding_decomposition_v1(
    const support_embedding_decomposition_v1 &decomposition) noexcept {
    using code = support_embedding_validation_code_v1;

    if (decomposition.schema_version != support_embedding_schema_version_v1)
        return failure(code::unsupported_schema);
    if (decomposition.reserved != 0u
        || !all_zero(decomposition.reserved2, sizeof(decomposition.reserved2)))
        return failure(code::nonzero_reserved);
    if (!operation::v2::valid_stable_id(decomposition.decomposition_identity))
        return failure(code::invalid_identity);
    if (decomposition.problem == nullptr)
        return failure(code::missing_problem);
    if (!operation::v2::validate_operation_problem(*decomposition.problem))
        return failure(code::invalid_problem);
    if (decomposition.problem->kind
            != operation::v2::operation_kind::contract_on_support
        || decomposition.problem->orientation
            != operation::v2::relation_orientation::forward)
        return failure(code::unsupported_operation);
    if (decomposition.problem->dense_width == 0u)
        return failure(code::invalid_embedding_width);
    if (decomposition.embedding_interval_count == 0u)
        return failure(code::invalid_interval_count);
    if (decomposition.embedding_intervals == nullptr)
        return failure(code::missing_intervals);
    if (decomposition.split_axis != split_axis_kind_v1::dense_channel
        || decomposition.kind != decomposition_kind_v1::disjoint
        || decomposition.fragment_role != fragment_role_v1::owned)
        return failure(code::invalid_vocabulary);
    if (decomposition.produces_partial_results
        || decomposition.requires_partial_algebra)
        return failure(code::invalid_partial_result_contract);

    std::uint32_t expected_begin = 0u;
    for (std::uint64_t index = 0u;
         index < decomposition.embedding_interval_count; ++index) {
        const auto &interval = decomposition.embedding_intervals[index];
        if (interval.count == 0u)
            return failure(code::empty_interval, index);
        if (interval.begin != expected_begin)
            return failure(code::interval_offset_mismatch, index);
        if (interval.count
            > std::numeric_limits<std::uint32_t>::max() - interval.begin)
            return failure(code::interval_range_overflow, index);
        expected_begin = interval.begin + interval.count;
        if (expected_begin > decomposition.problem->dense_width)
            return failure(code::interval_range_overflow, index);
    }
    if (expected_begin != decomposition.problem->dense_width)
        return failure(code::incomplete_partition,
            decomposition.embedding_interval_count - 1u);
    return {};
}

}  // namespace cellerator::compute::decomposition
