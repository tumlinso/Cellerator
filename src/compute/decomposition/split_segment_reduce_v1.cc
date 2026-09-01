#include <Cellerator/compute/decomposition/split_segment_reduce_v1.hh>

#include <limits>

namespace cellerator::compute::decomposition {
namespace {

split_segment_reduce_validation_result_v1 failure(
    split_segment_reduce_validation_code_v1 code,
    std::uint64_t segment_index = 0u,
    std::uint64_t fragment_index = 0u) noexcept {
    return {code, segment_index, fragment_index};
}

bool all_zero(const std::uint8_t *values, std::uint64_t count) noexcept {
    for (std::uint64_t index = 0u; index < count; ++index) {
        if (values[index] != 0u)
            return false;
    }
    return true;
}

}  // namespace

split_segment_reduce_validation_result_v1 validate_split_segment_reduce_v1(
    const split_segment_reduce_decomposition_v1 &decomposition) noexcept {
    using code = split_segment_reduce_validation_code_v1;

    if (decomposition.schema_version != split_segment_reduce_schema_version_v1)
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
        != operation::v2::operation_kind::segment_reduce)
        return failure(code::unsupported_operation);
    if (decomposition.segment_count == 0u)
        return failure(code::invalid_segment_count);
    if (decomposition.segment_member_counts == nullptr)
        return failure(code::missing_segment_counts);
    if (decomposition.fragment_count != 0u && decomposition.fragments == nullptr)
        return failure(code::missing_fragments);
    if (decomposition.split_axis != split_axis_kind_v1::logical_edge
        || decomposition.kind != decomposition_kind_v1::disjoint
        || decomposition.fragment_role != fragment_role_v1::owned)
        return failure(code::invalid_vocabulary);
    if (!decomposition.produces_partial_results
        || !decomposition.requires_partial_algebra)
        return failure(code::invalid_partial_result_contract);

    std::uint64_t fragment_index = 0u;
    for (std::uint64_t segment_index = 0u;
         segment_index < decomposition.segment_count; ++segment_index) {
        const auto extent = decomposition.segment_member_counts[segment_index];
        std::uint64_t expected_member = 0u;
        while (fragment_index < decomposition.fragment_count
            && decomposition.fragments[fragment_index].segment_index
                == segment_index) {
            const auto &fragment = decomposition.fragments[fragment_index];
            if (fragment.member_count == 0u)
                return failure(code::empty_fragment, segment_index,
                    fragment_index);
            if (fragment.member_begin != expected_member)
                return failure(code::member_offset_mismatch, segment_index,
                    fragment_index);
            if (fragment.member_count
                > std::numeric_limits<std::uint64_t>::max()
                    - fragment.member_begin)
                return failure(code::member_range_overflow, segment_index,
                    fragment_index);
            expected_member = fragment.member_begin + fragment.member_count;
            if (expected_member > extent)
                return failure(code::member_range_overflow, segment_index,
                    fragment_index);
            ++fragment_index;
        }
        if (expected_member != extent)
            return failure(code::incomplete_segment, segment_index,
                fragment_index);
        if (fragment_index < decomposition.fragment_count
            && decomposition.fragments[fragment_index].segment_index
                < segment_index + 1u)
            return failure(code::segment_index_mismatch, segment_index,
                fragment_index);
    }
    if (fragment_index != decomposition.fragment_count)
        return failure(code::extra_fragment, decomposition.segment_count,
            fragment_index);
    return {};
}

}  // namespace cellerator::compute::decomposition
