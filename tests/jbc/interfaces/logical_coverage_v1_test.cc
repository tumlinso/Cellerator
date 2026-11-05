#include <Cellerator/execution/preparation/logical_coverage_v1.hh>

#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

namespace preparation = cellerator::execution::preparation;
namespace execution = cellerator::execution;

execution::persistent_axis_identity axis(std::uint64_t seed) {
    return {{execution::biological_abi_version,
                execution::serialized_record_kind::persistent_axis_identity,
                sizeof(execution::persistent_axis_identity)},
        {seed + 1u, 1u}, {seed + 2u, 1u}, {seed + 3u, 1u},
        {seed + 4u, 1u}};
}

preparation::logical_coverage_view_v1 base_coverage() {
    preparation::logical_coverage_view_v1 coverage{};
    coverage.coverage_identity = {11u, 12u};
    coverage.structure = {21u, 22u};
    coverage.epoch = {1u};
    coverage.source_axis = axis(30u);
    coverage.destination_axis = axis(40u);
    return coverage;
}

int main() {
    const preparation::canonical_interval_v1 intervals[] = {
        {2u, 3u}, {8u, 2u}};
    auto coverage = base_coverage();
    coverage.logical_count = 5u;
    coverage.members = intervals;
    coverage.member_count = 2u;
    coverage.member_bytes = sizeof(intervals[0]);
    assert(preparation::validate_logical_coverage_v1(coverage));

    auto malformed = coverage;
    malformed.schema_version += 1u;
    assert(preparation::validate_logical_coverage_v1(malformed).code
        == preparation::logical_coverage_validation_code_v1::
            unsupported_schema);
    malformed = coverage;
    malformed.role_flags = 0u;
    assert(preparation::validate_logical_coverage_v1(malformed).code
        == preparation::logical_coverage_validation_code_v1::
            missing_exact_role);

    const preparation::canonical_interval_v1 overlapping[] = {
        {2u, 3u}, {4u, 2u}};
    malformed = coverage;
    malformed.logical_count = 5u;
    malformed.members = overlapping;
    assert(preparation::validate_logical_coverage_v1(malformed).code
        == preparation::logical_coverage_validation_code_v1::
            unordered_or_overlapping_members);

    const std::uint64_t ids[] = {3u, 7u, 11u, 19u};
    coverage = base_coverage();
    coverage.kind = preparation::logical_coverage_kind_v1::explicit_ids;
    coverage.logical_count = 4u;
    coverage.members = ids;
    coverage.member_count = 4u;
    coverage.member_bytes = sizeof(ids[0]);
    assert(preparation::validate_logical_coverage_v1(coverage));

    const std::uint64_t duplicate_ids[] = {3u, 7u, 7u, 19u};
    malformed = coverage;
    malformed.members = duplicate_ids;
    assert(preparation::validate_logical_coverage_v1(malformed).code
        == preparation::logical_coverage_validation_code_v1::
            duplicate_member);

    const preparation::coverage_union_reference_v1 union_members[] = {
        {{71u, 1u}}, {{72u, 1u}}};
    coverage = base_coverage();
    coverage.kind = preparation::logical_coverage_kind_v1::coverage_union;
    coverage.logical_count = 9u;
    coverage.members = union_members;
    coverage.member_count = 2u;
    coverage.member_bytes = sizeof(union_members[0]);
    assert(preparation::validate_logical_coverage_v1(coverage));

    const preparation::coverage_union_reference_v1 duplicate_union[] = {
        {{71u, 1u}}, {{71u, 1u}}};
    malformed = coverage;
    malformed.members = duplicate_union;
    assert(preparation::validate_logical_coverage_v1(malformed).code
        == preparation::logical_coverage_validation_code_v1::
            duplicate_member);

    std::uint32_t provider_members[] = {1u, 2u, 3u};
    coverage = base_coverage();
    coverage.kind = preparation::logical_coverage_kind_v1::provider_defined;
    coverage.logical_count = 3u;
    coverage.payload_schema = {91u, 1u};
    coverage.members = provider_members;
    coverage.member_count = 3u;
    coverage.member_bytes = sizeof(provider_members[0]);
    assert(preparation::validate_logical_coverage_v1(coverage));

    malformed = coverage;
    malformed.payload_schema = {};
    assert(preparation::validate_logical_coverage_v1(malformed).code
        == preparation::logical_coverage_validation_code_v1::
            missing_payload_schema);
    malformed = coverage;
    malformed.member_count = std::numeric_limits<std::uint64_t>::max();
    malformed.member_bytes = 2u;
    assert(preparation::validate_logical_coverage_v1(malformed).code
        == preparation::logical_coverage_validation_code_v1::
            member_bytes_overflow);

    // Deterministic property sweep: every strictly increasing explicit set is
    // accepted, and replacing one element with its predecessor is rejected.
    std::uint64_t state = 0x9e3779b97f4a7c15u;
    for (std::uint64_t count = 1u; count <= 64u; ++count) {
        std::vector<std::uint64_t> generated;
        generated.reserve(static_cast<std::size_t>(count));
        std::uint64_t value = 0u;
        for (std::uint64_t index = 0u; index < count; ++index) {
            state = state * 6364136223846793005u + 1442695040888963407u;
            value += 1u + (state & 31u);
            generated.push_back(value);
        }
        auto property = base_coverage();
        property.kind =
            preparation::logical_coverage_kind_v1::relation_edge_ids;
        property.logical_count = count;
        property.members = generated.data();
        property.member_count = count;
        property.member_bytes = sizeof(generated[0]);
        assert(preparation::validate_logical_coverage_v1(property));
        if (count > 1u) {
            generated[count / 2u] = generated[count / 2u - 1u];
            assert(preparation::validate_logical_coverage_v1(property).code
                == preparation::logical_coverage_validation_code_v1::
                    duplicate_member);
        }
    }

    return 0;
}
