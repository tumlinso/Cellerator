#include <Cellerator/compute/projection_family/transpose_relation_apply_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace family = cellerator::compute::projection_family;
namespace execution = cellerator::execution;

namespace {

execution::persistent_axis_identity axis(std::uint64_t base) {
    return {{execution::biological_abi_version,
             execution::serialized_record_kind::persistent_axis_identity,
             sizeof(execution::persistent_axis_identity)},
            {base + 1, base + 2}, {base + 3, base + 4},
            {base + 5, base + 6}, {base + 7, base + 8}};
}

family::support_family_descriptor_v1 descriptor() {
    family::support_family_descriptor_v1 value{};
    value.identity.family_identity = {1, 2};
    value.identity.exact_support_identity = {3, 4};
    value.identity.structure_identity = {5, 6};
    value.identity.structure_epoch = {7};
    value.identity.source_axis = axis(10);
    value.identity.destination_axis = axis(30);
    value.identity.logical_edge_order = {50, 51};
    value.identity.logical_edge_count = 5;
    value.supported_operations = family::support_relation_apply_v1
        | family::support_relation_apply_transpose_v1;
    return value;
}

} // namespace

int main() {
    const auto support = descriptor();
    const std::array<family::logical_relation_edge_v1, 5> edges{{
        {1, 0, 3}, {4, 0, 0}, {0, 2, 4}, {3, 2, 1}, {2, 3, 2}}};
    std::array<std::uint64_t, 5> forward_offsets{};
    std::array<std::uint64_t, 5> forward_sources{};
    std::array<std::uint64_t, 5> forward_ids{};
    std::array<std::uint8_t, 5> forward_marks{};
    family::forward_relation_apply_view_v1 forward{};
    assert(family::build_forward_relation_apply_view_v1(
               support, {100, 101}, {102, 103}, 5, 4,
               edges.data(), edges.size(),
               {forward_offsets.data(), forward_offsets.size(),
                forward_sources.data(), forward_sources.size(),
                forward_ids.data(), forward_ids.size(),
                forward_marks.data(), forward_marks.size()},
               &forward)
               .built());

    std::array<std::uint64_t, 6> offsets{};
    std::array<std::uint64_t, 5> destinations{};
    std::array<std::uint64_t, 5> logical_ids{};
    std::array<std::uint64_t, 5> cursors{};
    std::array<std::uint8_t, 5> marks{};
    family::transpose_relation_apply_view_v1 transpose{};
    assert(family::build_transpose_relation_apply_view_v1(
               support, forward, {200, 201}, {202, 203},
               {offsets.data(), offsets.size(), destinations.data(),
                destinations.size(), logical_ids.data(), logical_ids.size(),
                cursors.data(), cursors.size(), marks.data(), marks.size()},
               &transpose)
               .built());
    assert((offsets == std::array<std::uint64_t, 6>{0, 1, 2, 3, 4, 5}));
    assert((destinations == std::array<std::uint64_t, 5>{2, 0, 3, 2, 0}));
    assert((logical_ids == std::array<std::uint64_t, 5>{4, 3, 2, 1, 0}));

    auto malformed = forward;
    std::array<std::uint64_t, 5> duplicate_ids = forward_ids;
    duplicate_ids[1] = duplicate_ids[0];
    malformed.logical_edge_ids = duplicate_ids.data();
    assert(family::build_transpose_relation_apply_view_v1(
               support, malformed, {200, 201}, {202, 203},
               {offsets.data(), offsets.size(), destinations.data(),
                destinations.size(), logical_ids.data(), logical_ids.size(),
                cursors.data(), cursors.size(), marks.data(), marks.size()},
               &transpose)
               .code
           == family::transpose_relation_apply_code_v1::duplicate_logical_edge);

    auto other_family = support;
    other_family.identity.structure_epoch.value = 8;
    assert(family::build_transpose_relation_apply_view_v1(
               other_family, forward, {200, 201}, {202, 203},
               {offsets.data(), offsets.size(), destinations.data(),
                destinations.size(), logical_ids.data(), logical_ids.size(),
                cursors.data(), cursors.size(), marks.data(), marks.size()},
               &transpose)
               .code
           == family::transpose_relation_apply_code_v1::family_mismatch);
}
