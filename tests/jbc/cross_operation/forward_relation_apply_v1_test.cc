#include <Cellerator/compute/projection_family/forward_relation_apply_v1.hh>

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

family::support_family_descriptor_v1 descriptor(std::uint64_t edges) {
    family::support_family_descriptor_v1 value{};
    value.identity.family_identity = {1, 2};
    value.identity.exact_support_identity = {3, 4};
    value.identity.structure_identity = {5, 6};
    value.identity.structure_epoch = {7};
    value.identity.source_axis = axis(10);
    value.identity.destination_axis = axis(30);
    value.identity.logical_edge_order = {50, 51};
    value.identity.logical_edge_count = edges;
    value.supported_operations = family::support_relation_apply_v1;
    return value;
}

} // namespace

int main() {
    const std::array<family::logical_relation_edge_v1, 5> edges{{
        {1, 0, 3}, {4, 0, 0}, {0, 2, 4}, {3, 2, 1}, {2, 3, 2}}};
    std::array<std::uint64_t, 5> offsets{};
    std::array<std::uint64_t, 5> sources{};
    std::array<std::uint64_t, 5> logical_ids{};
    std::array<std::uint8_t, 5> marks{};
    const family::forward_relation_apply_storage_v1 storage{
        offsets.data(), offsets.size(), sources.data(), sources.size(),
        logical_ids.data(), logical_ids.size(), marks.data(), marks.size()};
    family::forward_relation_apply_view_v1 view{};
    const auto result = family::build_forward_relation_apply_view_v1(
        descriptor(edges.size()), {100, 101}, {102, 103},
        5, 4, edges.data(), edges.size(), storage, &view);
    assert(result.built());
    assert((offsets == std::array<std::uint64_t, 5>{0, 2, 2, 4, 5}));
    assert((sources == std::array<std::uint64_t, 5>{1, 4, 0, 3, 2}));
    assert((logical_ids == std::array<std::uint64_t, 5>{3, 0, 4, 1, 2}));
    assert(view.logical_edge_count == 5);

    auto malformed = edges;
    malformed[1].source_index = malformed[0].source_index;
    assert(family::build_forward_relation_apply_view_v1(
               descriptor(edges.size()), {100, 101}, {102, 103}, 5, 4,
               malformed.data(), malformed.size(), storage, &view)
               .code
           == family::forward_relation_apply_code_v1::unordered_edge);
    malformed = edges;
    malformed[1].logical_edge_id = malformed[0].logical_edge_id;
    assert(family::build_forward_relation_apply_view_v1(
               descriptor(edges.size()), {100, 101}, {102, 103}, 5, 4,
               malformed.data(), malformed.size(), storage, &view)
               .code
           == family::forward_relation_apply_code_v1::duplicate_logical_edge);

    auto small_storage = storage;
    small_storage.logical_edge_mark_capacity = 4;
    assert(family::build_forward_relation_apply_view_v1(
               descriptor(edges.size()), {100, 101}, {102, 103}, 5, 4,
               edges.data(), edges.size(), small_storage, &view)
               .code
           == family::forward_relation_apply_code_v1::
               insufficient_mark_capacity);

    // Global indices retain 64-bit width even when the local test has one row.
    const std::array<family::logical_relation_edge_v1, 1> wide_edge{{
        {UINT64_C(0x100000001), 0, 0}}};
    std::array<std::uint64_t, 2> wide_offsets{};
    std::array<std::uint64_t, 1> wide_sources{};
    std::array<std::uint64_t, 1> wide_ids{};
    std::array<std::uint8_t, 1> wide_marks{};
    assert(family::build_forward_relation_apply_view_v1(
               descriptor(1), {100, 101}, {102, 103},
               UINT64_C(0x100000002), 1, wide_edge.data(), 1,
               {wide_offsets.data(), 2, wide_sources.data(), 1,
                wide_ids.data(), 1, wide_marks.data(), 1},
               &view)
               .built());
    assert(wide_sources[0] == UINT64_C(0x100000001));
}
