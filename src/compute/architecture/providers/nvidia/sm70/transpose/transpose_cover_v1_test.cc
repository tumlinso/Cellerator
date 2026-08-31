#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_cover_v1.hh>

#include <cstdlib>

using namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose;

namespace {
void require(bool value) { if (!value) std::abort(); }
}

int main() {
    constexpr std::uint64_t high = (std::uint64_t{1u} << 32u) + 17u;
    global_relation_edge_v1 edges[]{{9u, high + 2u, high + 11u},
        {4u, high, high + 9u}, {7u, high, high + 10u}};
    std::uint64_t source_order[]{1u, 2u, 0u};
    std::uint64_t identity_order[]{1u, 2u, 0u};
    transpose_cover_input_v1 input{edges, source_order, identity_order, 3u,
        0x101u, 0x202u};
    transpose_cover_requirements_v1 requirements{};
    require(query_transpose_cover_requirements_v1(input, &requirements)
        == transpose_status_v1::success);
    require(requirements.placement_count == 3u && requirements.owner_count == 2u);
    transpose_edge_placement_v1 placements[3]{};
    source_owner_schedule_v1 owners[2]{};
    transpose_cover_view_v1 cover{};
    require(build_transpose_cover_v1(input, {placements, 3u, owners, 2u}, &cover)
        == transpose_status_v1::success);
    require(cover.owners[0].placement_count == 2u
        && cover.owners[1].placement_begin == 2u
        && cover.placements[0].global_source_id > 0xffffffffu);

    std::uint64_t duplicate_identity_order[]{1u, 1u, 0u};
    input.identity_order = duplicate_identity_order;
    require(query_transpose_cover_requirements_v1(input, &requirements)
        != transpose_status_v1::success);
    input.identity_order = identity_order;
    input.transpose_cover_id = input.forward_cover_id;
    require(query_transpose_cover_requirements_v1(input, &requirements)
        == transpose_status_v1::invalid_argument);
    return 0;
}
