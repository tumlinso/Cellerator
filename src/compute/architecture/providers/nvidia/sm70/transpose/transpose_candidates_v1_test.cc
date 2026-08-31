#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_candidates_v1.hh>

#include <cstdlib>

using namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose;

namespace { void require(bool value) { if (!value) std::abort(); } }

int main() {
    const transpose_candidate_catalog_v1 catalog = query_transpose_candidates_v1();
    require(catalog.candidate_count == 2u);
    require(validate_transpose_candidate_v1(catalog.candidates[0])
        == transpose_status_v1::success);
    require(validate_transpose_candidate_v1(catalog.candidates[1])
        == transpose_status_v1::success);
    require(catalog.candidates[0].kernel_id != catalog.candidates[1].kernel_id);

    transpose_edge_placement_v1 placements[]{
        {1u, 11u, 21u, 0u, 0u, 0u},
        {2u, 11u, 22u, 0u, 1u, 1u}};
    source_owner_schedule_v1 owner{11u, 0u, 2u, 0u, 0u};
    transpose_cover_view_v1 cover{transpose_cover_schema_v1, 0u, 9u, 10u,
        placements, 2u, &owner, 1u};
    float values[]{2.0f, 3.0f};
    float gradient[]{1.0f, 2.0f, 4.0f, 5.0f};
    float output[2]{};
    require(execute_transpose_reference_v1({cover, values, gradient, 2u, 2u,
        output, 2u}) == transpose_status_v1::success);
    require(output[0] == 14.0f && output[1] == 19.0f);
    return 0;
}
