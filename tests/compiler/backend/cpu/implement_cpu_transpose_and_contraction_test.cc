#include <Cellerator/compiler/backend/implement_cpu_transpose_and_contraction_v1.hh>

#include <cassert>

namespace cb = cellerator::compiler::backend::v1;
namespace cp = cellerator::compute::projection_family;

int main() {
    const std::uint64_t offsets[]{0, 2, 4};
    const std::uint64_t sources[]{0, 2, 1, 2};
    const std::uint64_t logical[]{2, 0, 3, 1};
    cp::forward_relation_apply_view_v1 projection{};
    projection.destination_offsets = offsets;
    projection.source_indices = sources;
    projection.logical_edge_ids = logical;
    projection.source_count = 3;
    projection.destination_count = 2;
    projection.logical_edge_count = 4;

    const float weights[]{4, 5, 2, 3};
    const float destination[]{10, 20, 30, 40};
    float transpose[6]{};
    cb::cpu_transpose_request_v1 transpose_request{};
    transpose_request.projection = projection;
    transpose_request.relation_values = weights;
    transpose_request.destination_values = destination;
    transpose_request.source_output = transpose;
    transpose_request.dense_width = 2;
    transpose_request.accumulation = cb::cpu_accumulation_v1::f64;
    assert(cb::apply_cpu_relation_transpose_v1(transpose_request)
        == cb::cpu_transpose_status_v1::success);
    const float transpose_reference[]{20, 40, 90, 120, 190, 280};
    for (int i = 0; i < 6; ++i) assert(transpose[i] == transpose_reference[i]);

    const float source[]{1, 2, 3, 4, 5, 6};
    float gradients[4]{};
    cb::cpu_edge_contraction_request_v1 contraction{};
    contraction.projection = projection;
    contraction.source_values = source;
    contraction.destination_values = destination;
    contraction.logical_edge_output = gradients;
    contraction.dense_width = 2;
    assert(cb::contract_cpu_relation_support_v1(contraction)
        == cb::cpu_transpose_status_v1::success);
    const float gradient_reference[]{170, 390, 50, 250};
    for (int i = 0; i < 4; ++i) assert(gradients[i] == gradient_reference[i]);

    const float first[]{1, 2, 3};
    const float second[]{4, 5, 6};
    const float* partials[]{first, second};
    float merged[3]{};
    assert(cb::merge_cpu_partials_v1({partials, 2, 3, merged,
               cb::cpu_accumulation_v1::f64})
        == cb::cpu_transpose_status_v1::success);
    assert(merged[0] == 5 && merged[1] == 7 && merged[2] == 9);
}
