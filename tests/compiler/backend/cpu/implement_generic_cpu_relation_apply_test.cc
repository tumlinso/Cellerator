#include <Cellerator/compiler/backend/implement_generic_cpu_relation_apply_v1.hh>

#include <cassert>
#include <cmath>

namespace cb = cellerator::compiler::backend::v1;
namespace co = cellerator::compute::operation;
namespace cp = cellerator::compute::projection_family;
namespace ex = cellerator::execution;

namespace {

ex::persistent_axis_identity axis(std::uint64_t seed) {
    return {{ex::biological_abi_version,
                ex::serialized_record_kind::persistent_axis_identity,
                sizeof(ex::persistent_axis_identity)},
        {seed, 1}, {seed + 1, 1}, {seed + 2, 1}, {seed + 3, 1}};
}

co::relation_algebra_problem_v1 problem() {
    co::relation_algebra_problem_v1 result{};
    result.operation_identity = {1, 2};
    result.relation.structure = {3, 4};
    result.relation.epoch = {1};
    result.relation.source_axis = axis(10);
    result.relation.destination_axis = axis(20);
    result.relation.logical_edge_count = 4;
    result.numeric = {ex::numeric_type::f32, ex::numeric_type::f32,
        ex::numeric_type::f32, ex::numeric_type::f32, ex::numeric_type::f32,
        ex::numeric_type::f32, cellerator::compute::math::core::rounding_policy::nearest_even,
        cellerator::compute::math::core::saturation_policy::none,
        co::nan_policy_v1::reject, {}};
    result.semantic_flags = co::alpha_applied_once | co::beta_applied_once;
    result.dense_width = 1;
    return result;
}

}  // namespace

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
    const float logical_values[]{4, 5, 2, 3};
    const float input[]{10, 20, 30};
    float output[]{7, 11};
    cb::cpu_relation_apply_request_v1 request{};
    request.problem = problem();
    request.projection = projection;
    request.relation_values = logical_values;
    request.input = input;
    request.output = output;
    request.alpha = 0.5F;
    request.beta = 2.0F;
    assert(cb::apply_cpu_relation_v1(request)
        == cb::cpu_relation_apply_status_v1::success);
    assert(output[0] == 84.0F);   // .5 * (2*10 + 4*30) + 2*7
    assert(output[1] == 127.0F);  // .5 * (3*20 + 5*30) + 2*11

    const std::uint64_t canonical_sources[]{2, 0, 1};
    const std::uint64_t canonical_destinations[]{1, 0};
    const float canonical_input[]{20, 30, 10};
    float canonical_output[]{11, 7};
    request.input = canonical_input;
    request.output = canonical_output;
    request.canonical_source_indices = canonical_sources;
    request.canonical_destination_indices = canonical_destinations;
    request.input_order = cb::cpu_relation_apply_order_v1::canonical;
    request.output_order = cb::cpu_relation_apply_order_v1::canonical;
    assert(cb::apply_cpu_relation_v1(request)
        == cb::cpu_relation_apply_status_v1::success);
    assert(canonical_output[0] == 127.0F && canonical_output[1] == 84.0F);

    const float non_finite[]{10, NAN, 30};
    request.input = non_finite;
    request.input_order = cb::cpu_relation_apply_order_v1::projection;
    assert(cb::apply_cpu_relation_v1(request)
        == cb::cpu_relation_apply_status_v1::non_finite_value);
}
