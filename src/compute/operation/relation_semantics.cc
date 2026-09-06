#include <Cellerator/compute/operation/relation_semantics.hh>

#include <limits>

namespace cellerator::compute::relation {
namespace {
bool same_axis(const axis_descriptor& a, const axis_descriptor& b) noexcept {
    const auto& x = a.identity;
    const auto& y = b.identity;
    return a.extent == b.extent
        && x.header.schema_version == y.header.schema_version
        && x.header.kind == y.header.kind && x.header.byte_count == y.header.byte_count
        && execution::same_identity(x.domain, y.domain)
        && execution::same_identity(x.order, y.order)
        && execution::same_identity(x.geometry, y.geometry)
        && execution::same_identity(x.partition, y.partition);
}
std::uint64_t float_bytes(execution::numeric_type type) noexcept {
    switch (type) {
    case execution::numeric_type::f16:
    case execution::numeric_type::bf16: return 2;
    case execution::numeric_type::f32: return 4;
    case execution::numeric_type::f64: return 8;
    default: return 0;
    }
}
bool fits(std::uint64_t extent, std::uint64_t width, std::uint64_t bytes) noexcept {
    return extent <= std::numeric_limits<std::uint64_t>::max() / width / bytes;
}
} // namespace

status validate(const operation_descriptor& op) noexcept {
    const auto& t = op.topology;
    if (!execution::valid_identity(t.identity) || t.epoch.value == 0
        || !execution::valid_identity(t.logical_edge_order))
        return {status_code::invalid_identity, "topology, epoch and edge order must be identified"};
    if (execution::validate_persistent_axis_identity(t.source.identity)
            != execution::biological_validation_code::ok
        || execution::validate_persistent_axis_identity(t.destination.identity)
            != execution::biological_validation_code::ok)
        return {status_code::invalid_axis, "invalid persistent axis record"};
    if (op.direction != orientation::forward && op.direction != orientation::transpose)
        return {status_code::invalid_argument, "unknown relation orientation"};
    if (op.dense_width == 0 || (t.edge_count != 0
        && (t.source.extent == 0 || t.destination.extent == 0)))
        return {status_code::invalid_shape, "nonempty edges require both axes; width must be nonzero"};
    if (op.update != output_update::overwrite && op.update != output_update::accumulate)
        return {status_code::unsupported_semantics, "update has no defined coefficient contract"};
    const auto& a = op.arithmetic;
    if (!float_bytes(a.relation_storage) || !float_bytes(a.input_storage)
        || !float_bytes(a.multiply) || !float_bytes(a.accumulation)
        || !float_bytes(a.output_storage))
        return {status_code::unsupported_numeric_policy, "relation arithmetic requires floating types"};
    if (a.nonfinite != nonfinite_policy::propagate && a.nonfinite != nonfinite_policy::reject)
        return {status_code::invalid_argument, "unknown nonfinite policy"};
    if (!fits(input_axis(op).extent, op.dense_width, float_bytes(a.input_storage))
        || !fits(result_axis(op).extent, op.dense_width, float_bytes(a.output_storage))
        || !fits(t.edge_count, 1, float_bytes(a.relation_storage)))
        return {status_code::invalid_shape, "logical storage size overflows 64 bits"};
    return {};
}

bool equivalent(const operation_descriptor& x, const operation_descriptor& y) noexcept {
    const auto& a = x.topology;
    const auto& b = y.topology;
    const auto& p = x.arithmetic;
    const auto& q = y.arithmetic;
    return execution::same_identity(a.identity, b.identity)
        && a.epoch.value == b.epoch.value
        && same_axis(a.source, b.source) && same_axis(a.destination, b.destination)
        && execution::same_identity(a.logical_edge_order, b.logical_edge_order)
        && a.edge_count == b.edge_count && x.direction == y.direction
        && p.relation_storage == q.relation_storage && p.input_storage == q.input_storage
        && p.multiply == q.multiply && p.accumulation == q.accumulation
        && p.output_storage == q.output_storage && p.permit_fma == q.permit_fma
        && p.permit_reassociation == q.permit_reassociation && p.nonfinite == q.nonfinite
        && x.dense_width == y.dense_width && x.update == y.update
        && x.input_output_aliasing_legal == y.input_output_aliasing_legal;
}
} // namespace cellerator::compute::relation
