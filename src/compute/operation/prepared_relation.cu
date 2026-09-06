#include <Cellerator/compute/operation/prepared_relation.hh>
#include <Cellerator/compute/operation/operation_core.hh>

namespace cellerator::compute::relation {
namespace {
namespace core = cellerator::compute::math::core;
// Persistent identities remain value-owned. Compact runtime handles are local
// registry slots, never truncated low words of biological identities.
struct native_contract {
    operation_descriptor semantic{};
    core::numeric_policy numeric{};
    core::structure_set_key structures{};
    core::operation_problem problem{};
    execution::axis_identity source{{1,1},{1,1},{1,1},{1,1}};
    execution::axis_identity destination{{2,1},{2,1},{2,1},{2,1}};
    execution::axis_identity column{{3,1},{3,1},{3,1},{3,1}};
};
status adapt(const operation_descriptor& op, native_contract& out) noexcept {
    auto result = validate(op);
    if (!result) return result;
    if (op.dense_width != 1)
        return {status_code::unsupported_width, "native pair supports N1 only"};
    const auto& a = op.arithmetic;
    if (a.relation_storage != execution::numeric_type::f16
        || a.input_storage != execution::numeric_type::f32
        || a.multiply != execution::numeric_type::f32
        || a.accumulation != execution::numeric_type::f32
        || a.output_storage != execution::numeric_type::f32
        || !a.permit_fma || !a.permit_reassociation
        || a.nonfinite != nonfinite_policy::propagate)
        return {status_code::unsupported_numeric_policy, "FMP1/CTP1 require f16/f32 and permitted FMA/reassociation with propagation"};
    if (op.update != output_update::overwrite || op.input_output_aliasing_legal)
        return {status_code::unsupported_semantics, "native pair requires nonaliasing overwrite"};
    out = {};
    out.semantic = op;
    out.numeric.sparse_storage = a.relation_storage;
    out.numeric.dense_storage = a.input_storage;
    out.numeric.multiply = a.multiply;
    out.numeric.accumulation = a.accumulation;
    out.numeric.output_storage = a.output_storage;
    out.numeric.scalar = execution::numeric_type::f32;
    out.structures.count = 1;
    out.structures.structures[0] = {op.topology.identity, {1,1}, op.topology.epoch};
    out.problem.operation = {1, op.direction == orientation::forward ? 1u : 2u};
    out.problem.input_count = out.problem.output_count = 1;
    out.problem.logical_work_items = op.topology.edge_count;
    return {};
}
} // namespace
} // namespace cellerator::compute::relation
