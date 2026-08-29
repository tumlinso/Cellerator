#include <Cellerator/compute/operation/relation_algebra.hh>

#include <limits>

namespace cellerator::compute::operation::compatibility_detail {
namespace {

bool valid_projection(const core::projection_key &projection) noexcept {
    return execution::valid_identity(projection.persistent)
        && execution::valid_handle(projection.runtime)
        && projection.schema_version != 0u;
}

} // namespace

// Transitional implementation seam for CE-GEO-81. CE-GEO-86 owns the public
// catalog declaration; keeping this declaration out of relation_algebra.hh
// prevents an unreviewed public ABI from escaping this compatibility task.
relation_algebra_status_v1 map_relation_apply_to_operation_core_v1(
    const relation_algebra_problem_v1 &typed,
    execution::structure_handle runtime_structure,
    const core::projection_key &projection,
    core::operation_problem *problem,
    core::structure_set_key *structures,
    core::numeric_policy *numeric,
    execution::persistent_axis_identity *input_axis,
    execution::persistent_axis_identity *output_axis) noexcept {
    if (problem == nullptr || structures == nullptr || numeric == nullptr
        || input_axis == nullptr || output_axis == nullptr)
        return relation_algebra_status_v1::invalid_operation;
    const relation_algebra_status_v1 status =
        validate_relation_algebra_problem_v1(typed);
    if (status != relation_algebra_status_v1::ok) return status;
    if (typed.kind != relation_algebra_kind_v1::relation_apply
        && typed.kind != relation_algebra_kind_v1::relation_apply_transpose)
        return relation_algebra_status_v1::invalid_operation;
    if (!execution::valid_handle(runtime_structure)
        || !valid_projection(projection))
        return relation_algebra_status_v1::invalid_identity;

    const bool transpose = typed.kind
        == relation_algebra_kind_v1::relation_apply_transpose;
    if (transpose
            != (projection.kind == core::projection_kind::transpose_or_backward))
        return relation_algebra_status_v1::invalid_operation_semantics;
    if (typed.relation.logical_edge_count == 0u
        || typed.relation.logical_edge_count
        > std::numeric_limits<std::uint64_t>::max() / typed.dense_width)
        return relation_algebra_status_v1::invalid_operation_semantics;

    core::operation_problem mapped_problem{};
    mapped_problem.schema_version = core::operation_core_schema_version;
    // Both current forward SpMM and v1 transpose/backward are frozen as
    // sparse_dense_multiply. Orientation remains a typed-algebra and
    // projection property; the v1 enum is never reinterpreted.
    mapped_problem.kind = core::operation_kind::sparse_dense_multiply;
    mapped_problem.operation = typed.operation_identity;
    mapped_problem.input_count = 1u;
    mapped_problem.output_count = 1u;
    mapped_problem.logical_work_items =
        typed.relation.logical_edge_count * typed.dense_width;

    core::structure_set_key mapped_structures{};
    mapped_structures.count = 1u;
    mapped_structures.structures[0] = {typed.relation.structure,
        runtime_structure, typed.relation.epoch};

    core::numeric_policy mapped_numeric{};
    mapped_numeric.sparse_storage = typed.numeric.relation_storage;
    mapped_numeric.dense_storage = typed.numeric.state_storage;
    mapped_numeric.output_storage = typed.numeric.output_storage;
    mapped_numeric.multiply = typed.numeric.multiply;
    mapped_numeric.accumulation = typed.numeric.accumulation;
    mapped_numeric.scalar = typed.numeric.scalar;
    mapped_numeric.rounding = typed.numeric.rounding;
    mapped_numeric.saturation = typed.numeric.saturation;
    mapped_numeric.quantization = core::quantization_granularity::none;

    *problem = mapped_problem;
    *structures = mapped_structures;
    *numeric = mapped_numeric;
    *input_axis = transpose ? typed.relation.destination_axis
                            : typed.relation.source_axis;
    *output_axis = transpose ? typed.relation.source_axis
                             : typed.relation.destination_axis;
    return relation_algebra_status_v1::ok;
}

} // namespace cellerator::compute::operation::compatibility_detail
