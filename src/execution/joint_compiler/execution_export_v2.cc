#include <Cellerator/profiling/joint_compiler/execution_export_v2.hh>

namespace cellerator::profiling::joint_compiler {
namespace {

using identity = execution::joint_compiler::persistent_identity_v1;

execution_export_validation_result_v2 failure(
    execution_export_validation_code_v2 code,
    std::uint64_t index = 0u,
    std::uint64_t nested = 0u) noexcept {
    return {code, index, nested};
}

bool valid_id(identity value) noexcept {
    return static_cast<bool>(
        execution::joint_compiler::validate_persistent_identity_v1(value));
}

bool zero_id(identity value) noexcept {
    return value.producer_namespace == 0u && value.local_identity == 0u;
}

bool less_id(identity lhs, identity rhs) noexcept {
    return lhs.producer_namespace < rhs.producer_namespace
        || (lhs.producer_namespace == rhs.producer_namespace
            && lhs.local_identity < rhs.local_identity);
}

bool less_order(execution::order_id lhs, execution::order_id rhs) noexcept {
    return lhs.high < rhs.high || (lhs.high == rhs.high && lhs.low < rhs.low);
}

}  // namespace

execution_export_validation_result_v2 validate_execution_export_v2(
    const execution_export_v2 &value) noexcept {
    if (value.schema_version != execution_export_schema_version_v2)
        return failure(execution_export_validation_code_v2::unsupported_schema);
    if (value.record_bytes != sizeof(execution_export_v2))
        return failure(
            execution_export_validation_code_v2::invalid_record_bytes);
    for (std::uint8_t item : value.reserved)
        if (item != 0u)
            return failure(
                execution_export_validation_code_v2::nonzero_reserved);
    if (!valid_id(value.export_identity))
        return failure(
            execution_export_validation_code_v2::invalid_export_identity);
    if (validate_generic_execution_export_v1(value.compatibility_v1)
        != export_status::success)
        return failure(
            execution_export_validation_code_v2::invalid_v1_compatibility);
    if (value.exact_coverage_count == 0u || value.exact_coverages == nullptr)
        return failure(execution_export_validation_code_v2::missing_coverages);
    for (std::uint64_t index = 0u; index < value.exact_coverage_count; ++index) {
        if (!execution::joint_compiler::validate_logical_coverage_v1(
                value.exact_coverages[index]))
            return failure(
                execution_export_validation_code_v2::invalid_coverage, index);
        if (index != 0u && !less_id(
                value.exact_coverages[index - 1u].coverage_identity,
                value.exact_coverages[index].coverage_identity))
            return failure(execution_export_validation_code_v2::
                duplicate_or_unordered_coverage, index);
    }
    if (value.decomposition == nullptr
        || !compute::decomposition::validate_decomposition_portfolio_v1(
            *value.decomposition))
        return failure(
            execution_export_validation_code_v2::invalid_decomposition);
    if (value.requirement_count == 0u || value.requirements == nullptr)
        return failure(
            execution_export_validation_code_v2::missing_requirements);
    for (std::uint64_t index = 0u; index < value.requirement_count; ++index) {
        if (!execution::joint_compiler::validate_atom_requirement_v1(
                value.requirements[index]))
            return failure(
                execution_export_validation_code_v2::invalid_requirement, index);
        if (index != 0u && !less_id(
                value.requirements[index - 1u].requirement_identity,
                value.requirements[index].requirement_identity))
            return failure(execution_export_validation_code_v2::
                duplicate_or_unordered_requirement, index);
    }
    if (value.affordance_count == 0u || value.affordances == nullptr)
        return failure(
            execution_export_validation_code_v2::missing_affordances);
    for (std::uint64_t index = 0u; index < value.affordance_count; ++index) {
        if (!execution::joint_compiler::validate_atom_affordance_v1(
                value.affordances[index]))
            return failure(
                execution_export_validation_code_v2::invalid_affordance, index);
        if (index != 0u && !less_id(
                value.affordances[index - 1u].affordance_identity,
                value.affordances[index].affordance_identity))
            return failure(execution_export_validation_code_v2::
                duplicate_or_unordered_affordance, index);
    }
    if (value.partial_algebra_count == 0u) {
        if (value.partial_algebras != nullptr)
            return failure(execution_export_validation_code_v2::
                inconsistent_partial_algebra_pointer);
    } else {
        if (value.partial_algebras == nullptr)
            return failure(execution_export_validation_code_v2::
                inconsistent_partial_algebra_pointer);
        for (std::uint64_t index = 0u; index < value.partial_algebra_count;
             ++index) {
            if (!compute::decomposition::validate_partial_result_algebra_v1(
                    value.partial_algebras[index]))
                return failure(execution_export_validation_code_v2::
                    invalid_partial_algebra, index);
            if (index != 0u && !less_id(
                    value.partial_algebras[index - 1u].algebra_identity,
                    value.partial_algebras[index].algebra_identity))
                return failure(execution_export_validation_code_v2::
                    duplicate_or_unordered_partial_algebra, index);
        }
    }
    if (value.persistent_order_count == 0u || value.persistent_orders == nullptr)
        return failure(execution_export_validation_code_v2::missing_orders);
    for (std::uint64_t index = 0u; index < value.persistent_order_count;
         ++index) {
        if (!execution::valid_identity(value.persistent_orders[index]))
            return failure(
                execution_export_validation_code_v2::invalid_order, index);
        if (index != 0u && !less_order(value.persistent_orders[index - 1u],
                value.persistent_orders[index]))
            return failure(execution_export_validation_code_v2::
                duplicate_or_unordered_order, index);
    }
    if (value.candidate_frontier == nullptr
        || !execution::joint_compiler::validate_atom_fragment_result_v1(
            *value.candidate_frontier))
        return failure(execution_export_validation_code_v2::
            invalid_candidate_frontier);
    if (value.stage_count == 0u || value.stages == nullptr)
        return failure(execution_export_validation_code_v2::missing_stages);
    for (std::uint64_t index = 0u; index < value.stage_count; ++index) {
        const atom_execution_stage_v2 &stage = value.stages[index];
        if (!valid_id(stage.stage_identity) || !valid_id(stage.candidate_identity)
            || !valid_id(stage.input_coverage)
            || !valid_id(stage.output_coverage) || stage.launch_count == 0u
            || (stage.dependency_count != 0u && stage.dependencies == nullptr))
            return failure(
                execution_export_validation_code_v2::invalid_stage, index);
        if (index != 0u
            && !less_id(value.stages[index - 1u].stage_identity,
                stage.stage_identity))
            return failure(execution_export_validation_code_v2::
                duplicate_or_unordered_stage, index);
        for (std::uint32_t dependency = 0u;
             dependency < stage.dependency_count; ++dependency) {
            if (stage.dependencies[dependency] >= index
                || (dependency != 0u && stage.dependencies[dependency]
                    <= stage.dependencies[dependency - 1u]))
                return failure(execution_export_validation_code_v2::
                    invalid_stage_dependency, index, dependency);
        }
    }
    if (value.complete_cost.execution_ns == 0u
        || value.complete_cost.expected_reuse == 0u)
        return failure(
            execution_export_validation_code_v2::invalid_complete_cost);
    if (value.correctness < correctness_compatibility_v2::unverified
        || value.correctness > correctness_compatibility_v2::verified_incompatible)
        return failure(
            execution_export_validation_code_v2::invalid_correctness);
    if (value.correctness == correctness_compatibility_v2::unverified) {
        if (!zero_id(value.correctness_receipt))
            return failure(execution_export_validation_code_v2::
                invalid_correctness_receipt);
    } else if (!valid_id(value.correctness_receipt)) {
        return failure(execution_export_validation_code_v2::
            invalid_correctness_receipt);
    }
    for (std::uint8_t item : value.performance.reserved)
        if (item != 0u)
            return failure(
                execution_export_validation_code_v2::nonzero_reserved);
    if (value.performance.status < performance_freshness_v2::analytical_only
        || value.performance.status > performance_freshness_v2::stale)
        return failure(execution_export_validation_code_v2::
            invalid_performance_freshness);
    const bool analytical =
        value.performance.status == performance_freshness_v2::analytical_only;
    if (analytical) {
        if (!zero_id(value.performance.evidence_identity)
            || !zero_id(value.performance.device_performance_identity)
            || !zero_id(value.performance.build_identity)
            || value.performance.evidence_revision != 0u)
            return failure(execution_export_validation_code_v2::
                invalid_performance_freshness);
    } else if (!valid_id(value.performance.evidence_identity)
        || !valid_id(value.performance.device_performance_identity)
        || !valid_id(value.performance.build_identity)
        || value.performance.evidence_revision == 0u) {
        return failure(execution_export_validation_code_v2::
            invalid_performance_freshness);
    }
    return {};
}

}  // namespace cellerator::profiling::joint_compiler
