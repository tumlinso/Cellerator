#include <Cellerator/compiler/ir/semantic/implement_gradient_and_publication_operations_v1.hh>

#include <algorithm>
#include <unordered_set>

namespace Cellerator::compiler::ir::semantic {

gradient_publication_status_ir_v1 validate_gradient_publication_program_ir_v1(
    const gradient_publication_program_ir_v1& program) noexcept {
    if (program.program_identity == 0 || !program.structure_identity.valid() ||
        program.structure_epoch == 0)
        return gradient_publication_status_ir_v1::invalid_identity;
    if (program.prepared_generation == 0)
        return gradient_publication_status_ir_v1::invalid_generation;
    if (validate_numeric_tuple_ir_v1(program.numerical) !=
        state_value_ir_validation_code_v1::success)
        return gradient_publication_status_ir_v1::invalid_numerical_policy;
    if (!program.update_policy.owned_by_caller ||
        program.update_policy.caller_policy_identity == 0 ||
        program.update_policy.prepared_update_candidate_identity == 0)
        return gradient_publication_status_ir_v1::update_policy_not_caller_owned;
    bool forward = false;
    bool transpose = false;
    bool value_gradient = false;
    bool publication = false;
    std::unordered_set<std::uint64_t> identities;
    for (const auto& stage : program.stages) {
        if (stage.identity == 0 || !identities.insert(stage.identity).second ||
            !stage.input_axis.valid() || !stage.output_axis.valid())
            return gradient_publication_status_ir_v1::invalid_stage;
        forward |= stage.kind == gradient_publication_operation_ir_v1::forward;
        transpose |= stage.kind == gradient_publication_operation_ir_v1::transpose;
        value_gradient |= stage.kind == gradient_publication_operation_ir_v1::value_gradient;
        if (stage.kind == gradient_publication_operation_ir_v1::publish_generation) {
            if (stage.consumed_generation == 0 ||
                stage.published_generation <= stage.consumed_generation)
                return gradient_publication_status_ir_v1::invalid_generation;
            publication = true;
        }
        if (stage.kind == gradient_publication_operation_ir_v1::canonicalize &&
            !stage.explicit_order_transform)
            return gradient_publication_status_ir_v1::invalid_canonicalization;
    }
    return forward && transpose && value_gradient && publication
        ? gradient_publication_status_ir_v1::success
        : gradient_publication_status_ir_v1::incomplete_gradient_closure;
}

gradient_publication_status_ir_v1 compare_gradient_program_with_training_v2(
    const gradient_publication_program_ir_v1& semantic,
    const cellerator::execution::training_v2::training_program_v2& training) noexcept {
    const auto status = validate_gradient_publication_program_ir_v1(semantic);
    if (status != gradient_publication_status_ir_v1::success) return status;
    if (training.schema_version !=
            cellerator::execution::training_v2::training_program_schema_version_v2 ||
        training.program_identity != semantic.program_identity ||
        training.epoch.value != semantic.structure_epoch ||
        training.prepared_generation.value != semantic.prepared_generation ||
        training.stage_count != semantic.stages.size() || training.stages == nullptr)
        return gradient_publication_status_ir_v1::training_contract_mismatch;
    for (std::size_t index = 0; index < semantic.stages.size(); ++index) {
        if (training.stages[index].stage_identity != semantic.stages[index].identity ||
            training.stages[index].kind !=
                lower_gradient_publication_stage_kind_v1(semantic.stages[index].kind))
            return gradient_publication_status_ir_v1::training_contract_mismatch;
    }
    return gradient_publication_status_ir_v1::success;
}

cellerator::execution::training_v2::training_stage_kind_v2
lower_gradient_publication_stage_kind_v1(
    gradient_publication_operation_ir_v1 kind) noexcept {
    using result = cellerator::execution::training_v2::training_stage_kind_v2;
    switch (kind) {
    case gradient_publication_operation_ir_v1::forward: return result::forward_relation_apply;
    case gradient_publication_operation_ir_v1::transpose: return result::transpose_relation_apply;
    case gradient_publication_operation_ir_v1::value_gradient: return result::logical_edge_gradient;
    case gradient_publication_operation_ir_v1::publish_generation:
    case gradient_publication_operation_ir_v1::caller_update_boundary:
        return result::publish_value_generation;
    case gradient_publication_operation_ir_v1::canonicalize:
        return result::explicit_canonicalize;
    }
    return result::forward_relation_apply;
}

}  // namespace Cellerator::compiler::ir::semantic
