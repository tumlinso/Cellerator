#include <Cellerator/execution/joint_compiler/lowering_resumption_v1.hh>

#include <cassert>

namespace joint_compiler = cellerator::execution::joint_compiler;

joint_compiler::lowering_stage_record_v1 record(
    joint_compiler::lowering_stage_v1 stage,
    joint_compiler::lowering_compatibility_v1 compatibility,
    std::uint64_t value) {
    joint_compiler::lowering_stage_record_v1 result{};
    result.stage = stage;
    result.compatibility = compatibility;
    result.artifact_identity = {1u, value};
    result.artifact_schema = {2u, value};
    result.validation_receipt = {3u, value};
    return result;
}

int main() {
    joint_compiler::lowering_stage_record_v1 records[] = {
        record(joint_compiler::lowering_stage_v1::semantic_atom_basis,
            joint_compiler::lowering_compatibility_v1::compatible, 1u),
        record(joint_compiler::lowering_stage_v1::target_cover,
            joint_compiler::lowering_compatibility_v1::compatible, 2u),
        record(joint_compiler::lowering_stage_v1::physical_projection,
            joint_compiler::lowering_compatibility_v1::compatible, 3u),
        record(joint_compiler::lowering_stage_v1::packed_operand_value,
            joint_compiler::lowering_compatibility_v1::compatible, 4u)};
    const joint_compiler::lowering_stage_v1 bypassed[] = {
        joint_compiler::lowering_stage_v1::target_cover,
        joint_compiler::lowering_stage_v1::physical_projection,
        joint_compiler::lowering_stage_v1::packed_operand_value};
    joint_compiler::lowering_resumption_v1 resumption{};
    resumption.resumption_identity = {4u, 1u};
    resumption.source_stage =
        joint_compiler::lowering_stage_v1::semantic_atom_basis;
    resumption.target_stage =
        joint_compiler::lowering_stage_v1::executable_recipe;
    resumption.fallback =
        joint_compiler::lowering_fallback_v1::rebuild_from_canonical;
    resumption.stage_records = records;
    resumption.stage_record_count = 4u;
    resumption.bypassed_stages = bypassed;
    resumption.bypassed_stage_count = 3u;
    assert(joint_compiler::validate_lowering_resumption_v1(resumption));

    records[2].compatibility =
        joint_compiler::lowering_compatibility_v1::requires_validation;
    assert(joint_compiler::validate_lowering_resumption_v1(resumption).code
        == joint_compiler::lowering_resumption_validation_code_v1::
            bypass_not_compatible);
    records[2].compatibility =
        joint_compiler::lowering_compatibility_v1::compatible;
    resumption.target_stage = resumption.source_stage;
    assert(joint_compiler::validate_lowering_resumption_v1(resumption).code
        == joint_compiler::lowering_resumption_validation_code_v1::
            invalid_stage_order);
    resumption.target_stage =
        joint_compiler::lowering_stage_v1::executable_recipe;
    resumption.bypassed_stages = nullptr;
    assert(joint_compiler::validate_lowering_resumption_v1(resumption).code
        == joint_compiler::lowering_resumption_validation_code_v1::
            inconsistent_bypass_pointer);
    resumption.bypassed_stage_count = 0u;
    assert(joint_compiler::validate_lowering_resumption_v1(resumption));
    return 0;
}
