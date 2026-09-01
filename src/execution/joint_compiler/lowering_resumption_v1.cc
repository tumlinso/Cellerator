#include <Cellerator/execution/joint_compiler/lowering_resumption_v1.hh>

namespace cellerator::execution::joint_compiler {
namespace {

lowering_resumption_validation_result_v1 failure(
    lowering_resumption_validation_code_v1 code,
    std::uint64_t index = 0u) noexcept {
    return {code, index};
}

bool valid_stage(lowering_stage_v1 stage) noexcept {
    return stage >= lowering_stage_v1::canonical
        && stage <= lowering_stage_v1::resident;
}

std::uint8_t stage_value(lowering_stage_v1 stage) noexcept {
    return static_cast<std::uint8_t>(stage);
}

}  // namespace

lowering_resumption_validation_result_v1 validate_lowering_resumption_v1(
    const lowering_resumption_v1 &resumption) noexcept {
    if (resumption.schema_version != lowering_resumption_schema_version_v1)
        return failure(
            lowering_resumption_validation_code_v1::unsupported_schema);
    if (resumption.record_bytes != sizeof(lowering_resumption_v1))
        return failure(
            lowering_resumption_validation_code_v1::invalid_record_bytes);
    for (std::uint8_t value : resumption.reserved)
        if (value != 0u)
            return failure(
                lowering_resumption_validation_code_v1::nonzero_reserved);
    if (!validate_persistent_identity_v1(resumption.resumption_identity))
        return failure(lowering_resumption_validation_code_v1::
            invalid_resumption_identity);
    if (!valid_stage(resumption.source_stage))
        return failure(
            lowering_resumption_validation_code_v1::invalid_source_stage);
    if (!valid_stage(resumption.target_stage))
        return failure(
            lowering_resumption_validation_code_v1::invalid_target_stage);
    if (stage_value(resumption.source_stage)
        >= stage_value(resumption.target_stage))
        return failure(
            lowering_resumption_validation_code_v1::invalid_stage_order);
    if (resumption.fallback < lowering_fallback_v1::reject
        || resumption.fallback
            > lowering_fallback_v1::rebuild_from_nearest_compatible)
        return failure(
            lowering_resumption_validation_code_v1::invalid_fallback);
    if (resumption.stage_record_count == 0u
        || resumption.stage_records == nullptr)
        return failure(
            lowering_resumption_validation_code_v1::missing_stage_records);
    if (resumption.stage_record_count > lowering_stage_count_v1)
        return failure(
            lowering_resumption_validation_code_v1::too_many_stage_records);

    bool found_source = false;
    for (std::uint64_t index = 0u; index < resumption.stage_record_count;
         ++index) {
        const lowering_stage_record_v1 &record = resumption.stage_records[index];
        for (std::uint8_t value : record.reserved)
            if (value != 0u)
                return failure(lowering_resumption_validation_code_v1::
                    nonzero_reserved, index);
        if (!valid_stage(record.stage)
            || record.compatibility < lowering_compatibility_v1::compatible
            || record.compatibility > lowering_compatibility_v1::incompatible)
            return failure(lowering_resumption_validation_code_v1::
                invalid_stage_record, index);
        if (index != 0u && stage_value(record.stage)
            <= stage_value(resumption.stage_records[index - 1u].stage))
            return failure(lowering_resumption_validation_code_v1::
                duplicate_or_unordered_stage_record, index);
        if (!validate_persistent_identity_v1(record.artifact_identity))
            return failure(lowering_resumption_validation_code_v1::
                invalid_artifact_identity, index);
        if (!validate_persistent_identity_v1(record.artifact_schema))
            return failure(lowering_resumption_validation_code_v1::
                invalid_artifact_schema, index);
        if (!validate_persistent_identity_v1(record.validation_receipt))
            return failure(lowering_resumption_validation_code_v1::
                invalid_validation_receipt, index);
        if (record.stage == resumption.source_stage
            && record.compatibility == lowering_compatibility_v1::compatible)
            found_source = true;
    }
    if (!found_source)
        return failure(
            lowering_resumption_validation_code_v1::source_record_missing);

    if (resumption.bypassed_stage_count == 0u) {
        if (resumption.bypassed_stages != nullptr)
            return failure(lowering_resumption_validation_code_v1::
                inconsistent_bypass_pointer);
        return {};
    }
    if (resumption.bypassed_stages == nullptr)
        return failure(lowering_resumption_validation_code_v1::
            inconsistent_bypass_pointer);
    for (std::uint64_t index = 0u; index < resumption.bypassed_stage_count;
         ++index) {
        const lowering_stage_v1 stage = resumption.bypassed_stages[index];
        if (!valid_stage(stage)
            || stage_value(stage) <= stage_value(resumption.source_stage)
            || stage_value(stage) >= stage_value(resumption.target_stage))
            return failure(lowering_resumption_validation_code_v1::
                invalid_bypassed_stage, index);
        if (index != 0u && stage_value(stage)
            <= stage_value(resumption.bypassed_stages[index - 1u]))
            return failure(lowering_resumption_validation_code_v1::
                duplicate_or_unordered_bypassed_stage, index);
        const lowering_stage_record_v1 *match = nullptr;
        for (std::uint64_t record_index = 0u;
             record_index < resumption.stage_record_count; ++record_index)
            if (resumption.stage_records[record_index].stage == stage)
                match = &resumption.stage_records[record_index];
        if (match == nullptr)
            return failure(lowering_resumption_validation_code_v1::
                bypass_record_missing, index);
        if (match->compatibility != lowering_compatibility_v1::compatible)
            return failure(lowering_resumption_validation_code_v1::
                bypass_not_compatible, index);
    }
    return {};
}

}  // namespace cellerator::execution::joint_compiler
