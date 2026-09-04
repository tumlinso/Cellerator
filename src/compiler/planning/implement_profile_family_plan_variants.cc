#include <Cellerator/compiler/planning/implement_profile_family_plan_variants_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::planning {

profile_family_plan_result_v1 implement_profile_family_plan_variants_v1(
    const std::vector<named_profile_plan_v1>& alternatives,
    std::uint64_t maximum_runtime_variants) {
    profile_family_plan_result_v1 result{};
    if (alternatives.empty() || maximum_runtime_variants == 0u) return result;
    if (alternatives.size() > maximum_runtime_variants) {
        result.code = profile_family_plan_code_v1::runtime_variant_limit_exceeded;
        return result;
    }
    result.plan.semantic_program_identity = alternatives[0].semantic_program_identity;
    result.plan.runtime_selection_limit = maximum_runtime_variants;
    for (std::size_t i = 0u; i < alternatives.size(); ++i) {
        result.alternative_index = i;
        const auto& alternative = alternatives[i];
        if (alternative.profile_identity == 0u || alternative.profile_name.empty() ||
            alternative.semantic_program_identity == 0u ||
            alternative.selected_candidate_identity == 0u ||
            alternative.artifact_compatibility_identity == 0u ||
            alternative.runtime_predicate_identity == 0u) return result;
        if (alternative.semantic_program_identity != result.plan.semantic_program_identity) {
            result.code = profile_family_plan_code_v1::semantic_program_mismatch;
            return result;
        }
        if (std::find(result.plan.profile_names.begin(), result.plan.profile_names.end(),
                      alternative.profile_name) != result.plan.profile_names.end()) {
            result.code = profile_family_plan_code_v1::duplicate_profile;
            return result;
        }
        result.plan.profile_names.push_back(alternative.profile_name);
        auto artifact = std::find_if(result.plan.shared_artifacts.begin(),
            result.plan.shared_artifacts.end(), [&](const auto& item) {
                return item.compatibility_identity ==
                    alternative.artifact_compatibility_identity;
            });
        std::uint32_t artifact_index = 0u;
        if (artifact == result.plan.shared_artifacts.end()) {
            artifact_index = static_cast<std::uint32_t>(result.plan.shared_artifacts.size());
            result.plan.shared_artifacts.push_back({
                alternative.artifact_compatibility_identity,
                alternative.semantic_program_identity, 1u});
        } else {
            artifact_index = static_cast<std::uint32_t>(
                artifact - result.plan.shared_artifacts.begin());
            ++artifact->reuse_count;
            ++result.plan.shared_artifact_reuses;
        }
        result.plan.variants.push_back({alternative.profile_identity,
            alternative.selected_candidate_identity, artifact_index,
            alternative.runtime_predicate_identity});
    }
    result.code = profile_family_plan_code_v1::ok;
    return result;
}

}  // namespace Cellerator::compiler::planning
