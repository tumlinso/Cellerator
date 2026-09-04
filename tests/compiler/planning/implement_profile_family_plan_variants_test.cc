#include <Cellerator/compiler/planning/implement_profile_family_plan_variants_v1.hh>

#include <cassert>
#include <vector>

namespace planning = Cellerator::compiler::planning;

int main() {
    const std::vector<planning::named_profile_plan_v1> alternatives{
        {1u, "small", 100u, 11u, 1000u, 101u},
        {2u, "medium", 100u, 12u, 1000u, 102u},
        {3u, "large", 100u, 13u, 2000u, 103u},
    };
    const auto result = planning::implement_profile_family_plan_variants_v1(
        alternatives, 4u);
    assert(result);
    assert(result.plan.semantic_program_identity == 100u);
    assert(result.plan.variants.size() == 3u);
    assert(result.plan.shared_artifacts.size() == 2u);
    assert(result.plan.shared_artifact_reuses == 1u);
    assert(result.plan.variants[0].shared_artifact_index ==
        result.plan.variants[1].shared_artifact_index);
    assert(result.plan.variants[2].shared_artifact_index !=
        result.plan.variants[0].shared_artifact_index);

    assert(planning::implement_profile_family_plan_variants_v1(alternatives, 2u).code ==
        planning::profile_family_plan_code_v1::runtime_variant_limit_exceeded);
    auto mismatch = alternatives;
    mismatch[2].semantic_program_identity = 999u;
    assert(planning::implement_profile_family_plan_variants_v1(mismatch, 4u).code ==
        planning::profile_family_plan_code_v1::semantic_program_mismatch);
}
