#include <Cellerator/compiler/sema/implement_structure_value_and_support_generation_typing_v1.hh>

namespace cellerator::compiler::sema::v1 {

generation_validation validate_generations(
    const generation_requirement &requirement,
    const generation_state &actual,
    publication_state publication) noexcept {
    if (requirement.expected.structure_epoch != actual.structure_epoch)
        return generation_validation::stale_structure;
    if (requirement.expected.value_generation != actual.value_generation)
        return generation_validation::stale_values;
    if (requirement.expected.active_support_generation != actual.active_support_generation)
        return generation_validation::stale_active_support;
    if (requirement.expected.order_generation != actual.order_generation)
        return generation_validation::stale_order;
    if (requirement.required_publication == publication_state::published
        && publication != publication_state::published)
        return generation_validation::unpublished;
    return generation_validation::ok;
}

bool generation_override_allows(generation_validation failure,
                                const expert_generation_override &override) noexcept {
    return failure != generation_validation::ok && override.explicitly_unsafe
        && override.permitted_mismatch == failure;
}

}  // namespace cellerator::compiler::sema::v1
