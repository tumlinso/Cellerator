#include <Cellerator/compiler/sema/implement_structure_value_and_support_generation_typing_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::sema::v1;
    generation_requirement requirement{{1, 2, 3, 4}, publication_state::published};
    auto actual = requirement.expected;
    assert(validate_generations(requirement, actual, publication_state::published) ==
        generation_validation::ok);
    actual.value_generation = 1;
    const auto stale = validate_generations(requirement, actual, publication_state::published);
    assert(stale == generation_validation::stale_values);
    assert(!generation_override_allows(stale, {false, stale}));
    assert(generation_override_allows(stale, {true, stale}));
    actual = requirement.expected;
    assert(validate_generations(requirement, actual, publication_state::staged) ==
        generation_validation::unpublished);
}
