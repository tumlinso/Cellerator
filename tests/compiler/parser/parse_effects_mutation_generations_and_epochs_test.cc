#include <Cellerator/compiler/frontend/parser/parse_effects_mutation_generations_and_epochs_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::frontend::parser;

int main() {
    const auto parsed = parse_effects_and_transitions_v1(R"cpp(
void update(weights_t& weights) effects(
    reads(expression), mutates(weights), advances(generationof(weights)),
    advances(generationof(state), 42), preserves(structure(weights), orderof(weights)),
    publishes(weights), deterministic);
ce::mutate_values(weights);
ce::mutate_support(active);
ce::mutate_order(axis, packed);
ce::mutate_structure(topology);
ce::publish_generation(weights, next);
ce::end_epoch(topology, differentiation);
ce::advance_epoch(topology, next_epoch);
ce::assert_generation(weights, expected);
ce::rebind_identity(view, identity);
)cpp");
    assert(parsed.accepted());
    assert(parsed.effects.size() == 7);
    assert(parsed.effects[2].generation == generation_mode_v1::automatic);
    assert(parsed.effects[3].generation == generation_mode_v1::explicit_value);
    assert(parsed.transitions.size() == 9);
    assert(parsed.transitions[0].kind == semantic_transition_kind_v1::mutate_structure);

    assert(!parse_effects_and_transitions_v1("void f() effects(reads(x)").accepted());
    assert(!parse_effects_and_transitions_v1("void f() effects(invents(x))").accepted());
    assert(!parse_effects_and_transitions_v1("ce::advance_epoch(topology").accepted());
}
