#include <Cellerator/compiler/sema/field/implement_statement_ordering_and_observable_effects_v1.hh>

#include <array>
#include <iostream>

namespace field = Cellerator::compiler::sema::field;

int main() {
    field::field_statement_semantics_v1 add{1, {1}, {2}, {}, {}, 0, 9, 7};
    field::field_statement_semantics_v1 scale{2, {3}, {4}, {}, {}, 0, 9, 7};
    const auto independent =
        field::implement_statement_ordering_and_observable_effects_v1(add, scale);
    if (!independent.reorder_permitted() || !independent.fusion_permitted()) {
        std::cerr << "independent equivalent statements were blocked\n";
        return 1;
    }

    const auto execute = [](bool reordered) {
        std::array<int, 5> value{0, 3, 0, 5, 0};
        const auto first = [&] { value[2] = value[1] + 4; };
        const auto second = [&] { value[4] = value[3] * 2; };
        if (reordered) { second(); first(); } else { first(); second(); }
        return value;
    };
    if (execute(false) != execute(true)) {
        std::cerr << "permitted reordering changed execution\n";
        return 1;
    }

    scale.reads = {2};
    if (field::implement_statement_ordering_and_observable_effects_v1(add, scale)
            .reorder_blocker != field::ordering_blocker_v1::data_dependency) {
        return 1;
    }
    scale.reads = {3};
    scale.observable_effects = field::field_effect_io_v1;
    if (field::implement_statement_ordering_and_observable_effects_v1(add, scale)
            .reorder_blocker != field::ordering_blocker_v1::observable_effect) {
        return 1;
    }
    scale.observable_effects = 0;
    scale.numerical_contract_id = 10;
    const auto numeric =
        field::implement_statement_ordering_and_observable_effects_v1(add, scale);
    if (!numeric.reorder_permitted() || numeric.fusion_blocker !=
        field::ordering_blocker_v1::numerical_contract) {
        return 1;
    }
    scale.numerical_contract_id = 9;
    add.generation_writes = {{8, 2}};
    scale.generation_reads = {{8, 1}};
    if (field::implement_statement_ordering_and_observable_effects_v1(add, scale)
            .reorder_blocker != field::ordering_blocker_v1::generation_dependency) {
        return 1;
    }
    return 0;
}
