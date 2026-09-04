#include <Cellerator/compiler/sema/field/define_execution_field_semantic_ownership_v1.hh>

#include <iostream>

namespace field = Cellerator::compiler::sema::field;
namespace source = Cellerator::compiler::frontend::source;

int main() {
    field::execution_field_definition_v1 definition;
    definition.stable_source_name = "model.cell";
    definition.explicit_field_name = "regulatory_step";
    definition.source = {{7, 100}, {7, 220}};
    definition.captured_values = {
        {"expression", 11, field::captured_value_access_v1::read},
        {"response", 12, field::captured_value_access_v1::write},
    };
    definition.observable_boundaries = {
        {{{7, 160}, {7, 175}}, field::field_effect_io_v1},
    };
    definition.profile_environment = {"pbmc3k-v1", 0x11, 0x22};
    definition.semantic_effects = field::field_effect_reads_memory_v1 |
        field::field_effect_writes_memory_v1;

    field::execution_field_semantics_v1 semantics;
    if (field::define_execution_field_semantic_ownership_v1(
            definition, &semantics) !=
            field::execution_field_definition_status_v1::success ||
        (semantics.identity.low == 0 && semantics.identity.high == 0) ||
        semantics.captured_values.size() != 2 ||
        !semantics.profile_environment.bound() ||
        (semantics.semantic_effects & field::field_effect_io_v1) == 0) {
        std::cerr << "execution-field ownership was not preserved\n";
        return 1;
    }

    const source::source_span_v1 inside{{7, 120}, {7, 150}};
    const source::source_span_v1 crosses_exit{{7, 210}, {7, 230}};
    const source::source_span_v1 before{{7, 20}, {7, 40}};
    const source::source_span_v1 other_file{{8, 120}, {8, 150}};
    if (!field::execution_field_owns_operation_v1(semantics, inside) ||
        field::execution_field_owns_operation_v1(semantics, crosses_exit) ||
        field::execution_field_owns_operation_v1(semantics, before) ||
        field::execution_field_owns_operation_v1(semantics, other_file)) {
        std::cerr << "an operation outside the lexical field was absorbed\n";
        return 1;
    }

    auto repeated = semantics;
    if (field::define_execution_field_semantic_ownership_v1(
            definition, &repeated) !=
            field::execution_field_definition_status_v1::success ||
        !(repeated.identity == semantics.identity)) {
        std::cerr << "source identity is not stable\n";
        return 1;
    }

    definition.observable_boundaries.push_back(
        {{{7, 219}, {7, 221}}, field::field_effect_synchronizes_v1});
    if (field::define_execution_field_semantic_ownership_v1(
            definition, &semantics) !=
        field::execution_field_definition_status_v1::boundary_outside_field) {
        std::cerr << "out-of-field observable boundary was accepted\n";
        return 1;
    }
    return 0;
}
